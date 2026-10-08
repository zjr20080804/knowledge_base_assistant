import operator
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.redis import RedisSaver
from sentence_transformers import CrossEncoder
import tiktoken

from src.model import get_llm


class RAGState(TypedDict):
    question: str
    context: str
    context_docs:list
    partial_summaries:list
    answer: str
    messages: Annotated[list, operator.add]


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

ENCODER = tiktoken.get_encoding("cl100k_base")
MAX_TOKENS = 3000

def trim_by_docs(docs: list, max_tokens: int) -> str:
    """按文档累加，不超上限。"""
    result = []
    total = 0
    for doc in docs:
        t = len(ENCODER.encode(doc))
        if total + t > max_tokens:
            break
        result.append(doc)
        total += t
    return "\n\n".join(result)

def build_rag_graph(retriever):
    llm = get_llm()
    reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个知识库助手。以下是检索到的参考资料，请基于资料回答用户问题。"),
        ("system", "【参考资料】\n{context}"),
        MessagesPlaceholder(variable_name="messages"),
        ("human", "{question}")])


    def retrieve_node(state: RAGState):
        # 第 1 步：粗筛，向量检索拿 Top-20
        question = state["question"]
        candidates = retriever.invoke(question)
        print(f"retrieve侯选数：{len(candidates)}")

        # 第 2 步：精排，Rerank 给每个候选打分
        pairs = [(question, doc.page_content) for doc in candidates]
        scores = reranker.predict(pairs)

        # 第 3 步：按分数从高到低排序，取 Top-3
        scored_docs = list(zip(candidates, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        top_docs = [doc for doc, _ in scored_docs[:2]]

        # 第 4 步：格式化后返回
        return {
                "context_docs":[doc.page_content for doc in top_docs]}

    #COMPRESS_PROMPT = """你是一个文本压缩助手,把以下参考资料压缩成简要摘要,保留和问题相关的关键信息,只输出压缩后的摘要,无需生成其他内容
    #问题：{question}
    #参考资料：{raw_context}
#
    #"""
    #def compress_node(state: RAGState):
    #    prompt = ChatPromptTemplate.from_messages([
    #        ("system", COMPRESS_PROMPT)
    #    ])
    #    chain = prompt | llm | StrOutputParser()
    #    compressed_context = chain.invoke({
    #        "question": state["question"],
    #        "raw_context": state["raw_context"],
    #    })
    #    print(f"压缩前字数：{len(state['raw_context'])}")
    #    print(f"压缩后字数：{len(compressed_context)}")
    #    return {"context": compressed_context}

    MAP_PROMPT = """你是一个资料压缩助手,把以下参考资料压缩成简洁摘要,保留关键信息,只输出摘要,不需要输出额外内容
    资料：{context}
    
    """
    REDUCE_PROMPT = """你是一个资料拼接助手,把以下多段简洁摘要整合成一段连贯的资料,保留与问题相关的关键信息,只输出整合后的结果,不需要输出额外内容
    问题：{question}
    摘要：{summaries}
    
    """

    def map_summarize_node(state: RAGState):
        prompt = ChatPromptTemplate.from_messages([("system", MAP_PROMPT)])
        chain = prompt | llm | StrOutputParser()
        summaries = []
        for doc in state["context_docs"]:
            s = chain.invoke({"context": doc})
            summaries.append(s)
        print(f"[map] 共 {len(summaries)} 段摘要")
        return {"partial_summaries": summaries}

    def reduce_summarize_node(state: RAGState):
        summaries_text = "\n\n".join(state["partial_summaries"])
        prompt = ChatPromptTemplate.from_messages([("system", REDUCE_PROMPT)])
        chain = prompt | llm | StrOutputParser()
        final_context = chain.invoke({
            "question": state["question"],
            "summaries": summaries_text,
        })
        print(f"[reduce] 合并后 context 字数：{len(final_context)}")
        return {"context": final_context}

    def generate_node(state: RAGState):
        #context = trim_by_docs(state["context_docs"], MAX_TOKENS)
        #print(f"trim后token数：{len(ENCODER.encode(context))}")
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({
            "context": state["context"],
            "question": state["question"],
            "messages": state["messages"],
        })
        return {"answer": answer}

    def update_history_node(state: RAGState):
        return {"messages": [
            HumanMessage(content=state["question"]),
            AIMessage(content=state["answer"]),
        ]}

    builder = StateGraph(RAGState)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("generate", generate_node)
    builder.add_node("update_history", update_history_node)
    #builder.add_node("compress", compress_node)
    builder.add_node("map_summarize", map_summarize_node)
    builder.add_node("reduce_summarize", reduce_summarize_node)

    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "map_summarize")
    builder.add_edge("map_summarize", "reduce_summarize")
    builder.add_edge("reduce_summarize", "generate")
    builder.add_edge("generate", "update_history")
    builder.add_edge("update_history", END)

    return builder