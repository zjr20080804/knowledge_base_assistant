import operator
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.redis import RedisSaver
from sentence_transformers import CrossEncoder

from src.model import get_llm


class RAGState(TypedDict):
    question: str
    context: str
    context_docs:list
    answer: str
    messages: Annotated[list, operator.add]


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_graph(retriever):
    llm = get_llm()
    reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个知识库助手。以下是检索到的参考资料，请基于资料回答用户问题。"),
        ("system", "【参考资料】\n{context}"),
        MessagesPlaceholder(variable_name="messages"),
        ("human", "{question}"),
    ])

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
        return {"context": format_docs(top_docs),
                "context_docs":[doc.page_content for doc in top_docs]}

    def generate_node(state: RAGState):
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

    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", "update_history")
    builder.add_edge("update_history", END)

    return builder