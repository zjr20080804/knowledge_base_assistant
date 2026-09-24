import operator
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.redis import RedisSaver

from src.model import get_llm


class RAGState(TypedDict):
    question: str
    context: str
    answer: str
    messages: Annotated[list, operator.add]


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_graph(retriever):
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个知识库助手。以下是检索到的参考资料，请基于资料回答用户问题。"),
        ("system", "【参考资料】\n{context}"),
        MessagesPlaceholder(variable_name="messages"),
        ("human", "{question}"),
    ])

    def retrieve_node(state: RAGState):
        docs = retriever.invoke(state["question"])
        return {"context": format_docs(docs)}

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