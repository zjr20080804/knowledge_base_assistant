from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from src.model import get_llm
from src.tools import get_weather, create_search_tool

CHAT_NODE = "chat_node"

class RAGState(TypedDict):
    question: str
    answer: str
    image_url: str
    messages: Annotated[list[AnyMessage], add_messages]


CHAT_PROMPT = """你是一个全能的知识库助手。
你可以使用以下工具：
1. get_weather: 查询天气。
2. search_knowledge_base: 查询游戏设定、角色信息等知识库内容。

如果用户问的是常识或闲聊，直接回答；如果涉及游戏内容，必须调用 search_knowledge_base。
如果你看到了图片，请结合图片内容和工具检索到的资料来综合回答。"""


def build_user_message(question: str, image_url: str = "") -> HumanMessage:
    if image_url:
        content = [
            {"type": "text", "text": question},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]
    else:
        content = question
    return HumanMessage(content=content)


def build_rag_graph(retriever):
    llm = get_llm()
    tools = [get_weather, create_search_tool(retriever, llm)]
    tool_node = ToolNode(tools)
    builder = StateGraph(RAGState)

    chat_llm = get_llm(streaming=True)
    chat_llm_with_tools = chat_llm.bind_tools(tools)
    system = SystemMessage(content=CHAT_PROMPT)

    async def chat_node(state: RAGState):
        response = await chat_llm_with_tools.ainvoke([system, *state.get("messages", [])])
        return {"messages": [response]}

    builder.add_node(CHAT_NODE, chat_node)
    builder.add_node("tools", tool_node)
    builder.add_edge(START, CHAT_NODE)
    builder.add_conditional_edges(
        CHAT_NODE,
        tools_condition,
        {
            "tools": "tools",
            END: END
        }
    )
    builder.add_edge("tools", CHAT_NODE)

    return builder