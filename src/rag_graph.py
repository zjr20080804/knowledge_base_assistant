from typing import Annotated, TypedDict
from langchain_core.messages import HumanMessage, AnyMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from src.model import get_llm
from src.tools import get_weather, create_search_tool

# ==========================================
# 1. 定义状态
# ==========================================
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


def build_rag_graph(retriever):
    llm = get_llm()
    tools = [get_weather, create_search_tool(retriever, llm)]
    tool_node = ToolNode(tools)
    builder = StateGraph(RAGState)

    async def chat_node(state: RAGState):
        llm = get_llm()
        tools = [get_weather, create_search_tool(retriever, llm)]
        llm_with_tools = llm.bind_tools(tools)

        messages = list(state["messages"])

        is_new_question = True
        for msg in reversed(messages[-5:]):
            if msg.type == "human":
                if isinstance(msg.content, str):
                    if msg.content == state["question"]:
                        is_new_question = False
                elif isinstance(msg.content, list):
                    for part in msg.content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            if part.get("text") == state["question"]:
                                is_new_question = False
                                break
                break

        if is_new_question:
            if state.get("image_url"):
                current_user_msg = HumanMessage(
                    content=[
                        {"type": "text", "text": state["question"]},
                        {"type": "image_url", "image_url": {"url": state["image_url"]}}
                    ]
                )
            else:
                current_user_msg = HumanMessage(content=state["question"])
            messages.append(current_user_msg)

            return_msg = [current_user_msg]
        else:
            return_msg = []

        prompt = ChatPromptTemplate.from_messages([
            ("system", CHAT_PROMPT),
            MessagesPlaceholder(variable_name="all_messages")
        ])

        chain = prompt | llm_with_tools
        response = await chain.ainvoke({"all_messages": messages})

        # 把用户问题和 AI 回复一起返回，保证第一次进图时存入记忆，第二次不重复存
        return {"messages": return_msg + [response]}

    builder.add_node("chat_node", chat_node)
    builder.add_node("tools", tool_node)
    builder.add_edge(START, "chat_node")
    # 核心路由：判断大模型是否要调用工具
    builder.add_conditional_edges(
        "chat_node",
        tools_condition,
        {
            "tools": "tools",
            END: END
        }
    )
    builder.add_edge("tools", "chat_node")

    return builder