import operator
from os import name
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END

from src.model import get_llm


class MultiAgentState(TypedDict):
    task: str
    messages: Annotated[list, operator.add]
    next: str


SUPERVISOR_PROMPT = """你是一个主管，负责协调以下成员完成任务：

- researcher（研究员）：负责检索和整理相关资料
- writer（写作者）：负责根据资料撰写内容
- reviewer（审核员）：负责检查内容质量

根据当前进度，决定下一步派给谁
writer(写作者)输出内容后需要交给reviewer(审核员)审核
reviewer(审核员)审核通过后，回复FINISH
若reviewer(审核员)审核不通过，回复writer

只输出一个词：researcher / writer / reviewer / FINISH

用户任务：{task}

当前进度：
{history}
"""


def build_multi_agent_graph():
    llm = get_llm()

    # ---- 主管节点 ----
    def supervisor_node(state: MultiAgentState):
        history_text = "\n".join(
            f"[{m.name or m.type}]: {m.content[:200]}"
            for m in state["messages"]
        )
        prompt = ChatPromptTemplate.from_template(SUPERVISOR_PROMPT)
        chain = prompt | llm | StrOutputParser()
        decision = chain.invoke({
            "task": state["task"],
            "history": history_text or "（还没开始）",
        }).strip()

        # 只保留合法值
        if decision not in ("researcher", "writer", "reviewer", "FINISH"):
            decision = "FINISH"

        return {"next": decision}

    # ---- 研究员节点 ----
    def researcher_node(state: MultiAgentState):
        prompt = f"你是研究员。针对以下任务检索/整理关键信息，简洁输出：\n\n{state['task']}"
        result = llm.invoke(prompt)
        return {"messages": [AIMessage(content=result.content, name="researcher")]}

    # ---- 写作者节点 ----
    def writer_node(state: MultiAgentState):
        prompt = f"你是写作者。基于现有信息，撰写一段内容：\n\n任务：{state['task']}\n\n现有信息：\n{chr(10).join(m.content for m in state['messages'] if m.name == 'researcher')}"
        result = llm.invoke(prompt)
        return {"messages": [AIMessage(content=result.content, name="writer")]}

    # ---- 审核员节点 ----
    def reviewer_node(state: MultiAgentState):
        writer_message = [m for m in state["messages"] if m.name == "writer"]
        if not writer_message:
            return {"messages": [AIMessage(content="writer未输出内容", name="writer")]}
        writer_msg = writer_message[-1]
        prompt = f"你是审核员。检查以下内容质量，指出问题或确认通过：\n\n任务：{state['task']}\n\n内容：\n{writer_msg.content}"
        result = llm.invoke(prompt)
        return {"messages": [AIMessage(content=result.content, name="reviewer")]}

    # ---- 路由函数 ----
    def route(state: MultiAgentState):
        return state["next"]

    # ---- 构图 ----
    builder = StateGraph(MultiAgentState)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("researcher", researcher_node)
    builder.add_node("writer", writer_node)
    builder.add_node("reviewer", reviewer_node)

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        route,
        {
            "researcher": "researcher",
            "writer": "writer",
            "reviewer": "reviewer",
            "FINISH": END,
        },
    )
    builder.add_edge("researcher", "supervisor")
    builder.add_edge("writer", "supervisor")
    builder.add_edge("reviewer", "supervisor")

    return builder