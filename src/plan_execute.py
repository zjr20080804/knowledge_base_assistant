import operator
from typing import Annotated, TypedDict

from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END

from src.model import get_llm


class PlanState(TypedDict):
    task: str
    plan: list
    past_steps: Annotated[list, operator.add]
    response: str


planner_prompt = """你是一个任务规划器。把用户的任务拆成若干可执行的步骤。

要求：
1. 每个步骤要具体、可独立执行
2. 一般 2~5 步
3. 只输出 JSON，不要其他文字

输出格式（严格 JSON）：
{{"steps": ["步骤1", "步骤2", "步骤3"]}}

用户任务：{task}
"""

execute_prompt = """你是一个步骤执行器。根据提供的参考资料，完成指定的步骤。

步骤：{step}

参考资料：
{context}

要求：
1. 基于参考资料回答这个步骤需要的内容
2. 如果资料不足，如实说明
3. 输出简洁、直接，不要客套
"""

summary_prompt = """你是一个汇总器。根据用户任务和已完成的步骤结果，给出最终回答。

用户任务：{task}

已完成的步骤和结果：
{past_steps_text}

要求：
1. 综合所有步骤的结果，给出一个完整的回答
2. 不要重复罗列步骤，要整合成一段自然的回答
3. 直接输出回答内容，不要客套话
"""

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def format_past_steps(past_steps):
    lines = []
    for i, (step, res) in enumerate(past_steps, 1):
        lines.append(f"步骤{i}：{step}\n结果：{res}")
    return "\n\n".join(lines)

def build_plan_graph(retriever):
    llm = get_llm()

    def planner_node(state: PlanState):
        prompt = ChatPromptTemplate.from_template(planner_prompt)
        chain = prompt | llm | JsonOutputParser()
        result = chain.invoke({"task": state["task"]})
        return {"plan": result["steps"]}

    def executor_node(state: PlanState):
        # 1. 取当前步骤
        current_step = state["plan"][0]

        # 2. 用这个步骤作为 query 检索
        docs = retriever.invoke(current_step)
        context = format_docs(docs)

        # 3. 让 LLM 基于检索结果生成这一步的回答
        prompt = ChatPromptTemplate.from_template(execute_prompt)
        chain = prompt | llm | StrOutputParser()
        step_result = chain.invoke({
            "step": current_step,
            "context": context,
        })

        # 4. 返回：追加 past_steps，移除已执行的 plan
        return {
            "past_steps": [(current_step, step_result)],
            "plan": state["plan"][1:],
        }

    def replanner_node(state: PlanState):
        # 情况 1：还有步骤 → 继续
        if state["plan"]:
            return {"plan": state["plan"]}

        # 情况 2：没步骤了 → 汇总
        prompt = ChatPromptTemplate.from_template(summary_prompt)
        chain = prompt | llm | StrOutputParser()
        response = chain.invoke({
            "task": state["task"],
            "past_steps_text": format_past_steps(state["past_steps"]),
        })
        return {"response": response}

    def should_continue(state: PlanState):
        if state["response"]:
            return "end"
        return "continue"

    builder = StateGraph(PlanState)
    builder.add_node("planner", planner_node)
    builder.add_node("executor", executor_node)
    builder.add_node("replanner", replanner_node)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "executor")
    builder.add_edge("executor", "replanner")
    builder.add_conditional_edges(
        "replanner",
        should_continue,
        {
            "continue": "executor",
            "end": END,
        },
    )

    return builder