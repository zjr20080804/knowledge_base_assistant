import httpx
import asyncio
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import tiktoken

ENCODER = tiktoken.get_encoding("cl100k_base")

@tool
def get_weather(city: str) -> str:
    """
    查询指定城市的实时天气。
    当用户询问天气情况时，请调用此工具。
    参数 city: 城市名称（例如："北京", "Guangzhou"）。
    """
    try:
        # 使用免费的真实天气 API (wttr.in)
        # 加上 ?format=j1 会返回 JSON，方便我们提取信息
        url = f"https://wttr.in/{city}?format=j1"
        response = httpx.get(url, timeout=10)
        data = response.json()

        # 提取当前天气情况
        current = data["current_condition"][0]
        temp_c = current["temp_C"]
        desc = current["weatherDesc"][0]["value"]
        humidity = current["humidity"]

        return f"{city}当前天气：{desc}，气温 {temp_c}℃，湿度 {humidity}%。"
    except Exception as e:
        return f"查询天气失败: {str(e)}"





@tool
def get_weather(city: str) -> str:
    """查询指定城市的实时天气。"""
    try:
        url = f"https://wttr.in/{city}?format=j1"
        response = httpx.get(url, timeout=10)
        data = response.json()
        return f"{city}当前天气：{data['current_condition'][0]['weatherDesc'][0]['value']}，气温 {data['current_condition'][0]['temp_C']}℃。"
    except Exception as e:
        return f"查询天气失败: {str(e)}"


def create_search_tool(retriever, llm):
    """
    使用闭包工厂模式创建检索工具。
    这样 retriever 和 llm 会作为环境变量被内部函数“记住”，无需全局变量。
    """

    @tool
    async def search_knowledge_base(query: str) -> str:
        """
        搜索本地知识库（包含游戏设定、角色信息、玩法规则等）。
        当用户的问题涉及到游戏的具体内容时，必须调用此工具。
        """
        # 1. 检索
        docs = await asyncio.to_thread(retriever.invoke,query)
        if not docs:
            return "知识库中未找到相关信息。"

        # 2. Map 阶段：并发压缩每篇文档
        map_prompt = ChatPromptTemplate.from_template(
            "请将以下文档压缩成精简摘要，保留核心信息。\n文档：\n{context}"
        )
        map_chain = map_prompt | llm | StrOutputParser()

        tasks = [map_chain.ainvoke({"context": doc.page_content}) for doc in docs]
        summaries = await asyncio.gather(*tasks)
        MAX_SUMMARY_TOKENS = 2000

        valid_summaries = []
        total_tokens = 0

        for s in summaries:
            # 计算这段摘要的 Token 数
            t = len(ENCODER.encode(s))
            # 如果加上这段摘要就超了，直接抛弃后面所有摘要（保证语义完整）
            if total_tokens + t > MAX_SUMMARY_TOKENS:
                break
            valid_summaries.append(s)
            total_tokens += t

        if not valid_summaries:
            return "检索到的内容过长，无法生成有效摘要。"

        # 3. Reduce 阶段：整合
        summaries_text = "\n\n".join(valid_summaries)
        reduce_prompt = ChatPromptTemplate.from_template(
            "请将以下摘要整合成一段连贯的资料，只输出资料。\n摘要：\n{summaries}"
        )
        reduce_chain = reduce_prompt | llm | StrOutputParser()
        final_context = await reduce_chain.ainvoke({"summaries": summaries_text})

        return f"【知识库检索与压缩结果】：\n{final_context}"

    return search_knowledge_base
# 将工具放入列表，方便统一管理
tools = [get_weather]