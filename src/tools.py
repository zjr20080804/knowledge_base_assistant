import httpx
from langchain_core.tools import tool


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


# 将工具放入列表，方便统一管理
tools = [get_weather]