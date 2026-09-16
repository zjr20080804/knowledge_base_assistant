from datetime import datetime
from langchain_core.tools import tool

@tool
def current_time():
    """当用户输入当前时间时调用"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def calculate(expression: str):
    """当用户需要计算时调用"""
    return str(numexpr.evaluate(expression))

all_tools = [current_time, calculate]
tool_map = {"current_time": current_time, "calculate": calculate}
