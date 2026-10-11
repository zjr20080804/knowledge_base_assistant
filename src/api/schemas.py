"""
API 请求 / 响应模型

作用：
- 定义客户端发来的 JSON 长什么样（请求）
- 定义服务端返回的 JSON 长什么样（响应）
- FastAPI 用这些模型做自动校验 + 自动生成 /docs
"""

from pydantic import BaseModel, Field


# ============================================================
# 请求模型
# ============================================================

class ChatRequest(BaseModel):
    """
    POST /chat 的请求体

    示例请求：
        {
            "question": "游戏主角是谁？",
            "session_id": "user_001"
        }
    """

    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="用户提问内容",
        examples=["游戏主角是谁？"],
    )

    session_id: str = Field(
        default="default",
        min_length=1,
        max_length=64,
        description="会话 ID，用于隔离不同用户的对话历史",
        examples=["user_001"],
    )


# ============================================================
# 响应模型
# ============================================================

class ChatResponse(BaseModel):
    """
    POST /chat 的响应体

    示例响应：
        {
            "answer": "游戏主角是……",
            "route": "knowledge"
        }
    """

    answer: str = Field(
        ...,
        description="LLM 生成的回答",
    )

    route: str | None = Field(
        default=None,
        description="本次走的路由（knowledge / chat），便于调试",
    )


class HealthResponse(BaseModel):
    """GET /health 的响应体"""

    status: str = Field(
        default="ok",
        description="服务状态",
    )