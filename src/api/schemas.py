from pydantic import BaseModel, Field, field_validator



class ChatRequest(BaseModel):

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
    image_url: str = Field(
        default="",
        max_length=8_000_000,
        description="图片的 Base64 Data URL（可不填），必须是 data:image/<type>;base64,<data> 形式",
    )

    @field_validator("image_url")
    @classmethod
    def _check_image_url(cls, v: str) -> str:
        if not v:
            return v
        head = v[:64]
        if not (head.startswith("data:image/") and ";base64," in head):
            raise ValueError("image_url 必须是 data:image/<type>;base64,<data> 形式的 Data URL")
        return v


class ChatResponse(BaseModel):

    answer: str = Field(
        ...,
        description="LLM 生成的回答",
    )

    has_tool_call:bool=Field(
        default=False,
    description="是否调用了工具")


class HealthResponse(BaseModel):

    status: str = Field(
        default="ok",
        description="服务状态",
    )