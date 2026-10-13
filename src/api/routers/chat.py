"""
Chat 路由

职责：
- 暴露 POST /chat 接口
- 解析请求 → 调 service → 返回响应
- 不写业务逻辑（业务逻辑在 RagService 里）
"""

from fastapi import APIRouter, Depends

from src.api.deps import get_rag_service
from src.api.schemas import ChatRequest, ChatResponse
from src.api.services.rag_service import RagService

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    service: RagService = Depends(get_rag_service),
) -> ChatResponse:
    """
    知识库问答接口

    请求体：
        {"question": "游戏主角是谁？", "session_id": "user_001"}

    响应体：
        {"answer": "...", "route": "knowledge"}
    """
    # 1. 调 service（业务逻辑）
    result = await service.ask(
        question=req.question,
        thread_id=req.session_id,    # ← 注意：session_id 映射为 thread_id
    )

    # 2. 包装成响应模型
    return ChatResponse(
        answer=result["answer"],
        route=result.get("route"),
    )