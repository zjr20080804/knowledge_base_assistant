"""
Chat 路由

职责：
- 暴露 POST /chat 接口
- 解析请求 → 调 service → 返回响应
- 不写业务逻辑（业务逻辑在 RagService 里）
"""

from fastapi import APIRouter, Depends,Request
from fastapi.responses import StreamingResponse
from src.api.deps import get_rag_service,get_rate_limiter
from src.api.schemas import ChatRequest, ChatResponse
from src.api.services.rag_service import RagService
from src.api.services.rate_limiter import RateLimiter
from src.api.deps import get_current_user
from src.api.logger import logger

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    service: RagService = Depends(get_rag_service),
    limiter: RateLimiter = Depends(get_rate_limiter),
 current_user: str = Depends(get_current_user)
) -> ChatResponse:
    """
    知识库问答接口

    请求体：
        {"question": "游戏主角是谁？", "session_id": "user_001"}

    响应体：
        {"answer": "...", "route": "knowledge"}
    """
    # 1. 调 service（业务逻辑）
    logger.info(f"用户 {current_user} 提出问题  session_id: {req.session_id}")
    user_key = f"rate_limit:user:{req.session_id}"
    await limiter.check(user_key)
    result = await service.ask(
        question=req.question,
        thread_id=req.session_id,    # ← 注意：session_id 映射为 thread_id
    )

    # 2. 包装成响应模型
    return ChatResponse(
        answer=result["answer"],
        route=result.get("route"),
    )


@router.post("/chat/stream")
async def chat_stream(
        req: ChatRequest,
        request: Request,
        limiter: RateLimiter = Depends(get_rate_limiter),
        current_user: str = Depends(get_current_user),
):
    """
    流式问答接口（SSE，打字机效果）
    """
    logger.info(f"用户 {current_user} 发起了流式提问")

    # 1. 同样要检查限流！
    await limiter.check(f"rate_limit:user:{current_user}")

    # 2. 从 app.state 获取 graph
    graph = request.app.state.graph

    # 3. 构造运行配置
    config = {"configurable": {"thread_id": req.session_id}}
    initial_state = {
        "question": req.question,
        "image_url": "",
        "context": "",
        "answer": "",
    }

    # 4. 定义异步生成器
    async def event_generator():
        try:
            # 使用官方推荐的 astream 方法，stream_mode="messages" 可以直接提取大模型的流式输出
            async for msg, metadata in graph.astream(initial_state, config=config, stream_mode="messages"):
                # 过滤出真正的 AI 消息文本（忽略工具调用等）
                node_name = metadata.get("langgraph_node","")
                if node_name in ["chat","generate"]:

                    if msg.content and not isinstance(msg, dict):
                    # SSE 格式要求：data: <内容>\n\n
                        yield f"data: {msg.content}\n\n"
        except Exception as e:
            logger.error(f"流式生成报错: {e}")
            yield f"data: [错误] 生成中断: {str(e)}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    # 5. 返回流式响应，Content-Type 必须是 text/event-stream
    return StreamingResponse(event_generator(), media_type="text/event-stream")