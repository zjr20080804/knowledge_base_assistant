import asyncio
import json

from langchain_core.messages import AIMessage

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from src.api.deps import get_graph, get_rate_limiter, get_current_user
from src.api.schemas import ChatRequest, ChatResponse
from src.api.services.rate_limiter import RateLimiter
from src.api.logger import logger
from src.api.exceptions import RagServiceError
from src.rag_graph import CHAT_NODE, build_user_message

router = APIRouter(tags=["chat"])

CHAT_TIMEOUT = 60


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _message_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return "" if content is None else str(content)


def _thread_config(session_id: str) -> dict:
    return {"configurable": {"thread_id": session_id}, "recursion_limit": 20}


@router.post("/chat", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    graph=Depends(get_graph),
    limiter: RateLimiter = Depends(get_rate_limiter),
    current_user: str = Depends(get_current_user),
) -> ChatResponse:
    logger.info(f"用户 {current_user} 提出问题  session_id: {req.session_id}")
    await limiter.check(f"rate_limit:user:{current_user}")

    config = _thread_config(req.session_id)
    initial_state = {"messages": [build_user_message(req.question, req.image_url)]}

    before = await graph.aget_state(config)
    n_before = len((before.values or {}).get("messages") or []) if before else 0

    try:
        result = await asyncio.wait_for(
            graph.ainvoke(initial_state, config), timeout=CHAT_TIMEOUT
        )
    except asyncio.TimeoutError:
        logger.error(f"问答超时 session_id={req.session_id}")
        raise RagServiceError("模型响应超时，请稍后重试")

    messages = result["messages"]
    new_messages = messages[n_before:]
    has_tool_call = any(m.type == "tool" for m in new_messages)

    return ChatResponse(
        answer=_message_text(messages[-1].content),
        has_tool_call=has_tool_call,
    )


@router.post("/chat/stream")
async def chat_stream(
    req: ChatRequest,
    request: Request,
    graph=Depends(get_graph),
    limiter: RateLimiter = Depends(get_rate_limiter),
    current_user: str = Depends(get_current_user),
):
    logger.info(f"用户 {current_user} 发起了流式提问")
    await limiter.check(f"rate_limit:user:{current_user}")

    config = _thread_config(req.session_id)
    initial_state = {"messages": [build_user_message(req.question, req.image_url)]}

    async def event_generator():
        emitted = 0
        try:
            async for chunk, meta in graph.astream(
                initial_state, config=config, stream_mode="messages"
            ):
                if meta.get("langgraph_node") != CHAT_NODE:
                    continue
                if not isinstance(chunk, AIMessage):
                    continue
                if getattr(chunk, "tool_call_chunks", None):
                    continue

                text = _message_text(chunk.content)
                if not text:
                    continue

                if await request.is_disconnected():
                    return
                emitted += 1
                yield _sse({"type": "delta", "content": text})

            if emitted == 0:
                logger.warning(
                    f"流式 0 帧输出 session_id={req.session_id}， "
                    f"检查节点名是否仍为 {CHAT_NODE}、模型是否 streaming=True"
                )
            yield _sse({"type": "done"})
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(f"流式生成失败 session_id={req.session_id}")
            yield _sse({"type": "error", "message": "生成中断，请重试"})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )