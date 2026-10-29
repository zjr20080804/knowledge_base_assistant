"""
FastAPI 主入口

职责：
- 创建 FastAPI 应用实例
- 用 lifespan 管理启动/关闭（加载 graph、连接 Redis 等）
- 挂载所有 router
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from langgraph.checkpoint.redis import AsyncRedisSaver   # ← 换成 Async 版本
import redis.asyncio as aioredis

from src.rag_factory import build_retriever
from src.rag_graph import build_rag_graph
from src.api.routers import chat, health,auth
from src.config import REDIS_TTL,REDIS_URL
from src.api.logger import setup_logging,logger
from src.api.exceptions import AppError
from src.api.error_handlers import app_error_handler,unhandled_exception_handler
setup_logging()
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理

    yield 之前：启动时执行（只一次）
    yield 之后：关闭时执行（只一次）
    """

    logger.info("=" * 60)
    logger.info("🚀 正在启动知识库问答服务...")
    logger.info("=" * 60)

    logger.info("[lifespan] 加载文档...")
    retriever,chunks= build_retriever()
    logger.info("[lifespan] retriever 构建完成")

    logger.info("[lifespan] 连接 Redis 并编译图...")

    redis_client = aioredis.from_url(REDIS_URL,decode_responses = True)
    app.state.redis_client = redis_client

    async with AsyncRedisSaver.from_conn_string(
        REDIS_URL,
        ttl={"default_ttl": REDIS_TTL, "refresh_on_read": True},
    ) as memory:
        await memory.asetup()
        graph = build_rag_graph(retriever).compile(checkpointer=memory)

        # 存进 app.state
        app.state.graph = graph
        app.state.retriever = retriever

        logger.info("[lifespan]  服务启动完成")
        logger.info("=" * 60)

        yield
    await redis_client.aclose()
    logger.info("[lifespan] Redis 连接已关闭")


# ============================================================
# FastAPI 应用
# ============================================================

app = FastAPI(
    title="Knowledge Base Assistant",
    description="基于 LangGraph 的知识库问答服务",
    version="0.1.0",
    lifespan=lifespan,
)



app.include_router(health.router)
app.include_router(chat.router)
app.include_router(auth.router)
app.add_exception_handler(AppError, app_error_handler)
app.add_exception_handler(Exception, unhandled_exception_handler)
