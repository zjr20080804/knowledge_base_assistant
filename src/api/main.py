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
from src.api.routers import chat, health
from src.config import REDIS_TTL,REDIS_URL

# ============================================================
# lifespan：启动/关闭逻辑
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理

    yield 之前：启动时执行（只一次）
    yield 之后：关闭时执行（只一次）
    """
    # ---------- 启动 ----------
    print("=" * 60)
    print("🚀 正在启动知识库问答服务...")
    print("=" * 60)

    # 第 1 步：构建 retriever
    print("[lifespan] 加载文档...")
    retriever,chunks= build_retriever()
    print("[lifespan] retriever 构建完成")

    # ---------- 第 2 步：用 AsyncRedisSaver 连接 Redis + 编译图 ----------
    print("[lifespan] 连接 Redis 并编译图...")

    #创建独立的Redis客户端(用于readiness探针)
    redis_client = aioredis.from_url(REDIS_URL,decode_responses = True)
    app.state.redis_client = redis_client

    # ★ async with：让 saver 在整个应用运行期都活着
    async with AsyncRedisSaver.from_conn_string(
        REDIS_URL,
        ttl={"default_ttl": REDIS_TTL, "refresh_on_read": True},
    ) as memory:
        # 异步 setup（初始化 Redis 结构）
        await memory.asetup()

        # 编译图，绑定异步 checkpointer
        graph = build_rag_graph(retriever).compile(checkpointer=memory)

        # 存进 app.state
        app.state.graph = graph
        app.state.retriever = retriever

        print("[lifespan] ✅ 服务启动完成")
        print("=" * 60)

        yield   # ← 服务在这里开始接收请求（yield 必须在 async with 内部！）

    # async with 退出，saver 自动关闭
    await redis_client.aclose()
    print("[lifespan] Redis 连接已关闭")


# ============================================================
# FastAPI 应用
# ============================================================

app = FastAPI(
    title="Knowledge Base Assistant",
    description="基于 LangGraph 的知识库问答服务",
    version="0.1.0",
    lifespan=lifespan,
)


# ============================================================
# 挂载路由
# ============================================================

app.include_router(health.router)
app.include_router(chat.router)