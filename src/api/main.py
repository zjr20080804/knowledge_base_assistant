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
from langgraph.checkpoint.redis import RedisSaver

from src.config import vectorstore_dir
from src.loader_splitter import load_split
from src.embedding import create_vectorstore, load_vectorstore
from src.rag_graph import build_rag_graph
from src.api.routers import chat, health


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
    # 注意：这里和第 2 讲要抽离到 src/rag_factory.py
    print("[lifespan] 加载文档...")
    chunks = load_split()

    print("[lifespan] 加载/创建向量库...")
    if not os.path.exists(vectorstore_dir):
        print("  首次运行，正在创建向量库...")
        vectorstore = create_vectorstore(chunks)
    else:
        print("  加载已有向量库...")
        vectorstore = load_vectorstore()

    from langchain_community.retrievers import BM25Retriever

    vectorstore_retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 6

    # 这里直接 import 你 main_graph.py 里的 hybrid_retrieve 和 HybridRetriever
    # 为了不动原文件，我们在 lifespan 内部重新定义一个轻量 HybridRetriever
    # 第 2 讲会把这些抽到公共模块
    from src.main_graph import hybrid_retrieve

    class HybridRetriever:
        def invoke(self, question):
            return hybrid_retrieve(question, vectorstore_retriever, bm25_retriever)

    retriever = HybridRetriever()
    print("[lifespan] retriever 构建完成")

    # 第 2 步：构建 graph（含 RedisSaver checkpointer）
    print("[lifespan] 连接 Redis 并编译图...")
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    memory_cm = RedisSaver.from_conn_string(
        redis_url,
        ttl={"default_ttl": 1800, "refresh_on_read": True},
    )
    memory = memory_cm.__enter__()
    memory.setup()

    graph = build_rag_graph(retriever).compile(checkpointer=memory)

    # 第 3 步：存到 app.state，供 deps.py 使用
    app.state.graph = graph
    app.state.memory_cm = memory_cm   # 保存上下文管理器，供关闭时 __exit__
    app.state.retriever = retriever

    print("[lifespan] ✅ 服务启动完成")
    print("=" * 60)

    yield  # ← 服务在这里开始接收请求

    # ---------- 关闭 ----------
    print("[lifespan] 正在关闭...")
    try:
        memory_cm.__exit__(None, None, None)
        print("[lifespan] Redis 连接已关闭")
    except Exception as e:
        print(f"[lifespan] 关闭 Redis 时出错: {e}")


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