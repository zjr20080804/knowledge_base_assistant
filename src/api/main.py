import os
import sys
from contextlib import asynccontextmanager,AsyncExitStack

from fastapi import FastAPI
from langgraph.checkpoint.redis import AsyncRedisSaver
from redis import asyncio as aioredis
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools


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
    async with AsyncExitStack() as stack:
        logger.info("=" * 60)
        logger.info("🚀 正在启动知识库问答服务...")
        logger.info("=" * 60)

        logger.info("[lifespan] 加载文档...")
        retriever,chunks= build_retriever()
        logger.info("[lifespan] retriever 构建完成")

        logger.info("[lifespan] 连接 Redis 并编译图...")

        redis_client = aioredis.from_url(REDIS_URL,decode_responses = True)
        app.state.redis_client = redis_client
        stack.push_async_callback(redis_client.aclose)

        memory = await stack.enter_async_context(AsyncRedisSaver.from_conn_string(REDIS_URL,
        ttl={"default_ttl": 1800, "refresh_on_read": True}))


        server_params = StdioServerParameters(command=sys.executable,
        args=["E:\\python\\PythonProject1\\练习\\mcp_step1_files.py"])
        read, write = await stack.enter_async_context(stdio_client(server_params))
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        mcp_tools = await load_mcp_tools(session)

        graph = build_rag_graph(retriever,extra_tools=mcp_tools).compile(checkpointer=memory)

        app.state.graph = graph
        app.state.retriever = retriever

        logger.info("[lifespan]  服务启动完成")
        logger.info("=" * 60)

        yield




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
