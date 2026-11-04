import os
import sys
import base64
import asyncio
import mimetypes
from contextlib import asynccontextmanager,AsyncExitStack
from langgraph.checkpoint.redis import AsyncRedisSaver
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from redis import asyncio as aioredis


from src.rag_factory import build_retriever
from src.rag_graph import build_rag_graph, build_user_message


def build_image_data_url(image_path: str) -> str:
    mime, _ = mimetypes.guess_type(image_path)
    if not mime or not mime.startswith("image/"):
        mime = "image/jpeg"
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


@asynccontextmanager
async def prepare_graph():
    async with AsyncExitStack() as stack:
        print("#" * 60)
        print("📚 知识库问答助手（LangGraph Agent 版）")
        print("#" * 60)

        REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        retriever, chunks = build_retriever()
        redis_client = aioredis.from_url(REDIS_URL, decode_responses=True)
        stack.push_async_callback(redis_client.aclose)

        memory = await stack.enter_async_context(AsyncRedisSaver.from_conn_string(redis_client= redis_client,
        ttl={"default_ttl": 1800, "refresh_on_read": True}))


        server_params = StdioServerParameters(command=sys.executable,
        args=["E:\\python\\PythonProject1\\练习\\mcp_step1_files.py"])
        read, write = await stack.enter_async_context(stdio_client(server_params))
        session = await stack.enter_async_context(ClientSession(read,write))
        await session.initialize()
        mcp_tools = await load_mcp_tools(session)

        graph = build_rag_graph(retriever, extra_tools=mcp_tools).compile(checkpointer=memory)


        yield graph
    print(f"资源释放完毕")

async def main():
    async with prepare_graph() as graph:
        while True:
            contact_id = input("请输入会话ID (输入 exit 退出): ")
            if contact_id == "exit":
                break

            config = {"configurable": {"thread_id": contact_id}}
            print(f"成功进入会话: {contact_id}")

            while True:
                image_path = input("请输入图片路径(无需传图可直接回车): ").strip()
                question = input("请输入问题 (输入 exit 退出会话): ").strip()

                if question == "exit":
                    break

                image_url = ""
                if image_path:
                    try:
                        image_url = build_image_data_url(image_path)
                    except Exception:
                        print("图片不存在")
                        continue

                try:
                    result = await graph.ainvoke(
                        {"messages": [build_user_message(question, image_url)]},
                        config=config
                    )
                    print(f"\nAI: {result['messages'][-1].content}\n")
                except Exception as e:
                    print(f"\n[发生错误]: {repr(e)}\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
