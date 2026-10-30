import os
import base64
import asyncio
import mimetypes
from contextlib import asynccontextmanager
from langgraph.checkpoint.redis import AsyncRedisSaver

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
    print("#" * 60)
    print("📚 知识库问答助手（LangGraph Agent 版）")
    print("#" * 60)

    retriever, chunks = build_retriever()
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    async with AsyncRedisSaver.from_conn_string(
            REDIS_URL,
            ttl={"default_ttl": 1800, "refresh_on_read": True},
    ) as memory:
        await memory.asetup()
        graph = build_rag_graph(retriever).compile(checkpointer=memory)
        print("✅ LangGraph 图已就绪\n")

        yield graph

        print("正在关闭 Redis 连接...")


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
