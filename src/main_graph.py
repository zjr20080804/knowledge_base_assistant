import os
import base64
import asyncio
from contextlib import asynccontextmanager
from langgraph.checkpoint.redis import AsyncRedisSaver

from src.rag_factory import build_retriever
from src.rag_graph import build_rag_graph


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ★ 魔法：把 prepare 变成一个异步的上下文管理器
@asynccontextmanager
async def prepare_graph():
    print("#" * 60)
    print("📚 知识库问答助手（LangGraph Agent 版）")
    print("#" * 60)

    retriever, chunks = build_retriever()
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    # 用 async with 完美管理连接生命周期
    async with AsyncRedisSaver.from_conn_string(
            REDIS_URL,
            ttl={"default_ttl": 1800, "refresh_on_read": True},
    ) as memory:
        await memory.asetup()
        graph = build_rag_graph(retriever).compile(checkpointer=memory)
        print("✅ LangGraph 图已就绪\n")

        # 把图交给外面用，此时连接一直活着
        yield graph

        # 等外面 while 循环彻底结束，自动执行到这里关闭连接
        print("正在关闭 Redis 连接...")


async def main():
    # ★ 使用优雅的 async with 语法
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
                        image_url = f"data:image/jpeg;base64,{encode_image(image_path)}"
                    except Exception:
                        print("图片不存在")
                        continue

                try:
                    result = await graph.ainvoke(
                        {"question": question, "image_url": image_url},
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
