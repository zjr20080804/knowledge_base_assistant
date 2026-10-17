import os


from langgraph.checkpoint.redis import RedisSaver
from src.rag_factory import build_retriever
from src.rag_graph import build_rag_graph


import base64

from langchain_community.retrievers import BM25Retriever


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def prepare():
    print("#" * 60)
    print("📚 知识库问答助手（LangGraph 版）")
    print("#" * 60)

    retriever,chunks= build_retriever()

    # 第 2 步：构建 LangGraph 图
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    memory = RedisSaver.from_conn_string(
        REDIS_URL,
        ttl={"default_ttl": 1800, "refresh_on_read": True},
    ).__enter__()
    memory.setup()
    graph = build_rag_graph(retriever).compile(checkpointer=memory)
    print("✅ LangGraph 图已就绪\n")
    return graph, memory
    # 第 3 步：测试两轮问答



def main():
    graph, memory = prepare()
    while True:
        contact_id = input("请输入会话ID：")
        if contact_id == "exit":
            print("谢谢使用")
            break
        config ={"configurable": {"thread_id": contact_id}}
        print("成功进入会话")

        while True:
            image_path = input("请输入图片路径(无需传图可直接回车)：")
            question = input("请输入问题：")
            if question == "exit":
                print("谢谢使用")
                break
            image_url = ""
            if image_path:
                try:
                    image_url = f"data:image/jpeg;base64,{encode_image(image_path)}"
                    print(f"已加载图片：{image_path}")
                except:
                    print(f"图片路径 {image_path} 不存在")
                    continue


            result = graph.invoke({"question": question, "image_url": image_url, "context": "", "answer": "", "messages": []}, config)
            print("AI:",result["answer"])



if __name__ == "__main__":
    main()





