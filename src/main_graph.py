import os

from fastapi.openapi import docs
from langgraph.checkpoint.redis import RedisSaver


from src.model import embeddings  # 你的 embedding 模块
from src.loader_splitter import load_split
from src.rag_graph import build_rag_graph
from src.embedding import create_vectorstore, load_vectorstore, get_retriever
from src.config import vectorstore_dir
import base64

from langchain_community.retrievers import BM25Retriever


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def prepare():
    print("#" * 60)
    print("📚 知识库问答助手（LangGraph 版）")
    print("#" * 60)

    # 第 1 步：准备向量库（沿用你原来的逻辑）
    chunks = load_split()
    if not os.path.exists("vectorstore_dir"):
        print("\n首次运行，正在创建向量库...")

        vectorstore = create_vectorstore(chunks)
    else:
        print("\n加载已有向量库...")
        vectorstore = load_vectorstore()

    vectorstore_retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 6
    class HybridRetriever:
        def invoke(self, question):
            return hybrid_retrieve(question, vectorstore_retriever, bm25_retriever)
    retriever = HybridRetriever()


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

def hybrid_retrieve(question, vector_retriever, bm25_retriever, k=6):
    """简单版混合检索：向量 + BM25 并集去重。"""
    vector_docs = vector_retriever.invoke(question)
    bm25_docs = bm25_retriever.invoke(question)

    # 按文档内容去重
    merged = {}
    for doc in vector_docs + bm25_docs:
        key = doc.page_content
        if key not in merged:
            merged[key] = doc

    return list(merged.values())[:k]

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





