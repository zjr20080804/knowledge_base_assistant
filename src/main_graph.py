import os
from langgraph.checkpoint.redis import RedisSaver


from src.model import embeddings  # 你的 embedding 模块
from src.loader_splitter import load_split
from src.rag_graph import build_rag_graph
from src.embedding import create_vectorstore, load_vectorstore, get_retriever
from src.config import vectorstore_dir


def prepare():
    print("#" * 60)
    print("📚 知识库问答助手（LangGraph 版）")
    print("#" * 60)

    # 第 1 步：准备向量库（沿用你原来的逻辑）
    if not os.path.exists("vectorstore_dir"):
        print("\n首次运行，正在创建向量库...")
        chunks = load_split()
        vectorstore = create_vectorstore(chunks)
    else:
        print("\n加载已有向量库...")
        vectorstore = load_vectorstore()

    retriever = vectorstore.as_retriever()

    # 第 2 步：构建 LangGraph 图
    REDIS_URL = "redis://localhost:6379/0"
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
            question = input("请输入问题：")
            if question == "exit":
                print("谢谢使用")
                break
            result = graph.invoke({"question": question, "context": "", "answer": "", "messages": []}, config)
            print("AI:",result["answer"])



if __name__ == "__main__":
    main()





