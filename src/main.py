"""主入口：知识库问答助手"""
import os
import warnings

from src.callback import MyCallbackHandler

warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_core.messages import HumanMessage

from src.config import vectorstore_dir
from src.loader_splitter import load_split
from src.embedding import create_vectorstore, load_vectorstore, get_retriever
from src.rag_agent import build_rag_chain, build_agent
from src.callback import MyCallbackHandler


def setup_rag(retriever):
    """初始化 RAG 链"""
    return build_rag_chain(retriever, with_history=True)


def main():
    print("#" * 60)
    print("📚 知识库问答助手")
    print("#" * 60)

    # 第1步：准备向量库
    if not os.path.exists("vectorstore_dir"):
        print("\n首次运行，正在创建向量库...")
        chunks = load_split()
        vectorstore = create_vectorstore(chunks)
    else:
        print("\n加载已有向量库...")
        vectorstore = load_vectorstore()

    retriever = get_retriever(vectorstore)

    # 第2步：创建 RAG 链
    rag_chain = setup_rag(retriever)
    print("✅ RAG 链已就绪\n")

    # 第3步：多轮问答
    session_id = "user_001"
    config = {
        "configurable": {"session_id": session_id},
        "callbacks": [MyCallbackHandler()],
    }

    questions = [
        "猫和老鼠手游的模式是什么",
        "猫和老鼠手游中杰瑞的是什么位",
    ]

    for q in questions:
        print("=" * 60)
        print(f"👤 用户：{q}")
        result = rag_chain.invoke({"question": q}, config=config)
        print(f"🤖 AI：{result}\n")


if __name__ == "__main__":
    main()
    input("按回车键退出")
