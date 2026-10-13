"""
RAG 工厂

职责：
- 把"构建 retriever"的逻辑集中一处
- main_graph.py（命令行入口）和 api/main.py（Web 入口）共用

为什么需要这个文件？
- 之前 main_graph.py 的 prepare() 里有一份构建逻辑
- api/main.py 的 lifespan 里又抄了一份
- 重复代码违反 DRY 原则——改一处要改两处
"""

import os

from langchain_community.retrievers import BM25Retriever

from src.config import vectorstore_dir
from src.loader_splitter import load_split
from src.embedding import create_vectorstore, load_vectorstore


# ============================================================
# 混合检索工具
# ============================================================

def hybrid_retrieve(question, vector_retriever, bm25_retriever, k: int = 6):
    """
    简单版混合检索：向量 + BM25 并集去重

    参数：
    - question：用户问题
    - vector_retriever：向量检索器
    - bm25_retriever：BM25 检索器
    - k：返回前 k 条

    返回：
    - 去重后的文档列表
    """
    vector_docs = vector_retriever.invoke(question)
    bm25_docs = bm25_retriever.invoke(question)

    # 按文档内容去重
    merged = {}
    for doc in vector_docs + bm25_docs:
        key = doc.page_content
        if key not in merged:
            merged[key] = doc

    return list(merged.values())[:k]


class HybridRetriever:
    """
    混合检索器包装类

    把 vector_retriever 和 bm25_retriever 组合成一个统一的 retriever 接口。
    """

    def __init__(self, vector_retriever, bm25_retriever):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever

    def invoke(self, question):
        return hybrid_retrieve(question, self.vector_retriever, self.bm25_retriever)


# ============================================================
# 构建 retriever（工厂函数）
# ============================================================

def build_retriever():
    """
    构建混合检索器

    流程：
    1. 加载并切分文档
    2. 加载或创建向量库
    3. 构建向量检索器 + BM25 检索器
    4. 组合成 HybridRetriever

    返回：
    - HybridRetriever 实例
    - chunks（文档块列表，备用）
    """
    # 第 1 步：加载并切分文档
    print("[factory] 加载文档...")
    chunks = load_split()

    # 第 2 步：加载或创建向量库
    print("[factory] 加载/创建向量库...")
    if not os.path.exists(vectorstore_dir):
        print("  首次运行，正在创建向量库...")
        vectorstore = create_vectorstore(chunks)
    else:
        print("  加载已有向量库...")
        vectorstore = load_vectorstore()

    # 第 3 步：构建两个检索器
    vectorstore_retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 6

    # 第 4 步：组合
    retriever = HybridRetriever(vectorstore_retriever, bm25_retriever)
    print("[factory] retriever 构建完成")

    return retriever, chunks