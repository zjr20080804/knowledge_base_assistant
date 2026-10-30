import os

from langchain_community.retrievers import BM25Retriever

from src.config import vectorstore_dir
from src.loader_splitter import load_split
from src.embedding import create_vectorstore, load_vectorstore



def hybrid_retrieve(question, vector_retriever, bm25_retriever, k: int = 6, rrf_k: int = 60):
    vector_docs = vector_retriever.invoke(question)
    bm25_docs = bm25_retriever.invoke(question)

    scores = {}
    doc_map = {}
    for retriever in (vector_retriever, bm25_retriever):
        for rank, doc in enumerate(retriever.invoke(question)):
            key = doc.page_content
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
            doc_map.setdefault(key, doc)

    top_keys = sorted(scores, key=scores.get, reverse=True)[:k]
    return [doc_map[key] for key in top_keys]


class HybridRetriever:

    def __init__(self, vector_retriever, bm25_retriever):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever

    def invoke(self, question):
        return hybrid_retrieve(question, self.vector_retriever, self.bm25_retriever)



def build_retriever():
    print("[factory] 加载文档...")
    chunks = load_split()

    print("[factory] 加载/创建向量库...")
    if not os.path.exists(vectorstore_dir):
        print("  首次运行，正在创建向量库...")
        vectorstore = create_vectorstore(chunks)
    else:
        print("  加载已有向量库...")
        vectorstore = load_vectorstore()

    vectorstore_retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 6

    retriever = HybridRetriever(vectorstore_retriever, bm25_retriever)
    print("[factory] retriever 构建完成")

    return retriever, chunks