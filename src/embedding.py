from langchain_chroma import Chroma
import src.config as cfg
import src.model as ml

def create_vectorstore(chunks):
    embeddings = ml.embeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(cfg.vectorstore_dir)
    )
    print(f"向量库已创建:{cfg.vectorstore_dir}")
    return vectorstore

def load_vectorstore():
    embeddings = ml.embeddings()
    vectorstore = Chroma(
        persist_directory=str(cfg.vectorstore_dir),
        embedding_function=embeddings
    )
    print(f"向量库已加载:{cfg.vectorstore_dir}")
    return vectorstore

def get_retriever(vectorstore):
    return vectorstore.as_retriever(search_kwargs={"k": cfg.Retrieval_k})
