from langchain_community.document_loaders import TextLoader
import src.config as cfg
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_split():
    loader = TextLoader(str(cfg.knowledge_file), encoding="utf-8")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n","",","]
    )
    chunks = splitter.split_documents(docs)
    print(f"切出了{len(chunks)}个文档")
    return chunks

