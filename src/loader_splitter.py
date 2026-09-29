from langchain_community.document_loaders import TextLoader
import src.config as cfg
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader

def load_split():
    loader = DirectoryLoader(
        str(cfg.knowledge_dir),
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", "", ",", " "],
    )
    chunks = splitter.split_documents(docs)
    print(f"切出了{len(chunks)}个文档")
    return chunks

