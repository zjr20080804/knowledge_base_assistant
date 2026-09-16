from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
import src.config as cfg

def get_llm(streaming=False):
    return ChatOpenAI(
        model=cfg.Model,
        api_key=cfg.DeepSeek_API_Key,
        base_url=cfg.DeepSeek_API_URL,
        temperature=0.7,
        streaming=streaming
    )

def embeddings():
    return HuggingFaceEmbeddings(model_name=cfg.embedding_model)
