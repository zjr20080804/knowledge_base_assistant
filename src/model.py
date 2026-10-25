from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
import src.config as cfg
import os

def get_llm(streaming=False):
    return ChatOpenAI(
        model=cfg.Model,
        api_key=cfg.DeepSeek_API_Key,
        base_url=cfg.DeepSeek_API_URL,
        temperature=0.7,
        streaming=streaming
    )




def embeddings():
    # 判断是否为 Docker 环境
    docker_path = "/app/local_model"
    if os.path.exists(docker_path):
        model_path = docker_path
    else:
        # 本地环境路径 (指向你本地 E:\...\demo\local_model)
        model_path = str(cfg.root_dir / "local_model")

    return HuggingFaceEmbeddings(model_name=model_path)