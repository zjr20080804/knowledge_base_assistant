from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.language_models.fake_chat_models import FakeListChatModel
import src.config as cfg
import os

def get_llm(streaming=False):
    if os.getenv("MOCK_LLM") == "1":
        return FakeListChatModel(responses=["压测环境测试性能专用"])

    return ChatOpenAI(
        model=cfg.Model,
        api_key=cfg.DeepSeek_API_Key,
        base_url=cfg.DeepSeek_API_URL,
        temperature=0.7,
        streaming=streaming,
        extra_body={"thinking":{"type":"disabled"}}
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