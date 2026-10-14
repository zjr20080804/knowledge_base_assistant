"""
配置管理

- 用 pydantic-settings 统一管理配置
- 从 .env 读取，带类型校验
- 对外仍暴露模块级变量，保持向后兼容
"""

import os

os.environ["HF_HUB_OFFLINE"] = "1"

from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict


# ============================================================
# 项目根目录 & 加载 .env
# ============================================================

root_dir = Path(__file__).parent.parent
load_dotenv(root_dir / ".env")


# ============================================================
# Settings 类
# ============================================================

class Settings(BaseSettings):
    """
    所有配置集中在这里
    """

    model_config = SettingsConfigDict(
        env_file=root_dir / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,     # ← .env 里大小写都认
    )

    # ---- API 配置 ----
    DEEPSEEK_API_KEY: str = ""
    DEEPSEEK_API_URL: str = "https://api.deepseek.com/v1"
    MODEL: str = "deepseek-flash"

    # ---- 嵌入模型 ----
    EMBEDDING_MODEL: str = "BAAI/bge-small-zh-v1.5"

    # ---- 检索参数 ----
    RETRIEVAL_K: int = 6

    # ---- Redis ----
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_TTL: int = 1800

    # ---- 路径（计算属性）----
    @property
    def data_dir(self) -> Path:
        return root_dir / "data"

    @property
    def vectorstore_dir(self) -> Path:
        return root_dir / "chroma_db"

    @property
    def knowledge_dir(self) -> Path:
        return self.data_dir / "knowledge"


# ============================================================
# 单例
# ============================================================

settings = Settings()


# ============================================================
# 向后兼容：模块级变量
# ============================================================

DeepSeek_API_Key = settings.DEEPSEEK_API_KEY
DeepSeek_API_URL = settings.DEEPSEEK_API_URL
Model = settings.MODEL

embedding_model = settings.EMBEDDING_MODEL
Retrieval_k = settings.RETRIEVAL_K

data_dir = settings.data_dir
vectorstore_dir = settings.vectorstore_dir
knowledge_dir = settings.knowledge_dir

REDIS_URL = settings.REDIS_URL
REDIS_TTL = settings.REDIS_TTL
