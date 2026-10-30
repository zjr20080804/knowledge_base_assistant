import os

os.environ["HF_HUB_OFFLINE"] = "1"

from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict



root_dir = Path(__file__).parent.parent
load_dotenv(root_dir / ".env")



class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=root_dir / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    DEEPSEEK_API_KEY: str = ""
    DEEPSEEK_API_URL: str = "https://api.deepseek.com/v1"
    MODEL: str = "deepseek-flash"

    EMBEDDING_MODEL: str = "BAAI/bge-small-zh-v1.5"

    RETRIEVAL_K: int = 6

    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_TTL: int = 1800

    JWT_SECRET_KEY: str = "super-secret-key-change-me-in-production"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 60

    @property
    def data_dir(self) -> Path:
        return root_dir / "data"

    @property
    def vectorstore_dir(self) -> Path:
        return root_dir / "chroma_db"

    @property
    def knowledge_dir(self) -> Path:
        return self.data_dir / "knowledge"



settings = Settings()



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

JWT_SECRET_KEY = settings.JWT_SECRET_KEY
JWT_ALGORITHM = settings.JWT_ALGORITHM
JWT_EXPIRE_MINUTES = settings.JWT_EXPIRE_MINUTES
