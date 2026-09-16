import os
os.environ["HF_HUB_OFFLINE"] = "1"
from pathlib import Path
from dotenv import load_dotenv



#定义项目根目录
root_dir = Path(__file__).parent.parent
load_dotenv(root_dir / ".env")
#API配置
DeepSeek_API_Key = os.getenv("DEEPSEEK_API_KEY")
DeepSeek_API_URL = "https://api.deepseek.com/v1"
Model = "deepseek-flash"
#配置路径
data_dir = root_dir / "data"
vectorstore_dir = root_dir / "chroma_db"
knowledge_file = data_dir / "knowledge"
#嵌入模型
embedding_model = "BAAI/bge-small-zh-v1.5"
Retrieval_k = 5




