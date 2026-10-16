"""
日志配置

职责：
- 配置 Python logging
- 输出到控制台 + 文件
- 统一格式

为什么独立一个模块？
- 避免各文件重复配置 logging
- 一处修改，全局生效
"""

import logging
import sys
from pathlib import Path

from src.config import root_dir


# ============================================================
# 日志文件路径
# ============================================================

LOG_DIR = root_dir / "logs"
LOG_DIR.mkdir(exist_ok=True)          # 目录不存在就创建
LOG_FILE = LOG_DIR / "app.log"


# ============================================================
# 格式
# ============================================================

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


# ============================================================
# 配置
# ============================================================

def setup_logging(level: int = logging.INFO):
    """
    配置根 logger

    参数：
    - level：日志级别（默认 INFO）
    """
    # 1. 创建 formatter
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    # 2. 控制台 handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # 3. 文件 handler
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)

    # 4. 配置根 logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # 清除已有 handler（防止重复添加）
    root_logger.handlers.clear()

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # 5. 抑制某些第三方库的日志
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


# ============================================================
# 提供一个全局 logger 给业务代码用
# ============================================================

logger = logging.getLogger("kb_assistant")