FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple/
WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY src/ ./src/


RUN uv sync --frozen --no-dev

WORKDIR /app
CMD ["uv", "run", "python", "-c", "import sys;sys.path.insert(0,'.');from src.rag_graph import build_rag_graph;print('ok')"]
