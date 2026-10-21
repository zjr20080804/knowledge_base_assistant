FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple/
WORKDIR /app
COPY pyproject.toml uv.lock ./



RUN uv sync --frozen --no-dev
COPY src/ ./src/
COPY local_model/ /app/local_model/
COPY local_reranker/ /app/local_reranker/
ENV HF_HUB_OFFLINE = 1

WORKDIR /app
CMD ["uv", "run", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
