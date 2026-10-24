FROM python:3.12-slim


WORKDIR /app


COPY requirements.txt ./
RUN pip install torch==2.14.1+cpu --index-url https://download.pytorch.org/whl/cpu
ENV PIP_INDEX_URL=https://mirrors.ustc.edu.cn/pypi/simple
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ ./src/
COPY local_model/ /app/local_model/
COPY local_reranker/ /app/local_reranker/
ENV HF_HUB_OFFLINE = 1

WORKDIR /app
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
