# 🧠 企业级知识库问答助手 (Knowledge Base Assistant)

![Python](https://img.shields.io/badge/Python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)
![LangChain](https://img.shields.io/badge/LangChain-0.1+-orange)
![Docker](https://img.shields.io/badge/Docker-Supported-blue)
![Redis](https://img.shields.io/badge/Redis-Async-red)

一个基于 **FastAPI + LangGraph + 本地大模型** 构建的高性能 RAG（检索增强生成）智能问答系统。支持多用户并发、意图路由、混合检索、多模态图片解析，并具备企业级的鉴权、限流与异常处理机制。

## 核心特性

- **极速异步架构**：全链路采用 `async/await`（FastAPI + `redis.asyncio` + 异步图执行），彻底告别事件循环阻塞。
- **智能 RAG 引擎**：基于 LangGraph 构建状态图，支持意图路由（闲聊/知识库检索）、多路召回、BGE-Reranker 精排与 Map-Reduce 摘要。
- ** 多模态支持**：无缝接入 DeepSeek 视觉模型，支持图片 Base64 编码输入与图文混合问答。
- **企业级安全与治理**：JWT 身份认证、基于 Redis 的滑动窗口接口限流（防止 API 额度恶意盗刷）、全局异常捕获与结构化日志落盘。
- **离线容器化部署**：采用 Docker Compose 一键编排 App 与 Redis，内置本地大模型文件，彻底摆脱国内网络环境依赖。

## 系统架构图

```text
[ 客户端 ] 
   │ (HTTP / JWT Bearer Token)
   ▼
[ FastAPI 应用 ]
   ├─ 中间件：全局异常拦截 (AppError Handler)
   ├─ 路由层：/login (认证) / /chat (业务)
   ├─ 依赖注入：依赖获取 Graph / Redis / 限流器
   ▼
[ 业务服务层 (RagService) ]
   ├─ 鉴权校验 (JWT)
   ├─ 频控校验 (Redis INCR)
   ▼
[ LangGraph 编排引擎 ]
   ├─ 意图路由 (knowledge / chat)
   ├─ 混合检索 (Chroma + BM25 + Rerank)
   ├─ 文档压缩 (Map-Reduce)
   ▼
[ 底层资源 ]
   ├─ Redis (会话持久化 + 限流计数)
   ├─ 本地模型 (BGE Embedding + BGE Reranker)
   └─ DeepSeek API (LLM 推理)