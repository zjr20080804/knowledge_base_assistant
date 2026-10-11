"""
依赖注入

职责：
- 把 lifespan 里初始化好的资源（graph、memory 等）暴露给路由函数
- 让路由函数只需要声明"我需要什么"，不需要关心怎么构造

FastAPI 的依赖注入机制：
- 路由函数签名里写 `x = Depends(get_x)`
- FastAPI 会在调用路由前先调用 get_x，把返回值传给 x
"""

from typing import Any

from fastapi import Request, Depends

from src.api.services.rag_service import RagService


# ============================================================
# 依赖 1：获取已编译的 RAG graph
# ============================================================

def get_graph(request: Request) -> Any:
    """
    从 app.state 里取出 lifespan 时构建好的 graph

    为什么用 request.app.state 而不是全局变量？
    - 全局变量在测试时无法替换
    - app.state 是 FastAPI 官方推荐存放"应用级单例资源"的地方
    - 每个 FastAPI 实例有独立的 state，测试时可以创建多个 app 互不干扰
    """
    return request.app.state.graph


# ============================================================
# 依赖 2：获取 RagService 实例
# ============================================================

def get_rag_service(graph=Depends(get_graph)) -> RagService:
    """
    构造 RagService 实例

    注意：
    - 这里没有用 @lru_cache 或全局单例
    - 因为 RagService 本身很轻（只持有 graph 引用）
    - 每次请求创建一个新实例开销极小
    - 如果你想缓存，可以加 @lru_cache，但要注意 graph 变了缓存要清
    """
    return RagService(graph)