"""
健康检查路由

作用：
- /health：进程存活探针（liveness）
- /ready：就绪探针（readiness），检查依赖是否就绪

为什么要两个？
- liveness 只关心"进程是否活着"，挂了就重启
- readiness 关心"依赖是否就绪"，没就绪就不给流量
- K8s、Docker、负载均衡器都用这两个探针
"""

from fastapi import APIRouter, Request

from src.api.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """
    存活检查

    只返回 200 表示进程活着，不检查任何依赖。
    """
    return HealthResponse(status="ok")


@router.get("/ready")

async def ready(request: Request) -> dict:
    """
    就绪检查

    检查依赖（graph / redis）是否已经初始化。
    检查redis是否能ping通
    """
    print(f"[ready]has_graph:{hasattr(request.app.state, 'graph')}")
    print(f"[ready]has_redis_client:{hasattr(request.app.state, 'redis_client')}")

    graph = getattr(request.app.state, "graph", None)
    has_graph = graph is not None

    # 检查 Redis
    redis_ok = False
    redis_client = getattr(request.app.state, "redis_client", None)
    if redis_client is not None:
        try:
            await redis_client.ping()
            redis_ok = True
        except Exception as e:
            redis_ok = False
            print(f"[ready]redis ping失败:{type(e).__name__} {e}")

    status = "ready" if (has_graph and redis_ok) else "not_ready"
    return {
        "status": status,
        "graph": has_graph,
        "redis": redis_ok,
    }