from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from src.api.logger import logger
from src.api.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get("/ready")

async def ready(request: Request) -> JSONResponse:
    graph = getattr(request.app.state, "graph", None)
    has_graph = graph is not None

    redis_ok = False
    redis_client = getattr(request.app.state, "redis_client", None)
    if redis_client is not None:
        try:
            await redis_client.ping()
            redis_ok = True
        except Exception:
            logger.warning("[ready] redis ping 失败", exc_info=True)

    is_ready = has_graph and redis_ok
    return JSONResponse(
        status_code=200 if is_ready else 503,
        content={
            "status": "ready" if is_ready else "not_ready",
            "graph": has_graph,
            "redis": redis_ok,
        },
    )