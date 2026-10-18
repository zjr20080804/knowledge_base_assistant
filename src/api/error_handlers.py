"""
全局异常处理器

职责：
- 拦截所有未处理的异常
- 业务异常：返回结构化 JSON 给客户端（带 detail）
- 未预期异常：记录完整 Traceback，返回通用错误（不带 detail）
"""

import os

from fastapi import Request
from fastapi.responses import JSONResponse

from src.api.exceptions import AppError
from src.api.logger import logger


# 判断是否 debug 模式（环境变量控制）
IS_DEBUG = os.getenv("DEBUG", "false").lower() == "true"


# ============================================================
# 处理器 1：业务异常（AppError 及其子类）
# ============================================================

async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    """
    处理 AppError

    - 返回 400 状态码
    - 返回详细的 code / message / detail
    """
    logger.warning(f"业务异常: {request.method} {request.url.path} | {exc.code} - {exc.detail}")

    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "code": exc.code,
                "message": exc.message,
                "detail": exc.detail,
            }
        },
    )


# ============================================================
# 处理器 2：未预期异常（兜底）
# ============================================================

async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    处理所有未预期异常

    - 记录完整 Traceback 到日志
    - 返回 500 状态码
    - debug 模式返回 detail，生产模式不返回
    """
    # 记录完整堆栈（logger.exception 会自动带上 Traceback）
    logger.exception(f"未预期异常: {request.method} {request.url.path}")

    content = {
        "error": {
            "code": "INTERNAL_ERROR",
            "message": "服务器内部错误",
        }
    }

    # 策略 C：只在 debug 模式返回 detail
    if IS_DEBUG:
        content["error"]["detail"] = str(exc)

    return JSONResponse(status_code=500, content=content)