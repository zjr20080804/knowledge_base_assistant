import os

from fastapi import Request
from fastapi.responses import JSONResponse

from src.api.exceptions import AppError
from src.api.logger import logger


IS_DEBUG = os.getenv("DEBUG", "false").lower() == "true"



async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:

    logger.warning(f"业务异常: {request.method} {request.url.path} | {exc.code} - {exc.detail}")

    headers = {}
    if exc.status_code == 401:
        headers["WWW-Authenticate"] = "Bearer"
    elif exc.status_code == 429:
        headers["Retry-After"] = "60"

    return JSONResponse(
        status_code=exc.status_code,
        headers=headers,
        content={
            "error": {
                "code": exc.code,
                "message": exc.message,
                "detail": exc.detail,
            }
        },
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:

    logger.exception(f"未预期异常: {request.method} {request.url.path}")

    content = {
        "error": {
            "code": "INTERNAL_ERROR",
            "message": "服务器内部错误",
        }
    }

    if IS_DEBUG:
        content["error"]["detail"] = str(exc)

    return JSONResponse(status_code=500, content=content)