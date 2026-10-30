class AppError(Exception):

    code: str = "APP_ERROR"
    message: str = "应用异常"
    status_code: int = 400

    def __init__(self, detail: str = ""):
        self.detail = detail
        super().__init__(detail or self.message)


class RagServiceError(AppError):

    code = "RAG_SERVICE_ERROR"
    message = "RAG 服务异常"
    status_code = 502


class InputError(AppError):

    code = "INPUT_ERROR"
    message = "输入不合法"


class DependencyError(AppError):

    code = "DEPENDENCY_ERROR"
    message = "依赖服务不可用"
    status_code = 503

class RateLimitError(AppError):

    code = "RATE_LIMIT_ERROR"
    message = "请求频率超过限制"
    status_code = 429
