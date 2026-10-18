"""
自定义异常

职责：
- 定义业务异常类
- 让 error_handler 能区分"业务异常"和"未预期异常"

设计原则：
- 业务异常继承自某个基类，方便统一处理
- 每个异常有 code（程序识别用）和 message（人类可读）
"""


class AppError(Exception):
    """
    应用异常基类

    所有业务异常继承自这个类。
    这样 error_handler 可以统一处理。
    """

    # 默认值，子类可以覆盖
    code: str = "APP_ERROR"
    message: str = "应用异常"

    def __init__(self, detail: str = ""):
        """
        参数：
        - detail：详细描述（可选）
        """
        self.detail = detail
        super().__init__(detail or self.message)


class RagServiceError(AppError):
    """
    RAG 服务异常

    什么时候抛？
    - 检索失败
    - LLM 调用失败
    - 图执行异常
    """

    code = "RAG_SERVICE_ERROR"
    message = "RAG 服务异常"


class InputError(AppError):
    """
    输入异常

    什么时候抛？
    - 用户输入不合法（比如问题太长、session_id 格式错）
    """

    code = "INPUT_ERROR"
    message = "输入不合法"


class DependencyError(AppError):
    """
    依赖异常

    什么时候抛？
    - Redis 不可用
    - 向量库不可用
    """

    code = "DEPENDENCY_ERROR"
    message = "依赖服务不可用"

class RateLimitError(AppError):
    """
    速率限制异常

    什么时候抛？
    - 用户请求频率超过限制
    """

    code = "RATE_LIMIT_ERROR"
    message = "请求频率超过限制"
