"""
限流服务

职责：
- 基于 Redis 实现请求频率限制
- 防止用户恶意刷接口
"""

from src.api.exceptions import RateLimitError
from src.api.logger import logger


class RateLimiter:
    def __init__(self, redis_client, max_requests: int = 5, window_seconds: int = 60):
        """
        参数：
        - redis_client: 异步 Redis 客户端
        - max_requests: 窗口期内最大请求数
        - window_seconds: 窗口大小（秒）
        """
        self.redis_client = redis_client
        self.max_requests = max_requests
        self.window_seconds = window_seconds

    async def check(self, key: str):
        """
        检查是否超限，超限则抛出 RateLimitError
        参数：
        - key: 限流标识（如 f"rate_limit:{session_id}"）
        """
        current = await self.redis_client.incr(key)

        # 如果是第一次请求，设置过期时间
        if current == 1:
            await self.redis_client.expire(key, self.window_seconds)

        if current > self.max_requests:
            logger.warning(f"触发限流: {key}, 当前次数: {current}")
            raise RateLimitError(f"每分钟最多请求 {self.max_requests} 次，请稍后再试")