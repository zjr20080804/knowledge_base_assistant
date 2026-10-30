from src.api.exceptions import RateLimitError
from src.api.logger import logger


RATE_LIMIT_SCRIPT = """
local current = redis.call('INCR', KEYS[1])
if current == 1 then
    redis.call('EXPIRE', KEYS[1], ARGV[1])
end
return current
"""


class RateLimiter:
    def __init__(self, redis_client, max_requests: int = 5, window_seconds: int = 60):

        self.redis_client = redis_client
        self.max_requests = max_requests
        self.window_seconds = window_seconds

    async def check(self, key: str):

        current = await self.redis_client.eval(RATE_LIMIT_SCRIPT, 1, key, self.window_seconds)

        if current > self.max_requests:
            logger.warning(f"触发限流: {key}, 当前次数: {current}")
            raise RateLimitError(f"每分钟最多请求 {self.max_requests} 次，请稍后再试")