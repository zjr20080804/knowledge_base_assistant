
from typing import Any

from fastapi import Request, Depends

from src.api.services.rate_limiter import RateLimiter
from src.api.services.auth_service import AuthService
from fastapi.security import HTTPBearer,HTTPAuthorizationCredentials



def get_graph(request: Request) -> Any:
    return request.app.state.graph

def get_rate_limiter(request: Request) -> RateLimiter:
    redis_client=request.app.state.redis_client
    return RateLimiter(redis_client,max_requests=5,window_seconds=60)

security = HTTPBearer()
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    token = credentials.credentials
    return AuthService.verify_token(token)