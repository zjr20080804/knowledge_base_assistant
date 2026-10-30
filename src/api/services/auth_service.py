from datetime import datetime, timedelta, timezone
import jwt
from src.config import JWT_SECRET_KEY, JWT_ALGORITHM, JWT_EXPIRE_MINUTES
from src.api.exceptions import AppError


class AuthError(AppError):
    code = "AUTH_ERROR"
    message = "认证失败"
    status_code = 401


class AuthService:
    @staticmethod
    def create_access_token(username: str) -> str:
        expire = datetime.now(timezone.utc) + timedelta(minutes=JWT_EXPIRE_MINUTES)
        payload = {
            "sub": username,
            "exp": expire
        }
        token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
        return token

    @staticmethod
    def verify_token(token: str) -> str:
        try:
            payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
            username = payload.get("sub")
            if not username:
                raise AuthError("Token 内容无效")
            return username
        except jwt.ExpiredSignatureError:
            raise AuthError("Token 已过期，请重新登录")
        except jwt.InvalidTokenError:
            raise AuthError("Token 无效或伪造")