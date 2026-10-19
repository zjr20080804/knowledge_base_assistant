from fastapi import APIRouter
from pydantic import BaseModel
from src.api.services.auth_service import AuthService

router = APIRouter(tags=["auth"])

class LoginRequest(BaseModel):
    username: str

class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"

@router.post("/login", response_model=LoginResponse)
async def login(req: LoginRequest):
    """获取 Token 接口"""
    token = AuthService.create_access_token(req.username)
    return LoginResponse(access_token=token)