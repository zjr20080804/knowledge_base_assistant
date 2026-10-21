import pytest
from fastapi.testclient import TestClient
from src.api.main import app

# 创建测试客户端
# 使用 with 语句可以触发 FastAPI 的 lifespan（启动加载资源）
@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_check(client):
    """测试存活探针"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_without_token(client):
    """测试没有 Token 时，接口必须拒绝访问（403 或 400）"""
    response = client.post(
        "/chat",
        json={"question": "你好", "session_id": "test_user"}
    )
    # 因为我们用了 HTTPBearer，没带 Token 会被 FastAPI 直接拦截
    assert response.status_code in [400, 403]


def test_login_and_get_token(client):
    """测试登录接口是否能正常发放 Token"""
    response = client.post(
        "/login",
        json={"username": "test_admin"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"