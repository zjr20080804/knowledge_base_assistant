import pytest
from fastapi.testclient import TestClient
from src.api.main import app

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_without_token(client):
    response = client.post(
        "/chat",
        json={"question": "你好", "session_id": "test_user"}
    )
    assert response.status_code in [400, 403]


def test_login_and_get_token(client):
    response = client.post(
        "/login",
        json={"username": "test_admin"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"