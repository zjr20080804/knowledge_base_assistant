from locust import HttpUser, task, between
import json
import random

class KBAssistantUser(HttpUser):
    wait_time = between(1, 3)
    host = "http://127.0.0.1:8000"

    def on_start(self):
        username = f"user_{random.randint(1,100000)}"
        response = self.client.post(
            "/login",
            json={"username":username}
        )
        if response.status_code == 200:
            token = response.json().get("access_token")
            self.headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }
            self.session_id=f"session_{username}"
        else:
            print("登录失败！")

    @task(weight=1)
    def test_chat_normal(self):
        payload = {
            "question": "游戏主角是谁？",
            "session_id":self.session_id
        }
        self.client.post("/chat", json=payload, headers=self.headers)

    @task(weight=2)
    def test_chat_stream(self):
        payload = {
            "question": "请简单介绍一下你自己。",
            "session_id":self.session_id
        }
        self.client.post("/chat/stream", json=payload, headers=self.headers)