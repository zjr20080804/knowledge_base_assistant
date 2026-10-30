import json
from typing import List

import redis
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage


class RedisChatMessageHistory(BaseChatMessageHistory):

    def __init__(
        self,
        session_id: str,
        redis_url: str = "redis://localhost:6379/0",
        ttl: int = 604800,
        key_prefix: str = "chat",
    ):

        self.session_id = session_id
        self.ttl = ttl
        self.key_prefix = key_prefix
        self.client = redis.from_url(redis_url, decode_responses=True)

    @property
    def key(self) -> str:
        return f"{self.key_prefix}:{self.session_id}:messages"

    def add_message(self, message: BaseMessage) -> None:
        record = {
            "type": message.type,
            "content": message.content,
        }
        self.client.rpush(self.key, json.dumps(record, ensure_ascii=False))

        self.client.expire(self.key, self.ttl)



    @property
    def messages(self) -> List[BaseMessage]:
        raw_items = self.client.lrange(self.key, 0, -1)

        result: List[BaseMessage] = []
        for raw in raw_items:
            data = json.loads(raw)
            if data["type"] == "human":
                result.append(HumanMessage(content=data["content"]))
            elif data["type"] == "ai":
                result.append(AIMessage(content=data["content"]))
        return result

    def clear(self) -> None:
        self.client.delete(self.key)