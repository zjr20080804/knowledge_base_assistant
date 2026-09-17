import json
from typing import List

import redis
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage


class RedisChatMessageHistory(BaseChatMessageHistory):
    """把对话历史存进 Redis 的 List 结构，每条消息是 JSON 序列化的字符串。"""

    def __init__(
        self,
        session_id: str,
        redis_url: str = "redis://localhost:6379/0",
        ttl: int = 604800,       # 7 天，单位：秒
        key_prefix: str = "chat",
    ):
        #设置默认参数

        self.session_id = session_id
        self.ttl = ttl
        self.key_prefix = key_prefix
        self.client = redis.from_url(redis_url, decode_responses=True)
        #连接 Redis 数据库,创建客户端对象

    @property
    def key(self) -> str:
        #前面的格式key_prefix(chat):session_id:messages
        return f"{self.key_prefix}:{self.session_id}:messages"

    def add_message(self, message: BaseMessage) -> None:
        record = {
            "type": message.type,       # "human" 或 "ai"
            "content": message.content,
        }
        self.client.rpush(self.key, json.dumps(record, ensure_ascii=False))
        #从尾部加入信息

        self.client.expire(self.key, self.ttl)
        #每次加入信息后，都更新过期时间



    @property
    def messages(self) -> List[BaseMessage]:
        raw_items = self.client.lrange(self.key, 0, -1)
        #从 Redis 中获取所有消息

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