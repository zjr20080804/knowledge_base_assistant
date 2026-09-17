from src.RedisHistory import RedisChatMessageHistory


def get_session_history(session_id):
    return RedisChatMessageHistory(session_id=session_id)
