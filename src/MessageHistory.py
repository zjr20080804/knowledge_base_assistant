from langchain_core.chat_history import InMemoryChatMessageHistory

id_store = {}

def get_session_history(session_id):
    if  session_id not in id_store:
        id_store[session_id] = InMemoryChatMessageHistory()
    return id_store[session_id]
