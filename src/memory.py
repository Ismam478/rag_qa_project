from langchain_core.chat_history import InMemoryChatMessageHistory





store = {}

def get_session_id(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory(max_length=5)
    return store[session_id]