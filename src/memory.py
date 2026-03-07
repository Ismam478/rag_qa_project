from langchain_core.chat_history import InMemoryChatMessageHistory


store = {}

def get_session_id(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory(max_length=5)
    return store[session_id]


def get_context_with_metadata(query: str, vector_db):

    contexts = vector_db.similarity_search(query, k=3)

    formatted_context = ""

    for i, context in enumerate(contexts):
        source = context.metadata.get('source', 'unknown')
        page = context.metadata.get('page', 'N/A')

        formatted_context += f"\n---\n[Source {i}]: {source} (Page: {page+1})\ncontext: {context.page_content}\n"

    return formatted_context, contexts