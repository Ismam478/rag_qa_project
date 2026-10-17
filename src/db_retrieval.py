import chromadb
from openai import OpenAI
import json
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)

def retrieve_parent_text(user_query: str, top_k: int = 3):
    """
    Takes an incoming text string from a user, embeds it, queries Chroma for the most similar small sentence vectors, bridges those vector hits back to full parent paragraphs using metadata IDs, and returns the full contexts.
    """

    chroma_client = chromadb.PersistentClient(path="./db/local_chroma_db")
    child_collection =  chroma_client.get_or_create_collection("Child_Collection")

    open_ai = OpenAI()

    with open("parent_store.json", "r") as f:
        parent_store = json.load(f)

    response = open_ai.embeddings.create(
        input= user_query,
        model="text-embedding-3-small"
    )

    query_vector = response.data[0].embedding

    results = child_collection.query(
        query_embeddings=[query_vector],
        n_results=top_k
    )

    retrieved_text = []
    seen_texts = set()

    for i in range(len(results["ids"][0])):
        child_text = results["documents"][0][i]
        parent_id = results["metadatas"][0][i]["parent_id"]
        distance = results["distances"][0][i]

        if parent_id in seen_texts:
            continue

        seen_texts.add(parent_id)
        parent_data = parent_store[parent_id]

        retrieved_text.append(parent_data["text"])

    return retrieved_text