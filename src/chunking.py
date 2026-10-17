import chromadb
from openai import OpenAI
import uuid
import json
from dotenv import load_dotenv, find_dotenv
from unstructured.staging.base import elements_from_dicts
from unstructured.chunking.title import chunk_by_title

load_dotenv(find_dotenv(), override=True)

def dicts_to_elements_chunks(list_of_dict):
    """
    Converts a list of dictionaries to Unstructured elements and then chunks them by title.
    """

    elements = elements_from_dicts(list_of_dict)
    cleaned_elements = chunk_by_title(
        elements,
        multipage_sections = True,
        combine_text_under_n_chars = 500,
        max_characters = 1500
    )

    return cleaned_elements



def push_parent_child_raw(unstructured_chunks):
    """
    Splits semantic chunks into sentence-level children, embeds the children, 
    and links them back to the full parent text.
    """

    chroma_client = chromadb.PersistentClient(path="./db/local_chroma_db")
    child_collection =  chroma_client.get_or_create_collection("Child_Collection")

    open_ai = OpenAI()

    parent_store = {}

    c_ids = []
    c_documents = []
    c_metadatas = []
    c_embeddings = []

    for chunk in unstructured_chunks:
        parent_id = str(uuid.uuid4())
        parent_text = chunk.text

        parent_store[parent_id] = {
            "text" : parent_text,
            "page_number" : chunk.metadata.page_number or 0,
            "file_name" : chunk.metadata.filename or "unknown"
        }

        child_sentences = parent_text.split(". ")
        

        for sentence in child_sentences:
            clean_sentence = sentence.strip()

            if len(clean_sentence) < 15:
                continue

            response = open_ai.embeddings.create(
                input=clean_sentence,
                model="text-embedding-3-small"
            )

            c_ids.append(str(uuid.uuid4()))
            c_documents.append(clean_sentence)
            c_metadatas.append({"parent_id" : parent_id})
            c_embeddings.append(response.data[0].embedding)

    child_collection.add(
        ids=c_ids,
        documents=c_documents,
        metadatas=c_metadatas,
        embeddings=c_embeddings
    )

    with open ("parent_store.json", "w") as f:
        json.dump(parent_store, f, indent=4)

    return child_collection