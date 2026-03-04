from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv, find_dotenv
import os


# Loading The key
load_dotenv(find_dotenv(), override=True)



# storing the chunks into vector
def get_vector_db(chunks = None):
    embedding_model = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536)

    chroma_path = os.path.join("DATABASE", "chroma")
    os.makedirs(chroma_path, exist_ok=True)
        
    if chunks:
        # create and persist directory
        vector_db = Chroma.from_documents(
            embedding=embedding_model,
            documents=chunks,
            persist_directory=chroma_path
        )

    else:
        # loading the vector
        vector_db = Chroma(
            persist_directory=chroma_path,
            embedding_function=embedding_model
        )

    return vector_db