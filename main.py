from src.document import load_document, chunk_data
from src.vector_db import get_vector_db
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv, find_dotenv
import os


file_path = "./Docs/PDF.pdf"

def main():
    if not os.path.exists("DATABASE/chroma"):
        print("Creating and Initializing Vector Database")
        docs = load_document(file_path)
        chunks = chunk_data(docs=docs)
        vector_db = get_vector_db(chunks=chunks)
    else:
        print("Loading Vector Database.......")
        vector_db = get_vector_db()

    while True:
        user_input = input("\nUSER: ")

        if user_input.lower() in ['exit', 'quit']:
            print("Exiting the Program")
            break

        context = vector_db.similarity_search(user_input, k=3)

        llm_openai = ChatOpenAI(model='gpt-4o-mini', temperature=0)

        prompt = ChatPromptTemplate.from_messages(
            [
                ('system', 'You are a helpful assistant who uses Provided context for answering questions'),
                ('user', 'Here is the question: {user_input} and here is context {context}')
            ]
        )

        chain = prompt | llm_openai

        print("\nAI: ")
        
        for chunk in chain.stream({'user_input': user_input, 'context': context}):
            if isinstance(chunk, AIMessage):
                print(chunk.content, end='', flush=True)



if __name__ == "__main__":
    main()
