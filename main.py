from src.document import process_pdf
from src.vector_db import get_vector_db
from src.memory import get_session_id, get_context_with_metadata
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv, find_dotenv
import os


file_path = "./Docs/PDF.pdf"

def main():
    # Checking if the file directory exists or not
    # Creating the directory if the directory doesn't exists
    if not os.path.exists("DATABASE/chroma"):
        print("Creating and Initializing Vector Database")
        docs = process_pdf(file_path)
        vector_db = get_vector_db(chunks=docs)
    else:
    
    # Using the Previous database if the Directory exists
        print("Loading Vector Database.......")
        vector_db = get_vector_db()

    while True:
        user_input = input("\nUSER: ")

        if user_input.lower() in ['exit', 'quit']:
            print("Exiting the Program")
            break
        

        # Similarity Searching based on the input
        context_text = get_context_with_metadata(user_input, vector_db=vector_db)


        # Creating LLM
        llm_openai = ChatOpenAI(model='gpt-4o-mini', temperature=0)


        # Creating the prompt
        prompt = ChatPromptTemplate.from_messages(
            [
                ('system', 'You are a helpful assistant who uses Provided context for answering questions'),
                MessagesPlaceholder(variable_name='chat_history'),
                ('user', """
                    You are a professional research assistant. Use the provided context to answer the user's question.
                    Every claim you make must be followed by a citation that uses the source and page values from the context, e.g. (Source: filname.pdf(just file name not the directory), Page: 3). If a page is unavailable, use Page: N/A.

                    Context:
                    {context}

                    Question: {user_input}

                    Answer:
                    """)
            ]
        )


        # Creating the chain
        chain = prompt | llm_openai


        # Creating the LLM with chat history
        llm_with_history = RunnableWithMessageHistory(
            chain,
            get_session_history= get_session_id,
            input_messages_key= 'user_input',
            history_messages_key= 'chat_history'
        )



        # Configuring for user Chat_history
        config = {'configurable': {'session_id': 'user_1'}}

        print("\nAI: ")
        

        # Streaming the answer for user
        for chunk in llm_with_history.stream({'user_input': user_input, 'context': context_text}, config=config):
            if isinstance(chunk, AIMessage):
                print(chunk.content, end='', flush=True)



if __name__ == "__main__":
    main()
