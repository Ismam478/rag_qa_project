import os
from src.document_loader import load_pdf
from src.chunking import push_parent_child_raw, dicts_to_elements_chunks
from src.db_retrieval import retrieve_parent_text
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)

def main():
    print("Hello from sec-rag-eval!")

    pdf_path = "PDF_PATH"
    open_ai_client = OpenAI()


    try:
        if not os.path.exists("./db/local_chroma_db"):
            print("--------------------Database not found. Starting expensive ingestion pipeline...--------------------")
            
            cleaned_elements = load_pdf(pdf_path)
            unstructured_chunks = dicts_to_elements_chunks(cleaned_elements)
            push_parent_child_raw(unstructured_chunks)

        else:
            print("--------------------Database Found!--------------------")

        print("--------System Ready----------")
        while user_query := input("\nEnter your question (or type 'exit' to quit): ").strip():
            
            if user_query.lower() == "exit":
                print("--------------------Exiting the program. Goodbye!--------------------")
                break

            retrieved_texts = retrieve_parent_text(user_query)

            if retrieved_texts:
                context_block = "\n\n".join(retrieved_texts)

                system_prompt = (
                    "You are a Professor of the topic below that extracts relevant information from the following document and explains user questions based on that information, as if the user is 5 years old kid. Now, explain the following question in a simple way but properly:\n\n"
                    f"Document: {context_block}\n\n"
                    "If you don't know the answer, just say 'I don't know'. Do not try to make up an answer.\n\n"
                )

                response = open_ai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            'role': 'system', 'content': system_prompt
                        },
                        {
                            'role': 'user', 'content': user_query
                        }
                    ],
                    temperature=0
                )
                
                print(f"\nAI: {response.choices[0].message.content}")
            
            else:
                print("No relevant information found for your query.")

    except Exception as e:
        print(f"An error occurred: {e}")




if __name__ == "__main__":
    main()
