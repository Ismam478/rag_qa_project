from langchain_community.document_loaders import PyMuPDFLoader
import os



# Loading the PDF
def load_document(pdf_path: str):
    
    if not os.path.exists(pdf_path):
        raise FileNotFoundError("The following directory maybe empty")
    
    name, ext = os.path.splitext(pdf_path)

    if ext == '.pdf':
        pdf_loader = PyMuPDFLoader(pdf_path)
    
    return pdf_loader.load()


# Chunking data
def chunk_data(docs):
    from langchain_classic.text_splitter import RecursiveCharacterTextSplitter

    if not docs:
        raise ValueError("The following file maybe empty")
    

    chunks = RecursiveCharacterTextSplitter(
        chunk_size =1000,
        chunk_overlap = 100
    ).split_documents(docs)

    return chunks