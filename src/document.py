from langchain_community.document_loaders import PyMuPDFLoader
import os
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter



# Loading the PDF
def process_pdf(pdf_path: str):
    
    if not os.path.exists(pdf_path):
        raise FileNotFoundError("The following directory maybe empty")
    
    name, ext = os.path.splitext(pdf_path)

    if ext == '.pdf':
        pdf_loader = PyMuPDFLoader(pdf_path)
        docs = pdf_loader.load()
        return docs

    chunks = RecursiveCharacterTextSplitter(
        chunk_size =1000,
        chunk_overlap = 100
    ).split_documents(docs)

    for chunk in chunks:
        if 'page' not in chunk.metadata:
            chunk.metadata['page'] = 'unknown'
        

    return chunks