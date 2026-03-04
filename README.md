
# RAG QA Project

Retrieval-Augmented Generation (RAG) demo that indexes a local PDF into ChromaDB using OpenAI embeddings, then answers user questions with contextual grounding via LangChain and GPT models.

## Features
- Loads a PDF from [Docs/PDF.pdf](Docs/PDF.pdf), splits it into overlapping text chunks, and stores them in a persistent Chroma vector store.
- Uses OpenAI `text-embedding-3-small` for embeddings and `gpt-4o-mini` for chat responses.
- Streams answers token-by-token for responsive CLI interaction.

## Prerequisites
- Python 3.11+
- OpenAI API key exported as `OPENAI_API_KEY` (recommended via `.env` in the project root).
- A PDF to index placed at [Docs/PDF.pdf](Docs/PDF.pdf).

## Setup
1) Create and activate a virtual environment (Windows PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

2) Install dependencies:
```bash
pip install "chromadb>=1.5.2" "dotenv>=0.9.9" "langchain>=1.2.10" "langchain-chroma>=1.1.0" "langchain-community>=0.4.1" "langchain-openai>=1.1.10" "pymupdf>=1.27.1"
```

3) Add your OpenAI key to `.env`:
```
OPENAI_API_KEY=sk-...
```

## Usage
Run the CLI:
```bash
python main.py
```

First run creates the Chroma store under `DATABASE/chroma` and indexes the PDF. Subsequent runs reuse the existing store for faster startup. Type questions at the `USER:` prompt; type `exit` or `quit` to stop.

## How It Works
- Document load and chunking: [src/document.py](src/document.py)
- Vector store build/load: [src/vector_db.py](src/vector_db.py)
- CLI and QA loop: [main.py](main.py)

## Notes
- Ensure the `Docs` and `DATABASE` directories are writable. `DATABASE/chroma` is created automatically if missing.
- If you change the source PDF, delete `DATABASE/chroma` to force a fresh embedding run.
