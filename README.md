# sec-rag-eval

This is a local RAG (Retrieval-Augmented Generation) pipeline I built to process and query documents. Right now, it's set up to ingest an ISO 27001 PDF, chunk the text, store it in a local Chroma database, and use OpenAI (`gpt-4o-mini`) to answer questions as if explaining them to a 5-year-old. 

It also includes a dedicated evaluation script using `ragas` to grade how well the retrieval and generation actually work against a golden dataset of Q&As.

## What's inside
* `main.py`: The main interactive script. If it doesn't find a local Chroma database, it'll run the ingestion pipeline (which can take a minute) before dropping you into a terminal chat loop.
* `evaluate_pipeline.py`: The evaluation script. It runs a predefined set of questions through the retrieval pipeline, gets answers, and then uses Ragas to score the results (context precision, context recall, faithfulness, and answer relevancy). It generates a `rag_evaluation_report.csv` at the end.
* `src/`: Contains the actual mechanics for loading the PDF, handling the parent-child chunking, and database retrieval.

## Getting Started

You'll need Python 3.12+ to run this.

1. **Clone the repo**
2. **Install dependencies:**
   The project uses `pyproject.toml`. You can install everything into a virtual environment using standard pip:
   ```bash
   pip install .
   ```
3. **Set up your environment variables:**
   Create a `.env` file in the root directory. **Do not commit this file.** You'll need to add your API keys to it:
   ```text
   OPENAI_API_KEY="your_openai_key"
   UNSTRUCTURED_API_KEY="your_unstructured_key"
   UNSTRUCTURED_SERVER_URL="https://platform.unstructuredapp.io/api/v1"
   ```
   *(See `.env.example` for a template).*
4. **Drop in your data:**
   Make sure your PDF is located at the path specified in the scripts (currently looking for `data/isolated.pdf`).

## Usage

To run the ingestion and chat with the document:
```bash
python main.py
```

To run the evaluation pipeline and generate a grading report:
```bash
python evaluate_pipeline.py
```