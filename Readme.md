# 📚 RAG Book Assistant

A simple **Retrieval-Augmented Generation (RAG)** application that allows users to upload a PDF and ask questions based on its content.

## Tech Stack

* Python
* Streamlit
* LangChain
* ChromaDB
* Hugging Face / OpenAI Embeddings
* Mistral AI
* PyPDF

## Project Structure

```text
rag-book-assistant/
├── app.py
├── ingest.py
├── rag.py
├── requirements.txt
├── .env
├── .gitignore
└── chroma_db/
```

## RAG Flow

```text
        Upload PDF
            │
            ▼
       PyPDFLoader
            │
            ▼
      Text Chunking
            │
            ▼
       Embeddings
    ┌───────┴────────┐
    │                │
Hugging Face      OpenAI
    │                │
    └───────┬────────┘
            ▼
         ChromaDB
            │
            ▼
      User Question
            │
            ▼
     MMR Retrieval
            │
            ▼
   Relevant PDF Context
            │
            ▼
       Mistral LLM
            │
            ▼
         Answer
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/thvvamshi/rag-book-assistant.git
cd rag-book-assistant
```

### 2. Install `uv`

`uv` is recommended for fast Python environment and dependency management.

**Windows PowerShell:**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Restart the terminal after installation.

Verify:

```bash
uv --version
```

### 3. Create virtual environment

```bash
uv venv
```

Activate it:

**Windows:**

```powershell
.venv\Scripts\activate
```

**Linux / macOS:**

```bash
source .venv/bin/activate
```

### 4. Install dependencies

```bash
uv pip install -r requirements.txt
```

> `pip install -r requirements.txt` also works if you prefer regular pip.

## Environment Variables

Create a `.env` file:

```env
MISTRAL_API_KEY=your_mistral_api_key
OPENAI_API_KEY=your_openai_api_key
```

`OPENAI_API_KEY` is only required when using OpenAI embeddings.

## Embedding Configuration

The embedding provider is selected in the backend, not in the UI.

In `ingest.py` and `rag.py`:

```python
EMBEDDING_PROVIDER = "huggingface"
```

Available options:

```text
huggingface
openai
```

* **Hugging Face** → Free, local embeddings
* **OpenAI** → API-based embeddings

> Use the same embedding provider in both files. If you switch providers, delete `chroma_db/` and process the PDF again.

## Run

With the virtual environment activated:

```bash
uv run streamlit run app.py
```

Or:

```bash
streamlit run app.py
```

Then open the Streamlit URL shown in the terminal.

## How to Use

1. Upload a PDF through the UI.
2. Click **Create Vector Database**.
3. The PDF is chunked, embedded, and stored in ChromaDB.
4. Enter your question.
5. The system retrieves relevant chunks and sends them to Mistral.
6. The answer is generated using the document context.

## Important

```text
PDF upload → ChromaDB creation → Ask questions
```

* `chroma_db/` is generated automatically.
* `.env` should never be committed.
* Delete `chroma_db/` when changing the embedding provider or source document.
* The Hugging Face embedding model is downloaded automatically on first use.

## License

For educational and demonstration purposes.
