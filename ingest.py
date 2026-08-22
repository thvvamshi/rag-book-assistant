import os
import shutil

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings


# Configuration
CHROMA_DIR = "chroma_db"

# Choose embedding provider here.
# Options: "huggingface" or "openai"
EMBEDDING_PROVIDER = "huggingface"


# Embedding Model
def get_embedding_model():

    if EMBEDDING_PROVIDER == "openai":

        return OpenAIEmbeddings(
            model="text-embedding-3-small"
        )

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# Create Vector Database
def create_vector_database(pdf_path):

    # Remove previous database
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)

    print("Loading PDF...")

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    print(f"Loaded {len(documents)} pages.")

    # Split document
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks.")

    # Create embeddings
    embedding_model = get_embedding_model()

    print(
        f"Using {EMBEDDING_PROVIDER} embeddings..."
    )

    # Create ChromaDB
    Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=CHROMA_DIR
    )

    print("ChromaDB created successfully.")

    return len(documents), len(chunks)