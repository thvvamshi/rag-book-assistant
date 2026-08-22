import os

from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()


# Configuration
CHROMA_DIR = "chroma_db"

# Must match ingest.py
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


# Load Vector Database
def get_vectorstore():

    embedding_model = get_embedding_model()

    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embedding_model
    )


# LLM
llm = ChatMistralAI(
    model="mistral-small-2506"
)


# Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.

            Use ONLY the provided context to answer the question.

            If the answer is not present in the context,
            say exactly:

            "I could not find the answer in the document."
            """
        ),
        (
            "human",
            """Context:
            {context}

            Question:
            {question}
            """
        )
    ]
)


# Ask Question
def ask_question(query):

    vectorstore = get_vectorstore()

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.5
        }
    )

    documents = retriever.invoke(query)

    if not documents:
        return "I could not find the answer in the document."

    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    final_prompt = prompt.invoke(
        {
            "context": context,
            "question": query
        }
    )

    response = llm.invoke(final_prompt)

    return response.content