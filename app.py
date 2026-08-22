import os
import tempfile

import streamlit as st

from ingest import create_vector_database
from rag import ask_question


# Page Configuration
st.set_page_config(
    page_title="RAG Book Assistant",
    page_icon="📚",
    layout="centered"
)


# Header
st.title("📚 RAG Book Assistant")

st.write(
    "Upload a PDF and ask questions from its content."
)


# PDF Upload
uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)


# Process PDF
if uploaded_file:

    st.success(
        f"Uploaded: {uploaded_file.name}"
    )

    if st.button(
        "Create Vector Database",
        type="primary"
    ):

        with st.spinner(
            "Processing PDF and creating ChromaDB..."
        ):

            try:

                # Create temporary PDF
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as temp_file:

                    temp_file.write(
                        uploaded_file.getbuffer()
                    )

                    temp_pdf_path = temp_file.name

                # Create ChromaDB
                pages, chunks = create_vector_database(
                    temp_pdf_path
                )

                # Remove temporary PDF
                os.remove(temp_pdf_path)

                st.session_state["db_created"] = True
                st.session_state["file_name"] = (
                    uploaded_file.name
                )

                st.success(
                    "✅ Vector database created successfully!"
                )

                st.info(
                    f"Processed {pages} pages into "
                    f"{chunks} chunks."
                )

            except Exception as e:

                st.error(
                    f"Error processing PDF: {str(e)}"
                )


# Question Answering
if os.path.exists("chroma_db"):

    st.divider()

    st.subheader(
        "💬 Ask Questions From Your Book"
    )

    query = st.text_input(
        "Enter your question",
        placeholder="What is deep learning?"
    )

    if query:

        with st.spinner(
            "Searching the document..."
        ):

            try:

                answer = ask_question(query)

                st.subheader("🤖 AI Answer")

                st.write(answer)

            except Exception as e:

                st.error(
                    f"Error generating answer: {str(e)}"
                )