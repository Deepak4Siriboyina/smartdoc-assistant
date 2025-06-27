# Imports for loading PDF documents, splitting text, embeddings, and vector store
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
import os
import tempfile

def embed_and_store(pdf_file_path: str):
    """
    Loads a PDF, embeds its contents, and returns an in-memory FAISS vector store.

    Args:
        pdf_file_path (str): Path to the uploaded PDF file.

    Returns:
        FAISS: A vector store object containing embedded document chunks.
    """
    loader = PyMuPDFLoader(pdf_file_path)  # Load PDF using PyMuPDF
    documents = loader.load()  # Extract text into LangChain Document format

    embedder = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # Google Gemini embeddings

    # Use a temporary directory to avoid saving files permanently on disk
    with tempfile.TemporaryDirectory() as tmpdir:
        vectorstore = FAISS.from_documents(documents, embedder)  # Create FAISS index from documents
        vectorstore.save_local(tmpdir)  # Save temporarily
        index = FAISS.load_local(tmpdir, embedder, allow_dangerous_deserialization=True)  # Reload as FAISS object
        return index  # Return the full vectorstore (not just retriever)


# ---------- Optional utility functions below (not used in current Streamlit flow) ---------- #

# Import configuration constants (for embedding model and FAISS storage path)
from backend.config import (
    EMBEDDING_MODEL_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    VECTOR_STORE_PATH
)

def load_pdf_chunks(file_path):
    """
    Splits a PDF into text chunks using RecursiveCharacterTextSplitter.

    Args:
        file_path (str): Path to the PDF file.

    Returns:
        list: A list of text chunks in Document format.
    """
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,        # Defined in config (e.g. 500 characters)
        chunk_overlap=CHUNK_OVERLAP   # e.g. 50 character overlap between chunks
    )
    return splitter.split_documents(documents)

def get_embedder():
    """
    Returns the embedding model used for generating document embeddings.

    Returns:
        GoogleGenerativeAIEmbeddings: Embedding object for Gemini model.
    """
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001")

def store_to_faiss(chunks, persist_path=VECTOR_STORE_PATH):
    """
    Stores a list of text chunks to FAISS vectorstore and saves it to disk.

    Args:
        chunks (list): List of document chunks.
        persist_path (str): Path where FAISS index will be stored.

    Returns:
        FAISS: Stored vector store object.
    """
    embedder = get_embedder()
    vectorstore = FAISS.from_documents(chunks, embedder)
    vectorstore.save_local(persist_path)
    return vectorstore

def load_vectorstore(persist_path=VECTOR_STORE_PATH):
    """
    Loads a FAISS vectorstore from disk.

    Args:
        persist_path (str): Path to previously saved FAISS index.

    Returns:
        FAISS: Loaded vector store object.
    """
    embedder = get_embedder()
    return FAISS.load_local(persist_path, embedder, allow_dangerous_deserialization=True)
