import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env

# API keys
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


# Embedding model for FAISS
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# PDF splitting settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# FAISS DB path
VECTOR_STORE_PATH = "backend/vector_store/index"
