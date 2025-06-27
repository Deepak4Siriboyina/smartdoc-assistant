# test_embed_and_store.py

# 🧪 Test script for verifying the PDF embedding and FAISS vectorstore setup
# This script helps ensure that embedding, storage, and retrieval logic works as expected
# Usage: python test_embed_and_store.py

import os
from backend.rag_utils import embed_and_store, load_vectorstore

def test_embed_and_store_with_sample():
    # 🔹 Path to sample test PDF (ensure it's placed in root for testing)
    sample_pdf = "sample.pdf"
    
    # ❗ Check if file exists
    if not os.path.exists(sample_pdf):
        print(f"❌ Sample file not found: {sample_pdf}")
        return

    # 🔹 Try embedding and storing the document
    try:
        print("📄 Embedding sample.pdf...")
        embed_and_store(sample_pdf)
        print("✅ Embedding and storage successful.")
    except Exception as e:
        print(f"❌ Failed during embedding: {e}")
        return

    # 🔹 Try retrieving relevant documents using vectorstore
    try:
        print("📦 Loading vectorstore...")
        vs = load_vectorstore()
        retriever = vs.as_retriever()
        docs = retriever.get_relevant_documents("What is this document about?")

        print(f"✅ Retrieved {len(docs)} relevant docs.")
        for i, doc in enumerate(docs):
            print(f"\n--- Doc {i+1} ---\n{doc.page_content[:300]}...\n")
    except Exception as e:
        print(f"❌ Failed to load vectorstore: {e}")

# 🔧 Entry point for running the test
if __name__ == "__main__":
    test_embed_and_store_with_sample()
