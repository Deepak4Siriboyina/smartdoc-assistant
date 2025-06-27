# frontend/streamlit_app.py

# Streamlit frontend for SmartDoc RAG Assistant
# ---------------------------------------------
# Uploads a PDF, indexes it temporarily using embeddings (Gemini),
# allows question-answering through a chat interface using LangGraph+LangChain.

import streamlit as st
import tempfile
import sys
import os
import pandas as pd

# Ensure backend modules are importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Core logic imports
from langgraph_app import build_graph
from backend.rag_utils import embed_and_store
from backend.chains import get_qa_chain

# ---------------- Page Setup ---------------- #
st.set_page_config(page_title="SmartDoc Assistant", page_icon="📄")

st.title("📄 SmartDoc RAG Assistant")
st.markdown("🔒 **Note:** Your uploaded document is temporarily used in memory for answering questions and is deleted immediately after processing.")

# ---------------- Session Initialization ---------------- #
# Used to persist app state across reruns
if "flow" not in st.session_state:
    st.session_state.flow = None

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------- File Upload + Indexing ---------------- #
uploaded_file = st.file_uploader("📂 Upload a PDF file", type="pdf")

# Only trigger embedding if new file is uploaded
if uploaded_file and ("last_uploaded" not in st.session_state or uploaded_file.name != st.session_state.get("last_uploaded")):
    st.success("PDF uploaded successfully.")

    # Save file temporarily to disk
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    # Convert PDF to embeddings and build retriever
    retriever = embed_and_store(temp_path).as_retriever()
    os.remove(temp_path)  # Clean up immediately

    # Build graph pipeline with updated retriever
    qa_chain = get_qa_chain(retriever)
    st.session_state.qa_chain = qa_chain
    st.session_state.flow = build_graph(qa_chain=qa_chain)

    # Reset chat history for new file
    st.session_state.chat_history.clear()
    st.session_state.last_uploaded = uploaded_file.name  # Track for deduplication
    st.info("✅ Ready to answer questions from the new document!")

# ---------------- Chat UI ---------------- #
# Triggered when LangGraph flow is built and ready
if st.session_state.flow:
    user_question = st.chat_input("💬 Ask a question")

    if user_question:
        with st.spinner("🤔 Thinking..."):
            output = st.session_state.flow.invoke({"question": user_question})
            answer = output.get("answer", "No answer found.")

            # Save Q&A pair to history
            st.session_state.chat_history.append({
                "question": user_question,
                "answer": answer
            })

# ---------------- Chat History Display ---------------- #
if st.session_state.chat_history:
    st.subheader("🗂️ Q&A Chat History")

    # Expanding dropdown for past questions
    with st.expander("🔽 Click to view full history"):
        for i, qa in enumerate(st.session_state.chat_history, 1):
            st.markdown(f"**Q{i}: {qa['question']}**")
            st.markdown(f"**A{i}: {qa['answer']}**")
            st.markdown("---")

# ---------------- Download Q&A Buttons ---------------- #
if st.session_state.chat_history:
    st.subheader("📥 Download Q&A")

    col1, col2 = st.columns(2)

    # Download as .txt
    with col1:
        txt_data = "\n\n".join([
            f"Q: {qa['question']}\nA: {qa['answer']}"
            for qa in st.session_state.chat_history
        ])
        st.download_button(
            label="⬇️ Download as .txt",
            data=txt_data,
            file_name="smartdoc_qa.txt",
            mime="text/plain"
        )

    # Download as .csv
    with col2:
        df = pd.DataFrame(st.session_state.chat_history)
        st.download_button(
            label="⬇️ Download as .csv",
            data=df.to_csv(index=False),
            file_name="smartdoc_qa.csv",
            mime="text/csv"
        )
