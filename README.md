# 📄 SmartDoc Assistant – RAG-based PDF QA Chatbot

SmartDoc Assistant is an end-to-end intelligent document Q&A assistant powered by **LangGraph**, **LangChain**, and **Gemini (Google Generative AI)**. It allows users to upload any PDF file and instantly ask questions about its contents using Retrieval-Augmented Generation (RAG).

✅ Uses Google Embeddings + Gemini LLM  
✅ Summarizes & answers questions based on document context  
✅ Fully in-memory (privacy-friendly: no permanent file storage)  
✅ Deployed via Streamlit Cloud (Free Tier)

---

## 🚀 Live Demo

- **SmartDoc Assistant** 👉 [Try it on Streamlit](https://smartdoc-assistant-aepsdqriept5vuzsbcdu7v.streamlit.app/)

---

## 🛠 Tech Stack

| Layer       | Tech                        |
|-------------|-----------------------------|
| UI          | Streamlit                   |
| LLM         | Google Generative AI (Gemini 2.5 Flash) |
| Embeddings  | Google Generative AI Embeddings (`embedding-001`) |
| Vector DB   | FAISS (In-memory)           |
| Graph Flow  | LangGraph                   |
| Framework   | LangChain                   |
| PDF Parser  | PyMuPDF                     |
| Language    | Python 3.11                 |
| Hosting     | Streamlit Cloud (free tier) |

---

## 📦 Project Structure

```bash
smartdoc-assistant/
├── backend/
│   ├── chains.py              # LLMs and chain setup
│   ├── config.py              # Constants and config
│   ├── rag_utils.py           # PDF loading, embedding, vectorstore utils
├── frontend/
│   └── streamlit_app.py       # Main Streamlit UI
├── langgraph_app.py           # LangGraph workflow (input -> retrieval -> output)
├── test_embed_and_store.py    # Simple CLI test for embedding logic
├── temp/                      # Temporary file store (auto-cleared)
├── .env                       # Google API Key & Config
├── requirements.txt           # Project dependencies
└── README.md                  # You're reading it!

```

## 📁 How to Run Locally

```bash
# Clone repo
git clone https://github.com/deepak4siriboyina/smartdoc-assistant.git
cd smartdoc-assistant

# Create virtual environment
python -m venv virtenvt
virtenvt\Scripts\activate  # (Use PowerShell)

# Install dependencies
pip install -r requirements.txt

# Set your API Key
echo GOOGLE_API_KEY=your-api-key > .env

# Run the app
streamlit run frontend/streamlit_app.py
```

## 🧠 How It Works
- **User uploads a PDF** file through the Streamlit UI.
- The PDF is parsed, chunked, and embedded using **Google's** `embedding-001` model.
- The chunks are stored in a temporary **in-memory FAISS vector store**.
- When the user asks a question:
  - The **LangGraph** flow is triggered:
  - → `input` → `retrieve` → `answer`
  - A retriever fetches relevant chunks, and **Gemini 2.5 Flash** answers using `RetrievalQA`.
- All Q&A pairs are saved in the session, can be viewed via dropdown, and downloaded as `.txt` or `.csv`

## ✨ Features
- 📄 Upload any PDF and ask questions interactively.
- ⚙️ Temporary in-memory processing – no persistent storage or data leakage.
- 🧠 Uses Google's latest Gemini Flash model for fast responses.
- 🗂️ Expandable chat history with full Q&A transcripts.
- ⏬ One-click download of chat history.
- ✅ Lightweight, free to run, and private by design.

## 📤 Deployment
- Streamlit Frontend → [Streamlit Cloud](https://streamlit.io/cloud)

## 🔐 Data Privacy
- All uploaded PDFs are processed in-memory and deleted after embedding.
- No document data is permanently stored.

## 🙌 Credits
- [LangChain](https://www.langchain.com/)
- [LangGraph](https://docs.langgraph.io/)
- [Google Generative AI](https://ai.google.dev/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Streamlit](https://streamlit.io/)
- [PyMuPDF](https://pymupdf.readthedocs.io/en/latest/)

## 🧑‍💻 Author
- Deepak Siriboyina – [LinkedIn](https://www.linkedin.com/in/deepak-siriboyina/)
