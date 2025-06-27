# Import necessary components for building LLM-based chains
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI  # Gemini chat model from Google

# Legacy import for Groq, currently not used
# from langchain.chat_models import ChatGroq
# from backend.config import GROQ_API_KEY


# Optional: Legacy Groq model setup (unused in current flow)
'''
def get_llm():
    return ChatGroq(
        temperature=0.3,
        model_name="mixtral-8x7b-32768",
        api_key=GROQ_API_KEY
    )
'''


def get_llm():
    """
    Returns a Google Gemini 2.5 Flash model instance with controlled temperature.

    Returns:
        ChatGoogleGenerativeAI: Google-hosted chat model with low randomness.
    """
    return ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash",  # Fast & affordable model
        temperature=0.3                   # Slight creativity, mostly deterministic
    )


def get_qa_chain(retriever):
    """
    Constructs a RetrievalQA chain that retrieves documents from a vector store
    and generates answers using the Gemini model.

    Args:
        retriever: The retriever (usually from FAISS) used to fetch relevant chunks.

    Returns:
        RetrievalQA: A LangChain chain that performs retrieval-augmented Q&A.
    """
    llm = get_llm()
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",                 # Basic chain that stuffs docs into context
        retriever=retriever,
        return_source_documents=True        # Useful for debugging / validation
    )


def get_summary_chain():
    """
    Constructs a simple LLM chain to summarize any given text response.

    Returns:
        LLMChain: LangChain chain that returns a summary using Gemini.
    """
    prompt = PromptTemplate.from_template(
        "Summarize the following answer:\n\n{input}"
    )
    return LLMChain(llm=get_llm(), prompt=prompt)
