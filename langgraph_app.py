# langgraph_app.py

from langgraph.graph import StateGraph, END
from backend.chains import get_qa_chain
from typing import TypedDict

# Define the structure of state dictionary passed between nodes
class GraphState(TypedDict):
    question: str
    docs: list
    answer: str

# Build the LangGraph flow with 2 nodes: input -> retrieve
def build_graph(qa_chain):
    graph = StateGraph(GraphState)

    # Node: Just prints the incoming question (logging/debugging)
    def input_node(state):
        print("🔹 Input received:", state.get("question", "[Missing 'question']"))
        return state

    # Node: Retrieves answer using the QA chain
    def retrieve_node(state):
        result = qa_chain({"query": state["question"]})
        state["answer"] = result["result"]
        state["docs"] = result["source_documents"]
        return state

    # Assemble the graph
    graph.add_node("input", input_node)
    graph.add_node("retrieve", retrieve_node)
    graph.set_entry_point("input")
    graph.add_edge("input", "retrieve")
    graph.add_edge("retrieve", END)

    return graph.compile()

# Optional CLI for testing via terminal
if __name__ == "__main__":
    from backend.rag_utils import load_vectorstore
    retriever = load_vectorstore().as_retriever()
    qa_chain = get_qa_chain(retriever)
    flow = build_graph(qa_chain)

    while True:
        user_query = input("\n💬 Ask a question (or type 'exit' to quit): ")
        if user_query.lower().strip() == "exit":
            print("👋 Exiting... Have a great day!")
            break

        output = flow.invoke({"question": user_query})
        print("🔎 Answer:", output.get("answer"))
