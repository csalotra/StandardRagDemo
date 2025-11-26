from src.data_loader import load_data
from src.vectorstore import FaissVectorStore
from src.search import RAGSearch

if __name__ == "__main__":
    docs = load_data("data")
    rag = RAGSearch()
    store=FaissVectorStore("faiss_store")
    # store.build_from_documents(docs)
    # store.load()
    # print(store.query("What is AI agent?", top_k=3))

    answer = rag.search_and_summarize("How to build AI Agent?", top_k=3)
    print( "Summarized Answer: ", answer)
