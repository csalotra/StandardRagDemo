import os
from dotenv import load_dotenv
from src.vectorstore import FaissVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

class RAGSearch:
    def __init__(
        self,
        persist_dir: str = "faiss_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "gemini-2.5-flash"
    ):
        """
        RAG Search: Loads FAISS and initializes LLM
        """
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)

        
        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")

        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            from src.data_loader import load_data
            docs = load_data("data")
            self.vectorstore.build_from_documents(docs)
        else:
            self.vectorstore.load()

        api_key = os.getenv("GEMINI_API_KEY")
        self.llm = ChatGoogleGenerativeAI(model=llm_model, api_key=api_key)
        print(f"[INFO] Gemini LLM initialized: {llm_model}")

    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        """
        Search FAISS + summarize using Gemini LLM
        """
        results = self.vectorstore.query(query, top_k=top_k)

        texts = [r["metadata"].get("text", "") for r in results]
        context = "\n\n".join(texts).strip()
        if not context:
            return "No relevant documents found" 
        prompt = (
            "Use the following document context to answer the question concisely.\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )
        response = self.llm.invoke(prompt)
        return response.content
