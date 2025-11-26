import os
import pickle
import faiss
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from src.embedding import EmbeddingPipeline


class FaissVectorStore:
    """
    FAISS-based vector store with metadata persistence.
    Stores:
      - FAISS index (embeddings)
      - Metadata for each vector
    """
    def __init__(self, 
                 persist_path: str = "faiss_store",     embedding_model: str ="all-MiniLM-L6-v2", chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        """
        Args:
            persist_path: Folder path where FAISS + metadata files will be saved.
            embedding_model: Instance of EmbeddingPipeline for generating embeddings.
            chunk_size (int): Size of each text chunk.
            chunk_overlap (int): Overlap between adjacent chunks.
        """
        self.persist_path = persist_path
        os.makedirs(self.persist_path, exist_ok=True)

        self.faiss_path = os.path.join(self.persist_path, "faiss.index")
        self.meta_path = os.path.join(self.persist_path, "metadata.pkl")

        self.embedding_model = embedding_model
        self.model = SentenceTransformer(embedding_model)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.index = None
        self.metadata = []
        self.dim = None


    def build_from_documents(self, documents: List[Any]) -> None:
        """
        Takes LangChain documents, chunks them, embeds them,
        and builds a FAISS index.

        Args:
            documents: List of LangChain Document objects.
        """

        print("[FaissVectorStore] Building index from documents...")
        emb_pipe = EmbeddingPipeline(model_name=self.embedding_model, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)

        #Chunk documents
        chunks = emb_pipe.chunk_documents(documents)
        print(f"[INFO] Total chunks: {len(chunks)}")

        #Generate embeddings
        embeddings = emb_pipe.embed_chunks(chunks)  # np.ndarray

        # Store metadata for each chunk
        metadata_list = [{"text":c.page_content} for c in chunks]

        #Add embeddings to store
        self.add_embeddings(np.array(embeddings).astype('float32'), metadata_list)

        #Save FAISS index + metadata
        self.save()
        print("[FaissVectorStore] Build complete.")


    def add_embeddings(self, embeddings: np.ndarray, metadata_list: List[Dict]):
        """
        Add embeddings to FAISS index and append metadata.

        Args:
            embeddings: np.ndarray shape (N, dim)
            metadata_list: list of metadata dicts
        """
        print(f"[FaissVectorStore] Adding {len(embeddings)} embeddings...")

        if self.index is None:
            # Create FAISS index
            self.dim = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(self.dim)
            print(f"[INIT] Created FAISS index of dim={self.dim}")

        # Add to index
        self.index.add(embeddings)

        # Add metadata
        self.metadata.extend(metadata_list)

        print("[INFO] Embeddings added to FAISS index.")


    def save(self):
        """Save FAISS index and metadata to disk."""

        print("[FaissVectorStore] Saving FAISS index & metadata...")

        if self.index is None:
            print("[WARN] No index to save.")
            return

        faiss.write_index(self.index, self.faiss_path)

        with open(self.meta_path, "wb") as f:
            pickle.dump(self.metadata, f)

        print(f"[SAVED] FAISS index -> {self.faiss_path}")
   

    def load(self):
        """Load FAISS index + metadata from disk."""

        print("[FaissVectorStore] Loading FAISS index & metadata...")

        if not os.path.exists(self.faiss_path):
            raise FileNotFoundError("FAISS index not found")

        self.index = faiss.read_index(self.faiss_path)

        with open(self.meta_path, "rb") as f:
            self.metadata = pickle.load(f)

        print(f"[LOADED] Loaded {len(self.metadata)} metadata entries.")

  
    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        """
        Perform a FAISS search.

        Args:
            query_embedding: np.ndarray shape (dim,)
        """
        if self.index is None:
            raise RuntimeError("FAISS index is not loaded or built.")

        distances, indices = self.index.search(query_embedding, top_k)
        results = []
        for idx, dist in zip(indices[0], distances[0]):
          meta = self.metadata[idx] if idx <len(self.metadata) else None
          results.append({"index":idx, "distance":dist, "metadata":meta})
  
        return results


    def query(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Full retrieval pipeline:
        - Embed query
        - FAISS search
        - Return matched docs + metadata + scores
        """
        print(f"[RAG] Querying for: {query}")

        query_emb = self.model.encode([query]).astype('float32')

        return self.search(query_emb, top_k=top_k)
