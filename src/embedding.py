import numpy as np
from typing import List, Any
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


class EmbeddingPipeline:
    """
    Handles document chunking and embedding generation using SentenceTransformer.
    """

    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        """
        Initialize the embedding pipeline.

        Args:
            model_name (str): SentenceTransformer model name.
            chunk_size (int): Size of each text chunk.
            chunk_overlap (int): Overlap between adjacent chunks.
        """

        print(f"[EmbeddingPipeline] Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        print(f"[EmbeddingPipeline] Chunk Size = {chunk_size}, Overlap = {chunk_overlap}")


    def chunk_documents(self, documents: List[Any]) -> List[Any]:
        """
        Split documents into smaller chunks suitable for embedding.

        Args:
            documents: List of LangChain Document objects.

        Returns:
            List of chunked Document objects.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
            length_function=len
        )

        print(f"[EmbeddingPipeline] Splitting {len(documents)} documents...")

        chunks = text_splitter.split_documents(documents)

        print(f"[EmbeddingPipeline] Created {len(chunks)} chunks.")
        
        # Display one example
        if len(chunks) > 0:
            print("\nExample Chunk:")
            print(f"Content: {chunks[0].page_content[:200]}...")
            print(f"Metadata: {chunks[0].metadata}")

        return chunks

    def embed_chunks(self, chunks: List[Any]) -> np.ndarray:
        """
        Generate embeddings for list of chunked documents.

        Args:
            chunks (List[Document]): List of Document chunks.

        Returns:
            np.ndarray: Embeddings with shape (num_chunks, embedding_dim)
        """
        texts = [doc.page_content for doc in chunks]
        print(f"[EmbeddingPipeline] Generating embeddings for {len(texts)} chunks...")

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True
        )
        print(f"[EmbeddingPipeline] Embedding shape: {embeddings.shape}")
        return embeddings
