"""Vector Store module for RAG pipeline.

Based on docs/rag_design.md, FAISS is recommended for:
- Speed and scalability
- Local operation (no external service dependency)
- Integration with LangChain-style workflows

This module provides:
- VectorStore abstract base class
- FAISSVectorStore implementation using OpenAI-compatible embeddings
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import pickle
from pathlib import Path

from .chunker import DocumentChunk


@dataclass
class RetrievalResult:
    """Result of a retrieval operation.
    
    Attributes:
        chunk: The retrieved document chunk
        score: Similarity score (higher = more similar)
        rank: Rank in the retrieval results (1-indexed)
    """
    chunk: DocumentChunk
    score: float
    rank: int


class VectorStore(ABC):
    """Abstract base class for vector stores.
    
    Defines the interface for storing and retrieving document chunks.
    """
    
    @abstractmethod
    def add_chunks(self, chunks: List[DocumentChunk], embeddings: List[List[float]]) -> None:
        """Add chunks with their embeddings to the store.
        
        Args:
            chunks: List of document chunks
            embeddings: List of embedding vectors (one per chunk)
        """
        pass
    
    @abstractmethod
    def search(self, query_embedding: List[float], k: int = 5) -> List[RetrievalResult]:
        """Search for similar chunks.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            
        Returns:
            List of RetrievalResult objects
        """
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save the vector store to disk.
        
        Args:
            path: Directory path to save to
        """
        pass
    
    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "VectorStore":
        """Load a vector store from disk.
        
        Args:
            path: Directory path to load from
            
        Returns:
            Loaded VectorStore instance
        """
        pass


class FAISSVectorStore(VectorStore):
    """FAISS-based vector store with OpenAI-compatible embeddings.
    
    Uses faiss-cpu for local, fast similarity search.
    Embeddings are generated via OpenAI-compatible API.
    """
    
    def __init__(self, embedding_model: str = "text-embedding-3-small",
                 api_base: Optional[str] = None,
                 api_key: Optional[str] = None):
        """Initialize the FAISS vector store.
        
        Args:
            embedding_model: Name of the embedding model
            api_base: Base URL for OpenAI-compatible API
            api_key: API key for the embedding service
        """
        self.embedding_model = embedding_model
        self.api_base = api_base
        self.api_key = api_key
        
        # Storage
        self.chunks: List[DocumentChunk] = []
        self.embeddings: List[List[float]] = []
        self._index = None  # FAISS index (created on first add)
        
        # Try to import faiss
        try:
            import faiss
            self._faiss = faiss
        except ImportError:
            raise ImportError(
                "FAISS is required. Install with: pip install faiss-cpu"
            )
        
        # Initialize embedding client
        self._init_embedding_client()
    
    def _init_embedding_client(self):
        """Initialize the embedding client."""
        from openai import OpenAI
        
        self._embed_client = OpenAI(
            base_url=self.api_base,
            api_key=self.api_key
        )
    
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        response = self._embed_client.embeddings.create(
            input=text,
            model=self.embedding_model
        )
        return response.data[0].embedding
    
    def _get_embeddings_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Get embeddings for multiple texts in batches.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for API calls
            
        Returns:
            List of embedding vectors
        """
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self._embed_client.embeddings.create(
                input=batch,
                model=self.embedding_model
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    
    def add_chunks(self, chunks: List[DocumentChunk], 
                   embeddings: Optional[List[List[float]]] = None) -> None:
        """Add chunks to the vector store.
        
        Args:
            chunks: List of document chunks
            embeddings: Optional pre-computed embeddings. If None, generates embeddings.
        """
        if not chunks:
            return
        
        # Generate embeddings if not provided
        if embeddings is None:
            texts = [chunk.content for chunk in chunks]
            embeddings = self._get_embeddings_batch(texts)
        
        # Convert to numpy arrays for FAISS
        import numpy as np
        embedding_array = np.array(embeddings, dtype=np.float32)
        
        # Create or update FAISS index
        dimension = len(embeddings[0])
        
        if self._index is None:
            # Create new index
            self._index = self._faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)
        
        # Add embeddings to index
        self._index.add(embedding_array)
        
        # Store chunks and embeddings
        self.chunks.extend(chunks)
        self.embeddings.extend([e.tolist() if hasattr(e, 'tolist') else e for e in embeddings])
    
    def search(self, query_embedding: List[float], k: int = 5) -> List[RetrievalResult]:
        """Search for similar chunks.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            
        Returns:
            List of RetrievalResult objects
        """
        if self._index is None or len(self.chunks) == 0:
            return []
        
        import numpy as np
        
        # Ensure query is 2D array
        query_array = np.array([query_embedding], dtype=np.float32)
        
        # Search
        k_actual = min(k, len(self.chunks))
        scores, indices = self._index.search(query_array, k_actual)
        
        # Build results
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.chunks):
                result = RetrievalResult(
                    chunk=self.chunks[idx],
                    score=float(score),
                    rank=i + 1
                )
                results.append(result)
        
        return results
    
    def search_by_text(self, query_text: str, k: int = 5) -> List[RetrievalResult]:
        """Search by text query (convenience method).
        
        Embeds the query text and performs search.
        
        Args:
            query_text: Query text
            k: Number of results to return
            
        Returns:
            List of RetrievalResult objects
        """
        query_embedding = self._get_embedding(query_text)
        return self.search(query_embedding, k)
    
    def save(self, path: str) -> None:
        """Save the vector store to disk.
        
        Saves:
        - FAISS index
        - Chunks metadata
        - Embeddings
        
        Args:
            path: Directory path to save to
        """
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        if self._index is not None:
            self._faiss.write_index(self._index, str(save_path / "index.faiss"))
        
        # Save chunks
        with open(save_path / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)
        
        # Save embeddings
        with open(save_path / "embeddings.pkl", "wb") as f:
            pickle.dump(self.embeddings, f)
        
        # Save config
        config = {
            "embedding_model": self.embedding_model,
            "api_base": self.api_base,
        }
        with open(save_path / "config.json", "w") as f:
            import json
            json.dump(config, f)
    
    @classmethod
    def load(cls, path: str) -> "FAISSVectorStore":
        """Load a FAISS vector store from disk.
        
        Args:
            path: Directory path to load from
            
        Returns:
            Loaded FAISSVectorStore instance
        """
        import json
        
        save_path = Path(path)
        
        # Load config
        with open(save_path / "config.json", "r") as f:
            config = json.load(f)
        
        # Create instance
        store = cls(
            embedding_model=config.get("embedding_model", "text-embedding-3-small"),
            api_base=config.get("api_base")
        )
        
        # Load chunks
        with open(save_path / "chunks.pkl", "rb") as f:
            store.chunks = pickle.load(f)
        
        # Load embeddings
        with open(save_path / "embeddings.pkl", "rb") as f:
            store.embeddings = pickle.load(f)
        
        # Load FAISS index
        index_path = save_path / "index.faiss"
        if index_path.exists():
            store._index = store._faiss.read_index(str(index_path))
        
        return store
    
    def __len__(self) -> int:
        """Return number of chunks in the store."""
        return len(self.chunks)
