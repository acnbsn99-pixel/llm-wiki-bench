"""RAG Pipeline module for llm-vs-rag-bench.

This module implements a Retrieval-Augmented Generation pipeline designed for
the UniDoc-Bench dataset, which contains multimodal documents (PNG page images).

Based on docs/rag_design.md:
- Chunking: Page-level (1 page = 1 chunk)
- Vector Store: FAISS with OpenAI-compatible embeddings
- Retrieval: Dense vector similarity search
- Generation: LLM-based answer synthesis with citations
"""

from .chunker import DocumentChunk, Chunker
from .vector_store import VectorStore, FAISSVectorStore
from .pipeline import RAGPipeline

__all__ = [
    "DocumentChunk",
    "Chunker",
    "VectorStore",
    "FAISSVectorStore",
    "RAGPipeline",
]
