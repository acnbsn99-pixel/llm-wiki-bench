"""RAG Pipeline module.

Orchestrates the full RAG workflow:
1. Ingest documents → chunk → embed → store
2. On query: retrieve → build prompt → generate answer

Returns BenchmarkResult dataclass with architecture="rag", answer, token usage, latency, retrieval count.
"""

import time
from typing import List, Optional
from pathlib import Path

from ..data.models import Document, Question, BenchmarkResult
from ..llm_client import LLMClient
from .chunker import Chunker, DocumentChunk
from .vector_store import FAISSVectorStore, RetrievalResult


class RAGPipeline:
    """RAG pipeline for UniDoc-Bench benchmark.
    
    Implements the architecture from docs/rag_design.md:
    - Page-level chunking (1 page = 1 chunk)
    - FAISS vector store with OpenAI-compatible embeddings
    - Dense retrieval with top-k results
    - LLM-based answer generation with citations
    
    Attributes:
        chunker: Document chunker
        vector_store: Vector store for chunk embeddings
        llm_client: LLM client for answer generation
        k: Number of chunks to retrieve per query
    """
    
    def __init__(self,
                 chunker: Optional[Chunker] = None,
                 vector_store: Optional[FAISSVectorStore] = None,
                 llm_client: Optional[LLMClient] = None,
                 k: int = 5,
                 api_base: Optional[str] = None,
                 api_key: Optional[str] = None,
                 embedding_model: str = "text-embedding-3-small"):
        """Initialize the RAG pipeline.
        
        Args:
            chunker: Document chunker (uses default page-level if None)
            vector_store: Vector store (creates new FAISS store if None)
            llm_client: LLM client (creates new client if None)
            k: Number of chunks to retrieve
            api_base: Base URL for OpenAI-compatible API
            api_key: API key for the API
            embedding_model: Name of embedding model
        """
        self.chunker = chunker or Chunker(mode="page")
        self.k = k
        self.api_base = api_base
        self.api_key = api_key
        
        # Initialize vector store
        if vector_store is None:
            self.vector_store = FAISSVectorStore(
                embedding_model=embedding_model,
                api_base=api_base,
                api_key=api_key
            )
        else:
            self.vector_store = vector_store
        
        # Initialize LLM client
        self.llm_client = llm_client or LLMClient()
        
        # Track ingested documents
        self._ingested_doc_ids = set()
    
    def ingest_document(self, document: Document, 
                        ocr_texts: Optional[List[str]] = None) -> int:
        """Ingest a document into the RAG pipeline.
        
        Chunks the document, embeds chunks, and stores in vector index.
        
        Args:
            document: Document to ingest
            ocr_texts: Optional OCR-extracted texts (one per page)
            
        Returns:
            Number of chunks created
        """
        # Prepare pages data
        pages = [
            {"image_path": page.image_path, "page_number": page.page_number}
            for page in document.pages
        ]
        
        # Chunk the document
        chunks = self.chunker.chunk_document(
            doc_id=document.doc_id,
            domain=document.domain,
            pages=pages,
            ocr_texts=ocr_texts
        )
        
        # Add to vector store (embeddings generated automatically)
        self.vector_store.add_chunks(chunks)
        
        # Track ingestion
        self._ingested_doc_ids.add(document.doc_id)
        
        return len(chunks)
    
    def ingest_documents(self, documents: List[Document],
                         ocr_texts_map: Optional[dict] = None) -> int:
        """Ingest multiple documents.
        
        Args:
            documents: List of documents to ingest
            ocr_texts_map: Optional dict mapping doc_id to list of OCR texts
            
        Returns:
            Total number of chunks created
        """
        total_chunks = 0
        for doc in documents:
            ocr_texts = None
            if ocr_texts_map and doc.doc_id in ocr_texts_map:
                ocr_texts = ocr_texts_map[doc.doc_id]
            chunks = self.ingest_document(doc, ocr_texts)
            total_chunks += chunks
        return total_chunks
    
    def query(self, question: Question) -> BenchmarkResult:
        """Query the RAG pipeline with a question.
        
        Retrieves relevant chunks, builds prompt, generates answer.
        
        Args:
            question: Question object from UniDoc-Bench
            
        Returns:
            BenchmarkResult with answer, token usage, latency, retrieval count
        """
        start_time = time.perf_counter()
        
        # Reset LLM stats for this query
        self.llm_client.reset_stats()
        
        # Retrieve relevant chunks
        retrieval_results = self.vector_store.search_by_text(
            question.text,
            k=self.k
        )
        
        # Build context from retrieved chunks
        context = self._build_context(retrieval_results)
        
        # Build prompt
        prompt = self._build_prompt(question.text, context)
        
        # Generate answer
        system_message = (
            "You are a helpful assistant that answers questions based on the provided context. "
            "If the context doesn't contain enough information to answer, say so. "
            "Always cite which page(s) your answer comes from when possible."
        )
        
        result = self.llm_client.call(
            prompt=prompt,
            system_message=system_message,
            max_tokens=1024
        )
        
        # Calculate latency
        end_time = time.perf_counter()
        latency_seconds = end_time - start_time
        
        # Get token usage from LLM client
        stats = self.llm_client.get_stats()
        token_usage = stats.total_tokens
        
        # Create BenchmarkResult
        benchmark_result = BenchmarkResult(
            pipeline_name="rag",
            question_id=question.question_id,
            predicted_answer=result.content,
            latency_seconds=latency_seconds,
            token_usage=token_usage,
            retrieval_count=len(retrieval_results),
            trajectory={
                "retrieved_chunks": [
                    {
                        "chunk_id": r.chunk.chunk_id,
                        "doc_id": r.chunk.doc_id,
                        "page_number": r.chunk.page_number,
                        "score": r.score,
                        "content_preview": r.chunk.content[:200] + "..." if len(r.chunk.content) > 200 else r.chunk.content
                    }
                    for r in retrieval_results
                ]
            }
        )
        
        return benchmark_result
    
    def _build_context(self, retrieval_results: List[RetrievalResult]) -> str:
        """Build context string from retrieval results.
        
        Args:
            retrieval_results: List of retrieval results
            
        Returns:
            Formatted context string
        """
        if not retrieval_results:
            return "No relevant context found."
        
        context_parts = []
        for i, result in enumerate(retrieval_results):
            chunk = result.chunk
            part = f"""
[Source {i+1}]
Document: {chunk.doc_id}
Page: {chunk.page_number}
Domain: {chunk.metadata.get('domain', 'unknown')}
Content: {chunk.content}
""".strip()
            context_parts.append(part)
        
        return "\n\n".join(context_parts)
    
    def _build_prompt(self, question_text: str, context: str) -> str:
        """Build the final prompt for the LLM.
        
        Args:
            question_text: The question text
            context: Retrieved context
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""Based on the following context, answer the question below.

Context:
{context}

Question: {question_text}

Answer:"""
        return prompt
    
    def save(self, path: str) -> None:
        """Save the RAG pipeline state.
        
        Args:
            path: Directory path to save to
        """
        self.vector_store.save(path)
        
        # Save ingested doc IDs
        import json
        save_path = Path(path)
        with open(save_path / "ingested_docs.json", "w") as f:
            json.dump(list(self._ingested_doc_ids), f)
    
    @classmethod
    def load(cls, path: str,
             llm_client: Optional[LLMClient] = None,
             k: int = 5) -> "RAGPipeline":
        """Load a RAG pipeline from disk.
        
        Args:
            path: Directory path to load from
            llm_client: LLM client (creates new if None)
            k: Number of chunks to retrieve
            
        Returns:
            Loaded RAGPipeline instance
        """
        import json
        
        load_path = Path(path)
        
        # Load vector store
        vector_store = FAISSVectorStore.load(path)
        
        # Create pipeline
        pipeline = cls(
            vector_store=vector_store,
            llm_client=llm_client,
            k=k,
            api_base=vector_store.api_base,
            api_key=vector_store.api_key
        )
        
        # Load ingested doc IDs
        ingested_file = load_path / "ingested_docs.json"
        if ingested_file.exists():
            with open(ingested_file, "r") as f:
                pipeline._ingested_doc_ids = set(json.load(f))
        
        return pipeline
