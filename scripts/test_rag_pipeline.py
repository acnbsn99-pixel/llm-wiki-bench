#!/usr/bin/env python3
"""Test script for RAG pipeline.

Loads 1 document, ingests it, asks 1 question, and prints:
- Answer
- Retrieval count
- Token usage
- Latency
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.models import Document, DocumentPage, Question, QuestionType, AnswerType
from rag.pipeline import RAGPipeline
from rag.chunker import Chunker


def create_sample_document() -> Document:
    """Create a sample document for testing."""
    # Sample document simulating healthcare domain
    doc = Document(
        doc_id="test_doc_001",
        domain="healthcare"
    )
    
    # Create sample pages (simulating PNG paths)
    for i in range(1, 4):  # 3 pages
        page = DocumentPage(
            image_path=f"images/healthcare/test_doc_001/test_doc_001_page_{i:04d}.png",
            page_number=i
        )
        doc.pages.append(page)
    
    return doc


def create_sample_question(document: Document) -> Question:
    """Create a sample question for testing."""
    return Question(
        question_id="test_q_001",
        text="What is the recommended dosage for this medication?",
        question_type=QuestionType.FACTUAL_RETRIEVAL,
        answer_type=AnswerType.TEXT_ONLY,
        ground_truth_answer="The recommended dosage is 500mg twice daily.",
        gt_image_paths=["images/healthcare/test_doc_001/test_doc_001_page_0002.png"],
        domain="healthcare",
        document=document
    )


def main():
    """Run the RAG pipeline test."""
    print("=" * 60)
    print("RAG Pipeline Test")
    print("=" * 60)
    
    # Check environment variables
    api_base = os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_base or not api_key:
        print("\n⚠️  Warning: OPENAI_BASE_URL or OPENAI_API_KEY not set.")
        print("Please set these environment variables before running.")
        print("\nExample:")
        print("  export OPENAI_BASE_URL=https://your-endpoint.com/v1")
        print("  export OPENAI_API_KEY=your-api-key")
        sys.exit(1)
    
    # Create sample document and question
    print("\n1. Creating sample document...")
    document = create_sample_document()
    print(f"   - Document ID: {document.doc_id}")
    print(f"   - Domain: {document.domain}")
    print(f"   - Pages: {document.page_count}")
    
    print("\n2. Creating sample question...")
    question = create_sample_question(document)
    print(f"   - Question ID: {question.question_id}")
    print(f"   - Question: {question.text[:80]}...")
    
    # Initialize RAG pipeline
    print("\n3. Initializing RAG pipeline...")
    print(f"   - API Base: {api_base}")
    print(f"   - Embedding model: text-embedding-3-small")
    print(f"   - Retrieval k: 5")
    
    try:
        pipeline = RAGPipeline(
            chunker=Chunker(mode="page"),
            k=5,
            api_base=api_base,
            api_key=api_key
        )
    except ImportError as e:
        print(f"\n❌ Error importing dependencies: {e}")
        print("Make sure faiss-cpu and openai are installed:")
        print("  pip install faiss-cpu openai")
        sys.exit(1)
    
    # Ingest document
    print("\n4. Ingesting document...")
    # For testing without OCR, we simulate page content
    ocr_texts = [
        "Page 1: Introduction to the medication. This drug is used for treating various conditions.",
        "Page 2: Dosage Information. The recommended dosage is 500mg twice daily with food.",
        "Page 3: Side effects and warnings. Common side effects include nausea and headache."
    ]
    
    chunk_count = pipeline.ingest_document(document, ocr_texts=ocr_texts)
    print(f"   - Created {chunk_count} chunks")
    print(f"   - Vector store size: {len(pipeline.vector_store)} chunks")
    
    # Query the pipeline
    print("\n5. Querying the pipeline...")
    print(f"   - Question: {question.text}")
    
    result = pipeline.query(question)
    
    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    print(f"\n📝 Answer:")
    print(f"   {result.predicted_answer}")
    
    print(f"\n📊 Metrics:")
    print(f"   - Retrieval Count: {result.retrieval_count}")
    print(f"   - Token Usage: {result.token_usage}")
    print(f"   - Latency: {result.latency_seconds:.2f} seconds")
    print(f"   - Pipeline: {result.pipeline_name}")
    
    if result.trajectory and "retrieved_chunks" in result.trajectory:
        print(f"\n🔍 Retrieved Chunks:")
        for chunk_info in result.trajectory["retrieved_chunks"]:
            print(f"   - Doc: {chunk_info['doc_id']}, Page: {chunk_info['page_number']}, Score: {chunk_info['score']:.4f}")
            print(f"     Preview: {chunk_info['content_preview']}")
    
    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
