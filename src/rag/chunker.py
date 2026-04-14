"""Chunking module for RAG pipeline.

Based on docs/rag_design.md, the recommended chunking strategy for UniDoc-Bench is:
- Page-level chunking: 1 page = 1 chunk (natural boundary for PNG images)
- Each chunk retains metadata: doc_id, page_number, domain

For OCR-extracted text, alternative chunking by sections/paragraphs is supported.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import hashlib


@dataclass
class DocumentChunk:
    """Represents a single chunk of a document.
    
    Attributes:
        chunk_id: Unique identifier for this chunk
        doc_id: ID of the source document
        page_number: Page number within the document (1-indexed)
        content: Text content of the chunk (OCR-extracted or placeholder)
        image_path: Path to the page image (PNG)
        metadata: Additional metadata (domain, section, etc.)
    """
    chunk_id: str
    doc_id: str
    page_number: int
    content: str
    image_path: str
    metadata: dict = field(default_factory=dict)
    
    @classmethod
    def from_page(cls, doc_id: str, page_number: int, image_path: str, 
                  text_content: Optional[str] = None, domain: str = "") -> "DocumentChunk":
        """Create a chunk from a document page.
        
        Args:
            doc_id: Document identifier
            page_number: Page number (1-indexed)
            image_path: Path to the page image
            text_content: Optional OCR-extracted text content
            domain: Domain category
            
        Returns:
            DocumentChunk instance
        """
        # Generate unique chunk ID
        chunk_key = f"{doc_id}_page_{page_number}"
        chunk_id = hashlib.md5(chunk_key.encode()).hexdigest()[:16]
        
        # Use provided text or placeholder for image-only chunks
        content = text_content or f"[Image: {image_path}]"
        
        return cls(
            chunk_id=chunk_id,
            doc_id=doc_id,
            page_number=page_number,
            content=content,
            image_path=image_path,
            metadata={"domain": domain, "image_path": image_path}
        )


class Chunker:
    """Chunks documents into retrievable units.
    
    Supports two modes based on rag_design.md recommendations:
    1. Page-level chunking (default): Each page image = 1 chunk
    2. Text-based chunking: Split OCR text by sections/paragraphs
    """
    
    def __init__(self, 
                 mode: str = "page",
                 chunk_size: int = 500,
                 chunk_overlap: int = 50):
        """Initialize the chunker.
        
        Args:
            mode: Chunking mode - "page" or "text"
            chunk_size: Target chunk size in tokens (for text mode)
            chunk_overlap: Overlap between consecutive chunks (for text mode)
        """
        self.mode = mode
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_document(self, 
                       doc_id: str,
                       domain: str,
                       pages: List[dict],
                       ocr_texts: Optional[List[str]] = None) -> List[DocumentChunk]:
        """Chunk a document into retrievable units.
        
        Args:
            doc_id: Document identifier
            domain: Domain category
            pages: List of page dicts with 'image_path' and 'page_number'
            ocr_texts: Optional list of OCR-extracted texts (one per page)
            
        Returns:
            List of DocumentChunk objects
        """
        if self.mode == "page":
            return self._chunk_by_page(doc_id, domain, pages, ocr_texts)
        elif self.mode == "text":
            return self._chunk_by_text(doc_id, domain, pages, ocr_texts)
        else:
            raise ValueError(f"Unknown chunking mode: {self.mode}")
    
    def _chunk_by_page(self,
                       doc_id: str,
                       domain: str,
                       pages: List[dict],
                       ocr_texts: Optional[List[str]] = None) -> List[DocumentChunk]:
        """Page-level chunking: 1 page = 1 chunk.
        
        This is the recommended approach for UniDoc-Bench (rag_design.md).
        """
        chunks = []
        for i, page in enumerate(pages):
            text_content = ocr_texts[i] if ocr_texts and i < len(ocr_texts) else None
            chunk = DocumentChunk.from_page(
                doc_id=doc_id,
                page_number=page.get("page_number", i + 1),
                image_path=page["image_path"],
                text_content=text_content,
                domain=domain
            )
            chunks.append(chunk)
        return chunks
    
    def _chunk_by_text(self,
                       doc_id: str,
                       domain: str,
                       pages: List[dict],
                       ocr_texts: Optional[List[str]] = None) -> List[DocumentChunk]:
        """Text-based chunking: split OCR text by sections/paragraphs.
        
        Falls back to page-level if no OCR text is available.
        """
        if not ocr_texts:
            # No OCR text, fall back to page-level
            return self._chunk_by_page(doc_id, domain, pages, None)
        
        chunks = []
        for i, page in enumerate(pages):
            page_text = ocr_texts[i] if i < len(ocr_texts) else ""
            if not page_text:
                # Empty OCR, create placeholder chunk
                chunk = DocumentChunk.from_page(
                    doc_id=doc_id,
                    page_number=page.get("page_number", i + 1),
                    image_path=page["image_path"],
                    text_content=None,
                    domain=domain
                )
                chunks.append(chunk)
            else:
                # Split text into chunks
                text_chunks = self._split_text(page_text)
                for j, text_chunk in enumerate(text_chunks):
                    chunk = DocumentChunk(
                        chunk_id=f"{doc_id}_page{i+1}_sub{j}",
                        doc_id=doc_id,
                        page_number=page.get("page_number", i + 1),
                        content=text_chunk,
                        image_path=page["image_path"],
                        metadata={
                            "domain": domain,
                            "image_path": page["image_path"],
                            "sub_chunk": j,
                            "total_sub_chunks": len(text_chunks)
                        }
                    )
                    chunks.append(chunk)
        return chunks
    
    def _split_text(self, text: str) -> List[str]:
        """Split text into chunks with overlap.
        
        Simple paragraph-based splitting with size constraints.
        """
        # Split by paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        if not paragraphs:
            return []
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for para in paragraphs:
            para_length = len(para.split())  # Approximate token count
            
            if current_length + para_length > self.chunk_size:
                # Save current chunk and start new one
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [para]
                current_length = para_length
            else:
                current_chunk.append(para)
                current_length += para_length
        
        # Don't forget the last chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
