"""Dataset loader for UniDoc-Bench dataset.

This module loads and processes the UniDoc-Bench dataset from HuggingFace,
following the exact schema documented in docs/dataset_analysis.md.

Key findings from analysis:
- Dataset has 8 domain splits including 'healthcare' (200 rows)
- Documents are PNG images (multimodal dataset), not plain text
- Schema columns: question, answer, gt_image_paths, question_type, 
  answer_type, domain, longdoc_image_paths
"""

from typing import List, Optional, Tuple
from datasets import load_dataset, Dataset
from pathlib import Path

from .models import Document, DocumentPage, Question, QuestionType, AnswerType


# Base URL for UniDoc-Bench images on HuggingFace
# Images are stored at: https://huggingface.co/datasets/Salesforce/UniDoc-Bench/tree/main/images/
HF_DATASET_NAME = "Salesforce/UniDoc-Bench"
IMAGE_BASE_URL = "https://huggingface.co/datasets/Salesforce/UniDoc-Bench/resolve/main/"


def load_healthcare_dataset() -> Dataset:
    """Load the healthcare subset of UniDoc-Bench.
    
    Returns:
        Dataset: The healthcare split with 200 samples
    """
    dataset = load_dataset(HF_DATASET_NAME, split="healthcare")
    return dataset


def load_all_domains() -> dict:
    """Load all domain splits of UniDoc-Bench.
    
    Returns:
        dict: Dictionary mapping domain names to their datasets
    """
    return load_dataset(HF_DATASET_NAME)


def filter_by_domain(dataset: Dataset, domain: str) -> Dataset:
    """Filter a dataset by domain.
    
    Args:
        dataset: The dataset to filter
        domain: Domain name to filter by (e.g., 'healthcare', 'finance')
        
    Returns:
        Dataset: Filtered dataset containing only samples from the specified domain
    """
    return dataset.filter(lambda x: x["domain"] == domain)


def _extract_doc_id_from_path(image_path: str) -> str:
    """Extract document ID from an image path.
    
    Example: 'images/healthcare/1851094/1851094_page_0001.png' -> '1851094'
    
    Args:
        image_path: Path to the image file
        
    Returns:
        str: Document ID extracted from the path
    """
    # Path format: images/{domain}/{doc_id}/{doc_id}_page_{num}.png
    parts = image_path.split("/")
    if len(parts) >= 3:
        return parts[2]
    # Fallback: use the full path as ID
    return image_path.replace("/", "_").replace(".png", "")


def _parse_question_type(qtype_str: str) -> QuestionType:
    """Parse question type string to QuestionType enum.
    
    Args:
        qtype_str: String representation of question type
        
    Returns:
        QuestionType: Corresponding enum value
    """
    try:
        return QuestionType(qtype_str)
    except ValueError:
        # Default to factual_retrieval if unknown type
        return QuestionType.FACTUAL_RETRIEVAL


def _parse_answer_type(atype_str: str) -> AnswerType:
    """Parse answer type string to AnswerType enum.
    
    Args:
        atype_str: String representation of answer type
        
    Returns:
        AnswerType: Corresponding enum value
    """
    try:
        return AnswerType(atype_str)
    except ValueError:
        # Default to text_only if unknown type
        return AnswerType.TEXT_ONLY


def _row_to_document(row: dict) -> Document:
    """Convert a dataset row to a Document object.
    
    Args:
        row: A single row from the UniDoc-Bench dataset
        
    Returns:
        Document: Document object with pages extracted from longdoc_image_paths
    """
    doc_id = _extract_doc_id_from_path(row["longdoc_image_paths"][0])
    domain = row["domain"]
    
    pages = []
    for idx, image_path in enumerate(row["longdoc_image_paths"]):
        page = DocumentPage(
            image_path=image_path,
            page_number=idx + 1  # 1-indexed
        )
        pages.append(page)
    
    return Document(
        doc_id=doc_id,
        domain=domain,
        pages=pages
    )


def _row_to_question(row: dict, document: Document) -> Question:
    """Convert a dataset row to a Question object.
    
    Args:
        row: A single row from the UniDoc-Bench dataset
        document: The Document object this question refers to
        
    Returns:
        Question: Question object with all fields populated
    """
    # Generate a unique question ID from domain and doc_id
    question_id = f"{row['domain']}_{document.doc_id}_{hash(row['question']) % 10000}"
    
    return Question(
        question_id=question_id,
        text=row["question"],
        question_type=_parse_question_type(row["question_type"]),
        answer_type=_parse_answer_type(row["answer_type"]),
        ground_truth_answer=row["answer"],
        gt_image_paths=row["gt_image_paths"],
        domain=row["domain"],
        document=document
    )


def load_documents(n: int, domain: str = "healthcare") -> List[Document]:
    """Load N documents from the specified domain.
    
    Documents are deduplicated by doc_id, so if multiple questions reference
    the same document, it will only be returned once.
    
    Args:
        n: Number of unique documents to load
        domain: Domain to load from (default: 'healthcare')
        
    Returns:
        List[Document]: List of Document objects
    """
    if domain == "healthcare":
        dataset = load_healthcare_dataset()
    else:
        all_data = load_all_domains()
        dataset = all_data[domain] if domain in all_data else load_healthcare_dataset()
    
    seen_doc_ids = set()
    documents = []
    
    for row in dataset:
        doc_id = _extract_doc_id_from_path(row["longdoc_image_paths"][0])
        if doc_id not in seen_doc_ids:
            doc = _row_to_document(row)
            documents.append(doc)
            seen_doc_ids.add(doc_id)
            
            if len(documents) >= n:
                break
    
    return documents


def load_questions(m: int, domain: str = "healthcare") -> List[Question]:
    """Load M questions from the specified domain.
    
    Each question includes its associated document and ground truth answer.
    
    Args:
        m: Number of questions to load
        domain: Domain to load from (default: 'healthcare')
        
    Returns:
        List[Question]: List of Question objects with associated documents
    """
    if domain == "healthcare":
        dataset = load_healthcare_dataset()
    else:
        all_data = load_all_domains()
        dataset = all_data[domain] if domain in all_data else load_healthcare_dataset()
    
    questions = []
    
    for i, row in enumerate(dataset):
        if i >= m:
            break
            
        document = _row_to_document(row)
        question = _row_to_question(row, document)
        questions.append(question)
    
    return questions


def load_documents_and_questions(
    n_docs: int, 
    m_questions: int, 
    domain: str = "healthcare"
) -> Tuple[List[Document], List[Question]]:
    """Load both documents and questions from the specified domain.
    
    This is the main entry point for loading benchmark data.
    
    Args:
        n_docs: Number of unique documents to load
        m_questions: Number of questions to load
        domain: Domain to load from (default: 'healthcare')
        
    Returns:
        Tuple[List[Document], List[Question]]: Documents and questions
    """
    documents = load_documents(n_docs, domain)
    questions = load_questions(m_questions, domain)
    return documents, questions


def get_dataset_schema() -> dict:
    """Return the schema information for UniDoc-Bench dataset.
    
    Returns:
        dict: Schema description with column names and types
    """
    return {
        "dataset_name": HF_DATASET_NAME,
        "splits": ["commerce_manufacturing", "construction", "crm", "education", 
                   "energy", "finance", "healthcare", "legal"],
        "columns": {
            "question": "string - The query/question to answer",
            "answer": "string - Ground truth answer text",
            "gt_image_paths": "list[string] - Paths to ground truth image(s)",
            "question_type": "string - Category of question",
            "answer_type": "string - Required answer format",
            "domain": "string - Domain identifier",
            "longdoc_image_paths": "list[string] - Paths to document page images"
        },
        "modality": "multimodal (PNG images)",
        "healthcare_size": 200
    }


def print_dataset_info(dataset: Optional[Dataset] = None) -> None:
    """Print detailed information about the dataset.
    
    Args:
        dataset: Optional dataset to inspect. If None, loads healthcare split.
    """
    if dataset is None:
        dataset = load_healthcare_dataset()
    
    print("=" * 60)
    print("UniDoc-Bench Dataset Information")
    print("=" * 60)
    print(f"\nDataset: {HF_DATASET_NAME}")
    print(f"Split: healthcare")
    print(f"Total samples: {len(dataset)}")
    print(f"\nSchema:")
    print(dataset.features)
    print(f"\nSample row:")
    sample = dataset[0]
    for key, value in sample.items():
        if isinstance(value, list) and len(value) > 3:
            print(f"  {key}: [{value[0]}, {value[1]}, ... ({len(value)} items)]")
        else:
            print(f"  {key}: {value}")
    print("=" * 60)
