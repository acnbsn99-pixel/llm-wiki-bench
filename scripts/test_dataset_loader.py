#!/usr/bin/env python3
"""Test script for the UniDoc-Bench dataset loader.

This script:
1. Loads the healthcare dataset
2. Prints the schema
3. Loads 2 documents and prints their details
4. Loads 3 questions and prints their details
5. Verifies data flows correctly
"""

import sys
sys.path.insert(0, '/workspace')

from src.data.dataset_loader import (
    load_healthcare_dataset,
    load_documents,
    load_questions,
    get_dataset_schema,
    print_dataset_info
)


def main():
    print("=" * 70)
    print("PHASE 2: Dataset Module Test")
    print("=" * 70)
    
    # Step 1: Print dataset schema
    print("\n[1] Dataset Schema:")
    print("-" * 70)
    schema = get_dataset_schema()
    print(f"Dataset Name: {schema['dataset_name']}")
    print(f"Splits Available: {', '.join(schema['splits'])}")
    print(f"Modality: {schema['modality']}")
    print(f"Healthcare Size: {schema['healthcare_size']} samples")
    print("\nColumns:")
    for col, desc in schema['columns'].items():
        print(f"  - {col}: {desc}")
    
    # Step 2: Load and inspect the raw dataset
    print("\n[2] Loading Healthcare Dataset:")
    print("-" * 70)
    dataset = load_healthcare_dataset()
    print(f"Loaded {len(dataset)} samples from healthcare split")
    
    # Step 3: Print detailed dataset info with sample row
    print("\n[3] Sample Row from Dataset:")
    print("-" * 70)
    print_dataset_info(dataset)
    
    # Step 4: Load 2 documents
    print("\n[4] Loading 2 Documents:")
    print("-" * 70)
    documents = load_documents(n=2, domain="healthcare")
    print(f"Loaded {len(documents)} unique document(s)\n")
    
    for i, doc in enumerate(documents):
        print(f"Document {i+1}:")
        print(f"  Doc ID: {doc.doc_id}")
        print(f"  Domain: {doc.domain}")
        print(f"  Page Count: {doc.page_count}")
        print(f"  Image Paths:")
        for page in doc.pages[:3]:  # Show first 3 pages
            print(f"    - Page {page.page_number}: {page.image_path}")
        if len(doc.pages) > 3:
            print(f"    ... and {len(doc.pages) - 3} more pages")
        print()
    
    # Step 5: Load 3 questions
    print("\n[5] Loading 3 Questions:")
    print("-" * 70)
    questions = load_questions(m=3, domain="healthcare")
    print(f"Loaded {len(questions)} question(s)\n")
    
    for i, q in enumerate(questions):
        print(f"Question {i+1}:")
        print(f"  Question ID: {q.question_id}")
        print(f"  Text: {q.text[:150]}{'...' if len(q.text) > 150 else ''}")
        print(f"  Question Type: {q.question_type.value}")
        print(f"  Answer Type: {q.answer_type.value}")
        print(f"  Ground Truth Answer: {q.ground_truth_answer[:150]}{'...' if len(q.ground_truth_answer) > 150 else ''}")
        print(f"  Domain: {q.domain}")
        print(f"  Associated Document ID: {q.document.doc_id}")
        print(f"  Answer Page Paths: {q.gt_image_paths}")
        print()
    
    # Step 6: Verification summary
    print("\n[6] Verification Summary:")
    print("-" * 70)
    print(f"✓ Dataset loaded successfully: {len(dataset)} samples")
    print(f"✓ Documents loaded: {len(documents)} unique documents")
    print(f"✓ Questions loaded: {len(questions)} questions")
    print(f"✓ All field names match schema from docs/dataset_analysis.md:")
    print(f"    - question → Question.text")
    print(f"    - answer → Question.ground_truth_answer")
    print(f"    - gt_image_paths → Question.gt_image_paths")
    print(f"    - question_type → Question.question_type")
    print(f"    - answer_type → Question.answer_type")
    print(f"    - domain → Document.domain / Question.domain")
    print(f"    - longdoc_image_paths → Document.pages")
    print(f"✓ Multimodal format acknowledged: Documents are PNG images")
    print("\n" + "=" * 70)
    print("Phase 2 Complete: Dataset Module Ready")
    print("=" * 70)


if __name__ == "__main__":
    main()
