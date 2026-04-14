#!/usr/bin/env python3
"""CLI entrypoint for llm-vs-rag-bench.

This module provides the main CLI interface for running benchmarks between
the LLM-Wiki-Agent and traditional RAG pipelines.

Usage:
    python main.py --n-docs 5 --m-questions 3
    python main.py --dry-run
    python main.py --n-docs 10 --m-questions 5 --verbose
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import typer

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.config import Config, get_config, ConfigError
from src.llm_client import LLMClient
from src.data.dataset_loader import load_documents_and_questions
from src.data.models import Document, Question, BenchmarkResult, Trajectory
from src.llm_wiki import WikiIngestor, WikiQuerier, TrajectoryLogger
from src.rag import RAGPipeline, Chunker, FAISSVectorStore
from src.evaluation import LLMJudge, MetricsCalculator, ReportGenerator
from src.trajectory import TrajectoryExporter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_app() -> typer.Typer:
    """Create the Typer CLI application."""
    app = typer.Typer(
        name="llm-vs-rag-bench",
        help="Benchmark LLM-Wiki-Agent vs traditional RAG pipeline",
        add_completion=False
    )
    return app


app = create_app()


@app.command()
def benchmark(
    n_docs: int = typer.Option(
        ..., 
        "--n-docs", "-n",
        help="Number of documents to load from the dataset"
    ),
    m_questions: int = typer.Option(
        ...,
        "--m-questions", "-m",
        help="Number of questions to benchmark"
    ),
    domain: str = typer.Option(
        "healthcare",
        "--domain", "-d",
        help="Dataset domain to use (default: healthcare)"
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Run with 1 document and 1 question for testing"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Print detailed agent traces and debugging info"
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir", "-o",
        help="Directory for results and trajectories (default: project root/results and /trajectories)"
    )
):
    """Run the full benchmark workflow.
    
    This command executes the complete benchmark pipeline:
    1. Load N documents and M questions from UniDoc-Bench dataset
    2. Ingest documents into both LLM-Wiki-Agent and RAG pipelines
    3. Query both pipelines with each question
    4. Evaluate answers using LLM-as-Judge
    5. Generate comparative report
    6. Export agent trajectories for SFT
    
    Examples:
        python main.py -n 5 -m 3
        python main.py -n 10 -m 5 --verbose
        python main.py --dry-run
    """
    # Handle dry-run mode
    if dry_run:
        n_docs = 1
        m_questions = 1
        logger.info("Dry-run mode: using 1 document and 1 question")
    
    # Set logging level based on verbose flag
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose mode enabled")
    
    try:
        # Initialize configuration
        logger.info("Loading configuration...")
        config = get_config()
        
        # Initialize LLM client
        logger.info("Initializing LLM client...")
        llm_client = LLMClient(config=config)
        
        # Step 1: Load dataset
        logger.info(f"Loading {n_docs} documents and {m_questions} questions from {domain} domain...")
        documents, questions = load_documents_and_questions(
            n_docs=n_docs,
            m_questions=m_questions,
            domain=domain
        )
        logger.info(f"Loaded {len(documents)} documents and {len(questions)} questions")
        
        if not documents or not questions:
            logger.error("No documents or questions loaded. Check dataset availability.")
            raise ValueError("Failed to load dataset")
        
        # Trim questions if we loaded fewer than requested
        if len(questions) < m_questions:
            logger.warning(f"Only {len(questions)} questions available (requested {m_questions})")
            m_questions = len(questions)
        
        # Limit to available documents
        if len(documents) < n_docs:
            logger.warning(f"Only {len(documents)} documents available (requested {n_docs})")
        
        # Step 2: Initialize pipelines
        logger.info("Initializing LLM-Wiki-Agent pipeline...")
        wiki_ingestor = WikiIngestor(client=llm_client)
        wiki_querier = WikiQuerier(client=llm_client)
        
        logger.info("Initializing RAG pipeline...")
        rag_pipeline = RAGPipeline(
            llm_client=llm_client,
            k=5,  # Retrieve top 5 chunks
            api_base=config.OPENAI_BASE_URL,
            api_key=config.OPENAI_API_KEY
        )
        
        # Step 3: Ingest documents into both pipelines
        logger.info("Ingesting documents into LLM-Wiki-Agent...")
        wiki_results = []
        wiki_trajectories = []
        
        for i, doc in enumerate(documents):
            logger.info(f"  Ingesting document {i+1}/{len(documents)}: {doc.doc_id}")
            try:
                result_str, metadata = wiki_ingestor.ingest_from_document_dataclass(
                    document=doc,
                    question_id=f"ingest_{doc.doc_id}"
                )
                logger.debug(f"    Result: {result_str}")
            except Exception as e:
                logger.error(f"    Failed to ingest {doc.doc_id}: {e}")
                if verbose:
                    logger.exception("Detailed error:")
                continue
        
        logger.info("Ingesting documents into RAG pipeline...")
        total_chunks = rag_pipeline.ingest_documents(documents)
        logger.info(f"  Created {total_chunks} chunks from {len(documents)} documents")
        
        # Step 4: Query both pipelines
        logger.info(f"Querying both pipelines with {m_questions} questions...")
        rag_results = []
        
        for i, question in enumerate(questions[:m_questions]):
            logger.info(f"\nQuestion {i+1}/{m_questions}: {question.text[:80]}...")
            
            # Query LLM-Wiki-Agent
            logger.info("  Querying LLM-Wiki-Agent...")
            try:
                wiki_result = wiki_querier.query_from_question_dataclass(
                    question=question,
                    save_path=None  # Don't save individual syntheses
                )
                wiki_results.append(wiki_result)
                logger.info(f"    Answer length: {len(wiki_result.predicted_answer)} chars")
                logger.info(f"    Latency: {wiki_result.latency_seconds:.2f}s")
                logger.info(f"    Tokens: {wiki_result.token_usage}")
                
                if verbose and wiki_result.trajectory:
                    logger.debug("    Trajectory messages:")
                    for msg in wiki_result.trajectory.get("messages", [])[:5]:
                        role = msg.get("role", "unknown")
                        content = msg.get("content", "")[:100]
                        logger.debug(f"      [{role}]: {content}...")
                
            except Exception as e:
                logger.error(f"    LLM-Wiki-Agent failed: {e}")
                if verbose:
                    logger.exception("Detailed error:")
                # Create a failed result placeholder
                wiki_results.append(BenchmarkResult(
                    pipeline_name="llm-wiki-agent",
                    question_id=question.question_id,
                    predicted_answer="",
                    latency_seconds=0.0,
                    token_usage=0,
                    retrieval_count=0,
                    trajectory={"error": str(e)}
                ))
            
            # Query RAG pipeline
            logger.info("  Querying RAG pipeline...")
            try:
                rag_result = rag_pipeline.query(question=question)
                rag_results.append(rag_result)
                logger.info(f"    Answer length: {len(rag_result.predicted_answer)} chars")
                logger.info(f"    Latency: {rag_result.latency_seconds:.2f}s")
                logger.info(f"    Tokens: {rag_result.token_usage}")
                logger.info(f"    Retrieved: {rag_result.retrieval_count} chunks")
                
                if verbose and rag_result.trajectory:
                    logger.debug("    Retrieved chunks:")
                    for chunk_info in rag_result.trajectory.get("retrieved_chunks", [])[:3]:
                        logger.debug(f"      Page {chunk_info.get('page_number')}: score={chunk_info.get('score', 0):.3f}")
                
            except Exception as e:
                logger.error(f"    RAG pipeline failed: {e}")
                if verbose:
                    logger.exception("Detailed error:")
                # Create a failed result placeholder
                rag_results.append(BenchmarkResult(
                    pipeline_name="rag",
                    question_id=question.question_id,
                    predicted_answer="",
                    latency_seconds=0.0,
                    token_usage=0,
                    retrieval_count=0,
                    trajectory={"error": str(e)}
                ))
        
        # Step 5: Evaluate results with LLM-as-Judge
        logger.info("\nEvaluating results with LLM-as-Judge...")
        judge = LLMJudge(llm_client=llm_client)
        
        for i, question in enumerate(questions[:m_questions]):
            logger.info(f"  Evaluating question {i+1}/{m_questions}...")
            
            # Evaluate LLM-Wiki-Agent answer
            if i < len(wiki_results) and wiki_results[i].predicted_answer:
                try:
                    wiki_judge_result = judge.evaluate(
                        question=question.text,
                        predicted_answer=wiki_results[i].predicted_answer,
                        ground_truth=question.ground_truth_answer,
                        question_id=question.question_id
                    )
                    wiki_results[i].score = wiki_judge_result.score
                    logger.debug(f"    LLM-Wiki-Agent score: {wiki_judge_result.score}")
                except Exception as e:
                    logger.error(f"    Failed to evaluate LLM-Wiki-Agent: {e}")
            
            # Evaluate RAG answer
            if i < len(rag_results) and rag_results[i].predicted_answer:
                try:
                    rag_judge_result = judge.evaluate(
                        question=question.text,
                        predicted_answer=rag_results[i].predicted_answer,
                        ground_truth=question.ground_truth_answer,
                        question_id=question.question_id
                    )
                    rag_results[i].score = rag_judge_result.score
                    logger.debug(f"    RAG score: {rag_judge_result.score}")
                except Exception as e:
                    logger.error(f"    Failed to evaluate RAG: {e}")
        
        # Step 6: Calculate metrics and generate report
        logger.info("\nCalculating metrics...")
        metrics_calculator = MetricsCalculator()
        llm_wiki_metrics, rag_metrics = metrics_calculator.calculate_all_metrics(
            llm_wiki_results=wiki_results,
            rag_results=rag_results
        )
        
        logger.info("Generating report...")
        report_generator = ReportGenerator(results_dir=config.PROJECT_ROOT / "results")
        comparison_data, csv_path, console_output = report_generator.generate_full_report(
            llm_wiki_metrics=llm_wiki_metrics,
            rag_metrics=rag_metrics,
            filename="benchmark_report.csv",
            title="LLM-vs-RAG Benchmark Comparison"
        )
        
        logger.info(f"Report saved to: {csv_path}")
        
        # Step 7: Export trajectories
        logger.info("\nExporting agent trajectories...")
        trajectory_exporter = TrajectoryExporter(
            output_dir=config.PROJECT_ROOT / "trajectories"
        )
        
        # Extract trajectories from LLM-Wiki-Agent results
        trajectories = []
        for result in wiki_results:
            if result.trajectory and result.trajectory.get("messages"):
                traj = Trajectory(
                    question_id=result.question_id,
                    messages=result.trajectory.get("messages", []),
                    metadata=result.trajectory.get("metadata", {})
                )
                trajectories.append(traj)
        
        if trajectories:
            jsonl_path = trajectory_exporter.export_to_jsonl(
                trajectories=trajectories,
                output_filename="agent_trajectories.jsonl"
            )
            logger.info(f"Exported {len(trajectories)} trajectories to: {jsonl_path}")
        else:
            logger.warning("No trajectories to export")
            # Create empty file to indicate completion
            jsonl_path = config.PROJECT_ROOT / "trajectories" / "agent_trajectories.jsonl"
            jsonl_path.touch()
        
        # Print summary
        print("\n" + "=" * 70)
        print("BENCHMARK COMPLETE")
        print("=" * 70)
        print(f"\nDocuments processed: {len(documents)}")
        print(f"Questions benchmarked: {min(m_questions, len(questions))}")
        print(f"\nResults saved to: {csv_path}")
        print(f"Trajectories saved to: {jsonl_path}")
        print("\n" + "=" * 70)
        
        return {
            "success": True,
            "documents_processed": len(documents),
            "questions_benchmarked": min(m_questions, len(questions)),
            "report_path": str(csv_path),
            "trajectories_path": str(jsonl_path)
        }
        
    except ConfigError as e:
        logger.error(f"Configuration error: {e}")
        logger.error("Please ensure .env file exists with required variables:")
        logger.error("  OPENAI_BASE_URL=http://az.gptplus5.com/v1")
        logger.error("  OPENAI_API_KEY=<your-api-key>")
        logger.error("  OPENAI_MODEL=gemini-3-flash-preview")
        raise typer.Exit(code=1)
    
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        if verbose:
            logger.exception("Detailed error traceback:")
        raise typer.Exit(code=1)


@app.command()
def inspect_dataset(
    domain: str = typer.Option(
        "healthcare",
        "--domain", "-d",
        help="Dataset domain to inspect"
    ),
    num_samples: int = typer.Option(
        3,
        "--samples", "-s",
        help="Number of sample rows to display"
    )
):
    """Inspect the UniDoc-Bench dataset schema and samples.
    
    This command loads and displays information about the dataset,
    including schema, splits, and sample rows.
    """
    from src.data.dataset_loader import print_dataset_info, get_dataset_schema
    
    logger.info(f"Inspecting {domain} dataset...")
    
    schema = get_dataset_schema()
    print("\n" + "=" * 70)
    print("UniDoc-Bench Dataset Schema")
    print("=" * 70)
    print(f"Dataset: {schema['dataset_name']}")
    print(f"Splits: {', '.join(schema['splits'])}")
    print(f"Modality: {schema['modality']}")
    print(f"\nColumns:")
    for col, desc in schema['columns'].items():
        print(f"  {col}: {desc}")
    print("=" * 70)
    
    try:
        from datasets import load_dataset
        dataset = load_dataset(schema['dataset_name'], split=domain)
        print(f"\n{domain} split size: {len(dataset)} samples")
        
        if len(dataset) > 0:
            print(f"\nShowing {min(num_samples, len(dataset))} sample rows:\n")
            for i in range(min(num_samples, len(dataset))):
                row = dataset[i]
                print(f"--- Sample {i+1} ---")
                print(f"Question: {row['question'][:200]}...")
                print(f"Answer: {row['answer'][:200]}...")
                print(f"Domain: {row['domain']}")
                print(f"Question Type: {row['question_type']}")
                print(f"Answer Type: {row['answer_type']}")
                print(f"Pages: {len(row['longdoc_image_paths'])}")
                print()
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise typer.Exit(code=1)


@app.command()
def test_llm(
    prompt: str = typer.Option(
        "Say hello!",
        "--prompt", "-p",
        help="Test prompt to send to the LLM"
    )
):
    """Test the LLM client connection.
    
    This command verifies that the LLM API is accessible and working.
    """
    try:
        config = get_config()
        llm_client = LLMClient(config=config)
        
        logger.info(f"Testing LLM connection with model: {config.OPENAI_MODEL}")
        logger.info(f"API Base URL: {config.OPENAI_BASE_URL}")
        
        result = llm_client.call(prompt=prompt, max_tokens=100)
        
        print("\n" + "=" * 70)
        print("LLM TEST SUCCESSFUL")
        print("=" * 70)
        print(f"\nPrompt: {prompt}")
        print(f"Response: {result.content}")
        print(f"\nTokens used: {result.usage.total_tokens}")
        print(f"Latency: {result.latency_ms:.2f}ms")
        print("=" * 70)
        
    except ConfigError as e:
        logger.error(f"Configuration error: {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        logger.error(f"LLM test failed: {e}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
