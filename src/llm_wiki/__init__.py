"""LLM Wiki Agent adaptation for llm-vs-rag-bench.

This module adapts the original llm-wiki-agent from:
https://github.com/SamurAIGPT/llm-wiki-agent

The original architecture is preserved faithfully. Changes made:
- LLM calling uses src/llm_client.py instead of direct litellm calls
- Document ingestion works with src/data/models.py Document dataclass
- Query interface accepts Question dataclass and returns BenchmarkResult/Trajectory
- Added tracking for token usage, latency, retrieval count, and full trajectory logging

Components:
- WikiIngestor: Ingest documents into the wiki (from tools/ingest.py)
- WikiQuerier: Query the wiki for answers (from tools/query.py)
- WikiGraphBuilder: Build knowledge graph (from tools/build_graph.py)
- WikiLinter: Lint the wiki for health issues (from tools/lint.py)
- TrajectoryLogger: Track agent trajectories for SFT export
"""

from .ingest import WikiIngestor
from .query import WikiQuerier
from .graph import WikiGraphBuilder
from .lint import WikiLinter
from .tracking import TrajectoryLogger

__all__ = [
    "WikiIngestor",
    "WikiQuerier",
    "WikiGraphBuilder",
    "WikiLinter",
    "TrajectoryLogger",
]