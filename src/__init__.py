"""
llm-vs-rag-bench: Benchmarking agentic retrieval vs traditional RAG.

This package provides:
- Configuration management (src.config)
- LLM client compatible with llm-wiki-agent patterns (src.llm_client)
- Data loading and preprocessing (src.data)
- Retrieval implementations (src.retrieval)
- Generation modules (src.generation)
- Evaluation harness (src.evaluation)
"""

from .config import Config, ConfigError, get_config, reset_config
from .llm_client import LLMClient, LLMCallResult, TokenUsage

__all__ = [
    "Config",
    "ConfigError",
    "get_config",
    "reset_config",
    "LLMClient",
    "LLMCallResult",
    "TokenUsage",
]