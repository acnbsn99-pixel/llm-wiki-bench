"""
Configuration module for llm-vs-rag-bench.

Loads environment variables and validates required configuration values.
Based on the llm-wiki-agent analysis, we need:
- LLM_MODEL: Primary model for complex tasks
- LLM_MODEL_FAST: Fast model for quick tasks
- OPENAI_BASE_URL: Base URL for OpenAI-compatible API
- OPENAI_API_KEY: API key for authentication
- OPENAI_MODEL: Model name for OpenAI client
- EMBEDDING_MODEL_NAME: Model name for embeddings
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


class ConfigError(Exception):
    """Raised when configuration validation fails."""
    pass


class Config:
    """
    Configuration manager that loads and validates environment variables.
    
    Attributes:
        openai_base_url: Base URL for OpenAI-compatible API endpoint
        openai_api_key: API key for authentication
        openai_model: Default model for chat completions
        embedding_model_name: Model name for embeddings
        llm_model: Primary LLM model (compatible with llm-wiki-agent)
        llm_model_fast: Fast LLM model for quick tasks (compatible with llm-wiki-agent)
    """
    
    def __init__(self, env_path: Optional[Path] = None):
        """
        Initialize configuration by loading environment variables.
        
        Args:
            env_path: Optional path to .env file. If None, uses default dotenv locations.
        """
        # Load environment variables from .env file if it exists
        if env_path is not None:
            load_dotenv(env_path)
        else:
            # Try to load from project root
            project_root = Path(__file__).parent.parent
            env_file = project_root / ".env"
            if env_file.exists():
                load_dotenv(env_file)
        
        # Load all configuration values
        self.openai_base_url = self._get_required("OPENAI_BASE_URL")
        self.openai_api_key = self._get_required("OPENAI_API_KEY")
        self.openai_model = self._get_required("OPENAI_MODEL")
        self.embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-small")
        
        # LLM Wiki Agent compatible settings
        self.llm_model = os.getenv("LLM_MODEL", self.openai_model)
        self.llm_model_fast = os.getenv("LLM_MODEL_FAST", self.llm_model)
    
    def _get_required(self, name: str) -> str:
        """
        Get a required environment variable.
        
        Args:
            name: Environment variable name
            
        Returns:
            The value of the environment variable
            
        Raises:
            ConfigError: If the environment variable is not set
        """
        value = os.getenv(name)
        if value is None or value.strip() == "":
            raise ConfigError(
                f"Required environment variable '{name}' is not set. "
                f"Please set it in your .env file or export it. "
                f"See .env.example for reference."
            )
        return value.strip()
    
    def get_llm_model(self, use_fast: bool = False) -> str:
        """
        Get the appropriate LLM model based on task complexity.
        
        Args:
            use_fast: If True, return the fast model; otherwise return the primary model
            
        Returns:
            Model name string
        """
        return self.llm_model_fast if use_fast else self.llm_model
    
    @property
    def api_headers(self) -> dict:
        """Get headers for API requests."""
        return {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json"
        }


# Global configuration instance
_config: Optional[Config] = None


def get_config(env_path: Optional[Path] = None) -> Config:
    """
    Get the global configuration instance.
    
    Args:
        env_path: Optional path to .env file
        
    Returns:
        Config instance
    """
    global _config
    if _config is None:
        _config = Config(env_path)
    return _config


def reset_config():
    """Reset the global configuration (useful for testing)."""
    global _config
    _config = None
