"""Configuration module for llm-vs-rag-bench.

Loads environment variables and exposes all config values needed for LLM operations.
Validates that required values are present.
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


class ConfigError(Exception):
    """Raised when configuration is invalid or missing required values."""
    pass


class Config:
    """Configuration container for LLM and benchmark settings."""
    
    # LLM API Configuration
    OPENAI_BASE_URL: str
    OPENAI_API_KEY: str
    OPENAI_MODEL: str
    
    # Secondary model for fast operations (optional)
    OPENAI_MODEL_FAST: Optional[str] = None
    
    # LLM Parameters
    DEFAULT_MAX_TOKENS: int = 8192
    DEFAULT_TEMPERATURE: float = 0.0
    
    # Retry Configuration
    MAX_RETRIES: int = 3
    RETRY_BASE_DELAY: float = 1.0  # seconds
    
    # Paths
    PROJECT_ROOT: Path
    DATA_DIR: Path
    WIKI_DIR: Path
    GRAPH_DIR: Path
    
    _initialized = False
    
    def __init__(self):
        if Config._initialized:
            return
        
        # Set project paths first (before loading env)
        self.PROJECT_ROOT = Path(__file__).parent.parent
        self.DATA_DIR = self.PROJECT_ROOT / "data"
        self.WIKI_DIR = self.PROJECT_ROOT / "wiki"
        self.GRAPH_DIR = self.PROJECT_ROOT / "graph"
        
        # Load environment variables from .env file
        self._load_env()
        
        # Validate and set required values
        self._validate_and_set_config()
        
        Config._initialized = True
    
    def _load_env(self) -> None:
        """Load environment variables from .env file."""
        env_path = self.PROJECT_ROOT / ".env"
        load_dotenv(dotenv_path=env_path)
    
    def _validate_and_set_config(self) -> None:
        """Validate required configuration and set attributes."""
        errors = []
        
        # Required: OPENAI_BASE_URL
        self.OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")
        if not self.OPENAI_BASE_URL:
            errors.append("OPENAI_BASE_URL is required")
        
        # Required: OPENAI_API_KEY
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
        if not self.OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY is required")
        
        # Required: OPENAI_MODEL
        self.OPENAI_MODEL = os.getenv("OPENAI_MODEL", "")
        if not self.OPENAI_MODEL:
            errors.append("OPENAI_MODEL is required")
        
        # Optional: OPENAI_MODEL_FAST (defaults to same as OPENAI_MODEL)
        self.OPENAI_MODEL_FAST = os.getenv("OPENAI_MODEL_FAST") or self.OPENAI_MODEL
        
        # Optional: MAX_TOKENS override
        max_tokens_str = os.getenv("MAX_TOKENS")
        if max_tokens_str:
            try:
                self.DEFAULT_MAX_TOKENS = int(max_tokens_str)
            except ValueError:
                errors.append(f"MAX_TOKENS must be an integer, got: {max_tokens_str}")
        
        # Optional: TEMPERATURE override
        temp_str = os.getenv("TEMPERATURE")
        if temp_str:
            try:
                self.DEFAULT_TEMPERATURE = float(temp_str)
            except ValueError:
                errors.append(f"TEMPERATURE must be a float, got: {temp_str}")
        
        # Optional: MAX_RETRIES override
        retries_str = os.getenv("MAX_RETRIES")
        if retries_str:
            try:
                self.MAX_RETRIES = int(retries_str)
            except ValueError:
                errors.append(f"MAX_RETRIES must be an integer, got: {retries_str}")
        
        # Optional: RETRY_BASE_DELAY override
        delay_str = os.getenv("RETRY_BASE_DELAY")
        if delay_str:
            try:
                self.RETRY_BASE_DELAY = float(delay_str)
            except ValueError:
                errors.append(f"RETRY_BASE_DELAY must be a float, got: {delay_str}")
        
        if errors:
            raise ConfigError("\n".join(errors))
    
    @classmethod
    def get(cls) -> "Config":
        """Get the singleton Config instance."""
        if not cls._initialized:
            cls._instance = cls()
        return cls._instance
    
    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (useful for testing)."""
        cls._initialized = False
        if hasattr(cls, "_instance"):
            delattr(cls, "_instance")


# Singleton instance accessor
def get_config() -> Config:
    """Get the global Config instance."""
    return Config.get()


# Convenience access to common config values at module level
config = get_config()
