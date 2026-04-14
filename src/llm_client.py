"""
LLM Client module for llm-vs-rag-bench.

This client is designed to be COMPATIBLE with how the original llm-wiki-agent
calls the LLM. Based on the analysis in docs/wiki_agent_analysis.md:

The original agent uses:
- litellm.completion() as the LLM client
- Single-shot prompts (not iterative tool use)
- Declarative JSON output format for structured tasks
- Two-model setup: LLM_MODEL for complex tasks, LLM_MODEL_FAST for quick tasks
- Basic parameters: model, messages (with role/content), max_tokens

This implementation:
- Uses litellm.completion() to match the original pattern exactly
- Supports OpenAI-compatible endpoints via base_url configuration
- Tracks token usage (prompt_tokens, completion_tokens) cumulatively
- Tracks latency per call
- Implements retry logic with exponential backoff (max 3 retries)
"""

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from litellm import completion
from litellm.exceptions import APIError, RateLimitError, Timeout

from .config import Config, get_config


@dataclass
class LLMCallResult:
    """
    Result of an LLM call with metadata.
    
    Attributes:
        content: The response content from the LLM
        prompt_tokens: Number of tokens in the prompt
        completion_tokens: Number of tokens in the completion
        total_tokens: Total tokens used
        latency_ms: Latency of the call in milliseconds
        model: Model that was used for the call
    """
    content: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    model: str


@dataclass
class TokenUsage:
    """
    Cumulative token usage tracker.
    
    Attributes:
        total_prompt_tokens: Total prompt tokens across all calls
        total_completion_tokens: Total completion tokens across all calls
        total_tokens: Total tokens across all calls
        call_count: Number of LLM calls made
    """
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    call_count: int = 0
    
    def add(self, prompt: int, completion: int, total: int):
        """Add token counts from a single call."""
        self.total_prompt_tokens += prompt
        self.total_completion_tokens += completion
        self.total_tokens += total
        self.call_count += 1
    
    def __str__(self) -> str:
        return (
            f"TokenUsage(calls={self.call_count}, "
            f"prompt={self.total_prompt_tokens}, "
            f"completion={self.total_completion_tokens}, "
            f"total={self.total_tokens})"
        )


class LLMClient:
    """
    LLM client compatible with llm-wiki-agent calling patterns.
    
    This client wraps litellm.completion() to provide:
    - Configuration management
    - Token usage tracking
    - Latency tracking
    - Retry logic with exponential backoff
    
    Usage:
        client = LLMClient()
        result = client.call("Your prompt here")
        print(result.content)
        print(client.usage)  # Cumulative token usage
    """
    
    def __init__(
        self,
        config: Optional[Config] = None,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
    ):
        """
        Initialize the LLM client.
        
        Args:
            config: Configuration instance. If None, uses global config.
            max_retries: Maximum number of retry attempts (default: 3)
            base_delay: Base delay for exponential backoff in seconds (default: 1.0)
            max_delay: Maximum delay between retries in seconds (default: 60.0)
        """
        self.config = config if config is not None else get_config()
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        
        # Token usage tracker
        self.usage = TokenUsage()
        
        # Latency history (for debugging/analysis)
        self.latency_history: list[float] = []
    
    def _calculate_backoff(self, attempt: int) -> float:
        """
        Calculate backoff delay for retry attempt.
        
        Args:
            attempt: Current retry attempt number (0-indexed)
            
        Returns:
            Delay in seconds
        """
        import random
        # Exponential backoff with jitter
        delay = min(self.base_delay * (2 ** attempt), self.max_delay)
        jitter = random.uniform(0, delay * 0.1)  # 10% jitter
        return delay + jitter
    
    def _should_retry(self, error: Exception) -> bool:
        """
        Determine if an error is retryable.
        
        Args:
            error: The exception that was raised
            
        Returns:
            True if the error should trigger a retry
        """
        # litellm exceptions that are retryable
        retryable_types = (
            RateLimitError,
            Timeout,
            APIError,
        )
        return isinstance(error, retryable_types)
    
    def call(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        model: Optional[str] = None,
        use_fast: bool = False,
        **kwargs: Any,
    ) -> LLMCallResult:
        """
        Make an LLM call with retry logic.
        
        This method matches the calling pattern from llm-wiki-agent:
        - Single user message (or system + user messages)
        - max_tokens parameter
        - Returns response.choices[0].message.content
        
        Args:
            prompt: The main prompt text
            system_prompt: Optional system prompt to prepend
            max_tokens: Maximum tokens to generate (default: 4096)
            temperature: Temperature for sampling (default: 0.7)
            model: Optional model override. If None, uses config.llm_model
            use_fast: If True, use the fast model instead of primary model
            **kwargs: Additional arguments passed to litellm.completion()
            
        Returns:
            LLMCallResult with content and metadata
            
        Raises:
            Exception: If all retry attempts fail
        """
        # Determine which model to use
        if model is None:
            if use_fast:
                model = self.config.llm_model_fast
            else:
                model = self.config.llm_model
        
        # Build messages list
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        last_error: Optional[Exception] = None
        
        for attempt in range(self.max_retries + 1):
            start_time = time.time()
            
            try:
                # Call litellm.completion() - matches llm-wiki-agent pattern exactly
                response = completion(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    api_base=self.config.openai_base_url,
                    api_key=self.config.openai_api_key,
                    **kwargs,
                )
                
                # Extract content - matches llm-wiki-agent pattern
                content = response.choices[0].message.content
                
                # Extract token usage
                usage_data = response.usage
                prompt_tokens = getattr(usage_data, "prompt_tokens", 0)
                completion_tokens = getattr(usage_data, "completion_tokens", 0)
                total_tokens = getattr(usage_data, "total_tokens", 0)
                
                # Calculate latency
                latency_ms = (time.time() - start_time) * 1000
                self.latency_history.append(latency_ms)
                
                # Update cumulative token usage
                self.usage.add(prompt_tokens, completion_tokens, total_tokens)
                
                return LLMCallResult(
                    content=content or "",
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    latency_ms=latency_ms,
                    model=model,
                )
                
            except Exception as e:
                last_error = e
                latency_ms = (time.time() - start_time) * 1000
                
                if attempt < self.max_retries and self._should_retry(e):
                    delay = self._calculate_backoff(attempt)
                    print(
                        f"LLM call failed (attempt {attempt + 1}/{self.max_retries + 1}): "
                        f"{type(e).__name__}: {e}. Retrying in {delay:.2f}s..."
                    )
                    time.sleep(delay)
                else:
                    # Either max retries exceeded or non-retryable error
                    break
        
        # All retries exhausted
        raise last_error  # type: ignore
    
    def call_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> dict:
        """
        Make an LLM call expecting JSON response.
        
        This is useful for structured outputs like the ingest.py pattern
        where the LLM returns declarative JSON specifying actions.
        
        Args:
            prompt: The main prompt text
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate
            model: Optional model override
            **kwargs: Additional arguments for call()
            
        Returns:
            Parsed JSON as dictionary
            
        Raises:
            json.JSONDecodeError: If response is not valid JSON
        """
        import json
        import re
        
        # Use lower temperature for more deterministic JSON output
        result = self.call(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=0.1,  # Lower temp for JSON
            model=model,
            **kwargs,
        )
        
        content = result.content.strip()
        
        # Strip markdown code fences if present
        if content.startswith("```"):
            # Remove opening fence
            content = re.sub(r"^```\w*\n", "", content)
            # Remove closing fence
            content = re.sub(r"\n```$", "", content)
            content = content.strip()
        
        # Parse JSON
        return json.loads(content)
    
    def get_average_latency(self) -> float:
        """
        Get average latency across all calls.
        
        Returns:
            Average latency in milliseconds, or 0.0 if no calls made
        """
        if not self.latency_history:
            return 0.0
        return sum(self.latency_history) / len(self.latency_history)
    
    def reset_usage(self):
        """Reset token usage and latency history."""
        self.usage = TokenUsage()
        self.latency_history = []
