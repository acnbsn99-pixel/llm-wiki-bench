"""LLM Client module for llm-vs-rag-bench.

This module provides an LLM client that is COMPATIBLE with how the original 
llm-wiki-agent calls the LLM. Based on the analysis in docs/wiki_agent_analysis.md:

- The original agent uses `litellm.completion()` 
- It passes model, messages (with role/content), and max_tokens
- Response format: response.choices[0].message.content

This client wraps litellm to add:
- Token usage tracking (prompt_tokens, completion_tokens) cumulatively
- Latency tracking per call
- Error handling with retries (max 3, exponential backoff)
"""

import time
from dataclasses import dataclass, field
from typing import Any, Optional

import litellm
from litellm import completion as litellm_completion

from .config import Config, get_config


@dataclass
class TokenUsage:
    """Tracks token usage for a single LLM call."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class CallResult:
    """Result of an LLM call including response, usage, and latency."""
    content: str
    usage: TokenUsage
    latency_ms: float
    model: str
    raw_response: Any = None


@dataclass
class CumulativeStats:
    """Cumulative statistics across all LLM calls."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    total_latency_ms: float = 0.0
    
    def average_latency_ms(self) -> float:
        """Calculate average latency per call."""
        if self.total_calls == 0:
            return 0.0
        return self.total_latency_ms / self.total_calls


class LLMClient:
    """LLM client compatible with llm-wiki-agent call patterns.
    
    The original wiki-agent uses litellm.completion() with this pattern:
        response = completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    
    This client wraps that pattern and adds tracking/metrics.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize the LLM client.
        
        Args:
            config: Config instance. If None, uses global config.
        """
        self.config = config or get_config()
        self.stats = CumulativeStats()
        
        # Configure litellm to use our custom base URL
        # litellm supports custom API bases via api_base parameter
        self.api_base = self.config.OPENAI_BASE_URL
        self.api_key = self.config.OPENAI_API_KEY
        self.default_model = self.config.OPENAI_MODEL
        self.default_model_fast = self.config.OPENAI_MODEL_FAST
        
        # Set litellm to not send telemetry
        litellm.set_verbose = False
        litellm.suppress_debug_info = True
    
    def _make_request(
        self,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float = 0.0,
        **kwargs
    ) -> Any:
        """Make the actual litellm completion request.
        
        Args:
            model: Model name to use
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            temperature: Temperature for sampling
            **kwargs: Additional arguments passed to litellm.completion
            
        Returns:
            Raw litellm response object
        """
        # Use openai/ prefix to force OpenAI-compatible API call
        # This prevents litellm from trying to use provider-specific SDKs
        # (e.g., gemini-* models would trigger Vertex AI without this prefix)
        return litellm_completion(
            model=f"openai/{model}",
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            api_base=self.api_base,
            api_key=self.api_key,
            **kwargs
        )
    
    def call(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> CallResult:
        """Call the LLM with a prompt.
        
        This is the main entry point, compatible with the wiki-agent's call_llm pattern.
        
        Args:
            prompt: The user prompt/message content
            system_message: Optional system message to prepend
            max_tokens: Max tokens to generate (uses config default if not specified)
            temperature: Temperature for sampling (uses config default if not specified)
            model: Model to use (uses config default if not specified)
            **kwargs: Additional arguments passed to litellm.completion
            
        Returns:
            CallResult with content, usage, latency, and model info
            
        Raises:
            Exception: After exhausting retries
        """
        if max_tokens is None:
            max_tokens = self.config.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.config.DEFAULT_TEMPERATURE
        if model is None:
            model = self.default_model
        
        # Build messages list
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        
        # Retry with exponential backoff
        last_error = None
        for attempt in range(self.config.MAX_RETRIES):
            start_time = time.perf_counter()
            try:
                response = self._make_request(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs
                )
                
                # Calculate latency
                end_time = time.perf_counter()
                latency_ms = (end_time - start_time) * 1000
                
                # Extract content
                content = response.choices[0].message.content
                
                # Extract token usage
                usage_data = response.usage if hasattr(response, 'usage') else None
                if usage_data:
                    usage = TokenUsage(
                        prompt_tokens=getattr(usage_data, 'prompt_tokens', 0),
                        completion_tokens=getattr(usage_data, 'completion_tokens', 0),
                        total_tokens=getattr(usage_data, 'total_tokens', 0)
                    )
                else:
                    usage = TokenUsage()
                
                # Update cumulative stats
                self.stats.total_calls += 1
                self.stats.successful_calls += 1
                self.stats.total_prompt_tokens += usage.prompt_tokens
                self.stats.total_completion_tokens += usage.completion_tokens
                self.stats.total_tokens += usage.total_tokens
                self.stats.total_latency_ms += latency_ms
                
                return CallResult(
                    content=content,
                    usage=usage,
                    latency_ms=latency_ms,
                    model=model,
                    raw_response=response
                )
                
            except Exception as e:
                last_error = e
                self.stats.total_calls += 1
                self.stats.failed_calls += 1
                
                if attempt < self.config.MAX_RETRIES - 1:
                    # Exponential backoff
                    delay = self.config.RETRY_BASE_DELAY * (2 ** attempt)
                    time.sleep(delay)
                else:
                    # Exhausted retries
                    raise
        
        # Should not reach here, but just in case
        raise last_error
    
    def call_with_messages(
        self,
        messages: list[dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> CallResult:
        """Call the LLM with a pre-built messages list.
        
        Useful when you need more control over the message structure
        (e.g., multi-turn conversations).
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Max tokens to generate
            temperature: Temperature for sampling
            model: Model to use
            **kwargs: Additional arguments passed to litellm.completion
            
        Returns:
            CallResult with content, usage, latency, and model info
        """
        if max_tokens is None:
            max_tokens = self.config.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.config.DEFAULT_TEMPERATURE
        if model is None:
            model = self.default_model
        
        # Retry with exponential backoff
        last_error = None
        for attempt in range(self.config.MAX_RETRIES):
            start_time = time.perf_counter()
            try:
                response = self._make_request(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs
                )
                
                # Calculate latency
                end_time = time.perf_counter()
                latency_ms = (end_time - start_time) * 1000
                
                # Extract content
                content = response.choices[0].message.content
                
                # Extract token usage
                usage_data = response.usage if hasattr(response, 'usage') else None
                if usage_data:
                    usage = TokenUsage(
                        prompt_tokens=getattr(usage_data, 'prompt_tokens', 0),
                        completion_tokens=getattr(usage_data, 'completion_tokens', 0),
                        total_tokens=getattr(usage_data, 'total_tokens', 0)
                    )
                else:
                    usage = TokenUsage()
                
                # Update cumulative stats
                self.stats.total_calls += 1
                self.stats.successful_calls += 1
                self.stats.total_prompt_tokens += usage.prompt_tokens
                self.stats.total_completion_tokens += usage.completion_tokens
                self.stats.total_tokens += usage.total_tokens
                self.stats.total_latency_ms += latency_ms
                
                return CallResult(
                    content=content,
                    usage=usage,
                    latency_ms=latency_ms,
                    model=model,
                    raw_response=response
                )
                
            except Exception as e:
                last_error = e
                self.stats.total_calls += 1
                self.stats.failed_calls += 1
                
                if attempt < self.config.MAX_RETRIES - 1:
                    delay = self.config.RETRY_BASE_DELAY * (2 ** attempt)
                    time.sleep(delay)
                else:
                    raise
        
        raise last_error
    
    def get_stats(self) -> CumulativeStats:
        """Get cumulative statistics across all LLM calls."""
        return self.stats
    
    def reset_stats(self) -> None:
        """Reset cumulative statistics."""
        self.stats = CumulativeStats()


# Convenience function matching the original wiki-agent pattern
def call_llm(prompt: str, max_tokens: int = 8192) -> str:
    """Simple LLM call function matching the original wiki-agent signature.
    
    This is a convenience wrapper for quick adoption of existing code.
    For full features (token tracking, latency, etc.), use LLMClient directly.
    
    Args:
        prompt: The user prompt
        max_tokens: Maximum tokens to generate
        
    Returns:
        The LLM response content as a string
    """
    client = LLMClient()
    result = client.call(prompt=prompt, max_tokens=max_tokens)
    return result.content


# Module-level client instance
_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """Get the module-level LLM client instance."""
    global _client
    if _client is None:
        _client = LLMClient()
    return _client
