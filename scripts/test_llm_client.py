#!/usr/bin/env python3
"""Test script for the LLM client.

This script verifies that the LLM API endpoint works correctly by:
1. Importing the LLM client
2. Sending a test message
3. Printing the response, token counts, and latency
"""

import sys

# Add src to path for imports
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from src.llm_client import LLMClient, get_llm_client
from src.config import get_config, ConfigError


def main():
    """Run the LLM client test."""
    print("=" * 60)
    print("LLM Client Test")
    print("=" * 60)
    
    # First, verify config is loaded
    try:
        config = get_config()
        print(f"\nConfiguration loaded successfully:")
        print(f"  Base URL: {config.OPENAI_BASE_URL}")
        print(f"  Model: {config.OPENAI_MODEL}")
        print(f"  Max Tokens: {config.DEFAULT_MAX_TOKENS}")
        print(f"  Max Retries: {config.MAX_RETRIES}")
    except ConfigError as e:
        print(f"\nConfiguration Error: {e}")
        print("\nPlease ensure you have a .env file with required values:")
        print("  OPENAI_BASE_URL=http://az.gptplus5.com/v1")
        print("  OPENAI_API_KEY=<your-api-key>")
        print("  OPENAI_MODEL=gemini-3-flash-preview")
        return 1
    
    # Create LLM client
    print("\nInitializing LLM client...")
    client = get_llm_client()
    
    # Send test message
    test_prompt = "Hello! This is a test message to verify the LLM client is working correctly. Please respond with a brief confirmation."
    
    print(f"\nSending test message...")
    print(f"Prompt: {test_prompt}")
    print("-" * 60)
    
    try:
        result = client.call(
            prompt=test_prompt,
            max_tokens=100,
            temperature=0.7
        )
        
        print(f"\nResponse received!")
        print(f"\nContent:\n{result.content}")
        print(f"\nToken Usage:")
        print(f"  Prompt tokens:     {result.usage.prompt_tokens}")
        print(f"  Completion tokens: {result.usage.completion_tokens}")
        print(f"  Total tokens:      {result.usage.total_tokens}")
        print(f"\nLatency: {result.latency_ms:.2f} ms")
        print(f"Model used: {result.model}")
        
        # Show cumulative stats
        stats = client.get_stats()
        print(f"\nCumulative Statistics:")
        print(f"  Total calls:        {stats.total_calls}")
        print(f"  Successful calls:   {stats.successful_calls}")
        print(f"  Failed calls:       {stats.failed_calls}")
        print(f"  Total tokens used:  {stats.total_tokens}")
        print(f"  Average latency:    {stats.average_latency_ms():.2f} ms")
        
        print("\n" + "=" * 60)
        print("TEST PASSED - LLM client is working correctly!")
        print("=" * 60)
        return 0
        
    except Exception as e:
        print(f"\nERROR: LLM call failed!")
        print(f"Exception type: {type(e).__name__}")
        print(f"Error message: {e}")
        
        # Show failure stats
        stats = client.get_stats()
        print(f"\nCumulative Statistics (after failure):")
        print(f"  Total calls:      {stats.total_calls}")
        print(f"  Successful calls: {stats.successful_calls}")
        print(f"  Failed calls:     {stats.failed_calls}")
        
        print("\n" + "=" * 60)
        print("TEST FAILED")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
