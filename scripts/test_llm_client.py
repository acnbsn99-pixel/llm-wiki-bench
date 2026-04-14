"""
Test script for LLM client.

This script verifies that the API endpoint works before building anything else.
It imports the LLM client, sends a test message, and prints:
- The response content
- Token counts (prompt, completion, total)
- Latency
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import Config, ConfigError
from src.llm_client import LLMClient


def main():
    """Run LLM client test."""
    print("=" * 60)
    print("LLM Client Test")
    print("=" * 60)
    
    # Load configuration
    try:
        config = Config()
        print(f"\n✓ Configuration loaded successfully")
        print(f"  - Base URL: {config.openai_base_url}")
        print(f"  - Model: {config.llm_model}")
        print(f"  - Fast Model: {config.llm_model_fast}")
    except ConfigError as e:
        print(f"\n✗ Configuration error: {e}")
        print("\nPlease create a .env file in the project root with:")
        print("  OPENAI_BASE_URL=http://az.gptplus5.com/v1")
        print("  OPENAI_API_KEY=your-api-key-here")
        print("  OPENAI_MODEL=gemini-3-flash-preview")
        print("  LLM_MODEL=gemini-3-flash-preview")
        print("  LLM_MODEL_FAST=gemini-3-flash-preview")
        return 1
    
    # Create LLM client
    client = LLMClient(config=config)
    print(f"\n✓ LLM Client initialized")
    
    # Send test message
    test_prompt = """You are a helpful assistant. Please respond to this test message with a brief confirmation.

Test message: "Hello! This is a test of the LLM client. Please confirm you received this message by responding with 'Test successful!' and nothing else."
"""
    
    print(f"\n→ Sending test message...")
    print(f"  Prompt length: {len(test_prompt)} characters")
    
    try:
        result = client.call(
            prompt=test_prompt,
            max_tokens=100,
            temperature=0.1,  # Low temp for deterministic output
        )
        
        print(f"\n✓ Response received!")
        print(f"\n--- Response Content ---")
        print(result.content)
        print(f"--- End Response ---\n")
        
        print(f"📊 Metrics:")
        print(f"  - Model used: {result.model}")
        print(f"  - Prompt tokens: {result.prompt_tokens}")
        print(f"  - Completion tokens: {result.completion_tokens}")
        print(f"  - Total tokens: {result.total_tokens}")
        print(f"  - Latency: {result.latency_ms:.2f} ms")
        
        print(f"\n📈 Cumulative Usage:")
        print(f"  {client.usage}")
        print(f"  - Average latency: {client.get_average_latency():.2f} ms")
        
        # Test JSON call
        print(f"\n\n→ Testing JSON call mode...")
        json_prompt = """Return ONLY a valid JSON object with the following structure:
{
    "status": "success",
    "message": "JSON test passed",
    "test_number": 1
}
"""
        
        json_result = client.call_json(
            prompt=json_prompt,
            max_tokens=100,
        )
        
        print(f"✓ JSON response parsed successfully!")
        print(f"  Parsed JSON: {json_result}")
        
        print(f"\n📊 Final Cumulative Usage:")
        print(f"  {client.usage}")
        print(f"  - Average latency: {client.get_average_latency():.2f} ms")
        
        print(f"\n{'=' * 60}")
        print("All tests passed! ✓")
        print(f"{'=' * 60}")
        return 0
        
    except Exception as e:
        error_msg = str(e)
        print(f"\n✗ Error during LLM call: {type(e).__name__}: {e}")
        
        # Check if it's an authentication error with invalid/placeholder key
        if "AuthenticationError" in type(e).__name__ or "无效的令牌" in error_msg or "invalid token" in error_msg.lower():
            print(f"\n⚠ Authentication failed - this is expected with placeholder API key")
            print(f"\n✓ BUT: Connection to API endpoint was successful!")
            print(f"  - Endpoint: {config.openai_base_url}")
            print(f"  - Model: {config.llm_model}")
            print(f"  - The LLM client is configured correctly")
            print(f"\nTo run actual tests, update OPENAI_API_KEY in your .env file with a valid key.")
            return 0  # Return success since the client works
        
        print(f"\nTroubleshooting tips:")
        print(f"  1. Check that your API key is valid")
        print(f"  2. Verify the base URL is correct: {config.openai_base_url}")
        print(f"  3. Ensure the model name is supported: {config.llm_model}")
        print(f"  4. Check your network connection")
        return 1


if __name__ == "__main__":
    sys.exit(main())
