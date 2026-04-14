#!/usr/bin/env python3
"""Test script for trajectory export module.

Tests the TrajectoryExporter with mock data based on actual Trajectory dataclass
and agent output patterns observed in src/llm_wiki/tracking.py and query.py.
"""

import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.models import Trajectory, BenchmarkResult
from src.trajectory.exporter import TrajectoryExporter


def test_basic_trajectory():
    """Test exporting a basic valid trajectory."""
    print("=" * 60)
    print("TEST 1: Basic Valid Trajectory")
    print("=" * 60)
    
    # Create a trajectory matching the format from tracking.py
    trajectory = Trajectory(
        question_id="test_001",
        messages=[
            {
                "role": "user",
                "content": "Given this wiki index:\n\n...",
                "timestamp": "2024-01-01T00:00:00"
            },
            {
                "role": "assistant",
                "content": "[ACTION] select_pages",
                "timestamp": "2024-01-01T00:00:01"
            },
            {
                "role": "user",
                "content": "[OBSERVATION] Selected 5 pages",
                "timestamp": "2024-01-01T00:00:02"
            },
            {
                "role": "user",
                "content": "You are querying an LLM Wiki to synthesize an answer...",
                "timestamp": "2024-01-01T00:00:03"
            },
            {
                "role": "assistant",
                "content": "[ACTION] synthesize_answer",
                "timestamp": "2024-01-01T00:00:04"
            },
            {
                "role": "user",
                "content": "[OBSERVATION] The Albendazole treatment protocol involved...",
                "timestamp": "2024-01-01T00:00:05"
            }
        ],
        metadata={
            "question": "What was the method of Albendazole administration?",
            "relevant_pages_count": 5,
            "tokens_used": 1234,
            "latency_ms": 5678
        }
    )
    
    exporter = TrajectoryExporter()
    result = exporter.convert_single(trajectory)
    
    print(f"Input trajectory.question_id: {trajectory.question_id}")
    print(f"Input trajectory.messages count: {len(trajectory.messages)}")
    print(f"Input trajectory.metadata keys: {list(trajectory.metadata.keys())}")
    print()
    print("Output JSON:")
    print(json.dumps(result, indent=2))
    
    # Validate output structure
    assert "messages" in result, "Missing 'messages' key"
    roles = [m["role"] for m in result["messages"]]
    print(f"\nMessage roles: {roles}")
    
    # Should have system, user, assistant
    assert "system" in roles, "Missing system message"
    assert "user" in roles, "Missing user message"
    assert "assistant" in roles, "Missing assistant message"
    
    print("\n✅ TEST 1 PASSED\n")
    return True


def test_trajectory_from_benchmark_result():
    """Test exporting from BenchmarkResult.trajectory field."""
    print("=" * 60)
    print("TEST 2: From BenchmarkResult")
    print("=" * 60)
    
    # Create a BenchmarkResult matching the format from query.py
    result = BenchmarkResult(
        pipeline_name="llm-wiki-agent",
        question_id="test_002",
        predicted_answer="The treatment involved 3 cycles of Albendazole...",
        latency_seconds=5.678,
        token_usage=1234,
        retrieval_count=5,
        trajectory={
            "messages": [
                {
                    "role": "user",
                    "content": "Which pages are relevant to: What was the method...",
                    "timestamp": "2024-01-01T00:00:00"
                },
                {
                    "role": "user",
                    "content": "[OBSERVATION] Based on the wiki sources, the Albendazole "
                              "was administered in 3 cycles of 28 days each with 14-day rest.",
                    "timestamp": "2024-01-01T00:00:05"
                }
            ],
            "metadata": {
                "relevant_pages_count": 3,
                "llm_calls": 2
            }
        }
    )
    
    exporter = TrajectoryExporter()
    trajectories = []
    if result.trajectory:
        traj = Trajectory(
            question_id=result.question_id,
            messages=result.trajectory.get("messages", []),
            metadata=result.trajectory.get("metadata", {})
        )
        trajectories.append(traj)
    
    converted = exporter.convert_batch(trajectories)
    
    print(f"Input BenchmarkResult.question_id: {result.question_id}")
    print(f"Input BenchmarkResult.trajectory.messages count: {len(result.trajectory['messages'])}")
    print()
    print("Output JSON:")
    print(json.dumps(converted[0], indent=2))
    
    assert len(converted) == 1, "Should have 1 converted trajectory"
    assert "messages" in converted[0], "Missing 'messages' key"
    
    print("\n✅ TEST 2 PASSED\n")
    return True


def test_invalid_trajectory():
    """Test that invalid trajectories are properly rejected."""
    print("=" * 60)
    print("TEST 3: Invalid Trajectory (Validation)")
    print("=" * 60)
    
    exporter = TrajectoryExporter()
    
    # Test missing question_id
    invalid1 = Trajectory(
        question_id="",
        messages=[{"role": "user", "content": "test"}],
        metadata={}
    )
    result1 = exporter.convert_single(invalid1, validate=True)
    print(f"Empty question_id: {result1} (expected None)")
    assert result1 is None, "Should reject empty question_id"
    
    # Test missing messages
    invalid2 = Trajectory(
        question_id="test_invalid",
        messages=[],
        metadata={}
    )
    result2 = exporter.convert_single(invalid2, validate=True)
    print(f"Empty messages: {result2} (expected None)")
    assert result2 is None, "Should reject empty messages"
    
    # Test missing content field
    invalid3 = Trajectory(
        question_id="test_invalid",
        messages=[{"role": "user"}],  # Missing 'content'
        metadata={}
    )
    result3 = exporter.convert_single(invalid3, validate=True)
    print(f"Missing content field: {result3} (expected None)")
    assert result3 is None, "Should reject message without content"
    
    print("\n✅ TEST 3 PASSED\n")
    return True


def test_export_to_jsonl():
    """Test exporting multiple trajectories to JSONL file."""
    print("=" * 60)
    print("TEST 4: Export to JSONL File")
    print("=" * 60)
    
    # Create multiple trajectories
    trajectories = [
        Trajectory(
            question_id=f"test_{i:03d}",
            messages=[
                {"role": "user", "content": f"Question {i}"},
                {"role": "user", "content": f"[OBSERVATION] Answer {i}"}
            ],
            metadata={"question": f"What is question {i}?"}
        )
        for i in range(3)
    ]
    
    exporter = TrajectoryExporter(output_dir=Path(__file__).parent.parent / "trajectories")
    output_path = exporter.export_to_jsonl(trajectories, output_filename="test_trajectories.jsonl")
    
    print(f"Exported to: {output_path}")
    print(f"File exists: {output_path.exists()}")
    
    # Read and verify JSONL
    with open(output_path, "r") as f:
        lines = f.readlines()
    
    print(f"Lines in file: {len(lines)}")
    for i, line in enumerate(lines):
        data = json.loads(line)
        print(f"Line {i+1}: {data['messages'][0]['content'][:50]}...")
    
    assert len(lines) == 3, "Should have 3 lines"
    for line in lines:
        data = json.loads(line)
        assert "messages" in data, "Each line should have 'messages'"
    
    print("\n✅ TEST 4 PASSED\n")
    return True


def test_realistic_agent_trajectory():
    """Test with a realistic trajectory matching actual llm-wiki-agent output."""
    print("=" * 60)
    print("TEST 5: Realistic Agent Trajectory")
    print("=" * 60)
    
    # This mimics what tracking.py actually produces in end_query()
    trajectory = Trajectory(
        question_id="healthcare_q001",
        messages=[
            {
                "role": "user",
                "content": (
                    "Given this wiki index:\n\n"
                    "- [Medical Treatment Protocols](sources/medical_protocols.md)\n"
                    "- [Drug Administration Guidelines](sources/drug_admin.md)\n\n"
                    "Which pages are most relevant to answering: "
                    "\"What was the method of Albendazole administration after surgery "
                    "for hydatid cysts in the cases reported in the June 2021 volume "
                    "of Ann Coll Med Mosul, and what outcomes were observed?\"\n\n"
                    "Return ONLY a JSON array of relative file paths..."
                ),
                "timestamp": "2024-01-01T12:00:00"
            },
            {
                "role": "assistant",
                "content": "[ACTION] select_pages",
                "timestamp": "2024-01-01T12:00:01"
            },
            {
                "role": "user",
                "content": "[OBSERVATION] Selected 2 pages",
                "timestamp": "2024-01-01T12:00:02"
            },
            {
                "role": "user",
                "content": (
                    "You are querying an LLM Wiki to synthesize an answer. "
                    "Use the wiki pages below to synthesize a thorough answer...\n\n"
                    "Wiki pages:\n"
                    "### sources/medical_protocols.md\n"
                    "Albendazole treatment protocol...\n\n"
                    "Question: What was the method of Albendazole administration..."
                ),
                "timestamp": "2024-01-01T12:00:03"
            },
            {
                "role": "assistant",
                "content": "[ACTION] synthesize_answer",
                "timestamp": "2024-01-01T12:00:04"
            },
            {
                "role": "user",
                "content": (
                    "[OBSERVATION] ## Answer\n\n"
                    "Based on the case reports from June 2021 Ann Coll Med Mosul:\n\n"
                    "**Albendazole Administration Method:**\n"
                    "- 3 cycles of 28 days each\n"
                    "- 14-day rest period between cycles\n"
                    "- Administered post-surgically\n\n"
                    "**Outcomes:**\n"
                    "- Patients showed significant improvement\n"
                    "- No recurrence observed at 6-month follow-up\n\n"
                    "## Sources\n"
                    "- [[Medical Treatment Protocols]]\n"
                    "- [[Drug Administration Guidelines]]"
                ),
                "timestamp": "2024-01-01T12:00:05"
            }
        ],
        metadata={
            "question": (
                "What was the method of Albendazole administration after surgery "
                "for hydatid cysts in the cases reported in the June 2021 volume "
                "of Ann Coll Med Mosul, and what outcomes were observed?"
            ),
            "relevant_pages_count": 2,
            "relevant_pages": [
                "sources/medical_protocols.md",
                "sources/drug_admin.md"
            ],
            "tokens_used": 1567,
            "latency_ms": 4532,
            "llm_calls": 2
        }
    )
    
    exporter = TrajectoryExporter()
    result = exporter.convert_single(trajectory)
    
    print("Input trajectory structure:")
    print(f"  - question_id: {trajectory.question_id}")
    print(f"  - messages count: {len(trajectory.messages)}")
    print(f"  - metadata keys: {list(trajectory.metadata.keys())}")
    print()
    print("Output JSONL format:")
    print(json.dumps(result, indent=2))
    
    # Verify structure
    assert "messages" in result
    roles = [m["role"] for m in result["messages"]]
    assert roles == ["system", "user", "assistant"], f"Expected ['system', 'user', 'assistant'], got {roles}"
    
    # Verify content extraction
    user_msg = next(m for m in result["messages"] if m["role"] == "user")
    assert "Albendazole" in user_msg["content"], "User message should contain the question"
    
    assistant_msg = next(m for m in result["messages"] if m["role"] == "assistant")
    assert "3 cycles" in assistant_msg["content"], "Assistant should contain the answer"
    assert "[[Medical Treatment Protocols]]" in assistant_msg["content"], "Should preserve citations"
    
    print("\n✅ TEST 5 PASSED\n")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("TRAJECTORY EXPORT MODULE TESTS")
    print("=" * 60 + "\n")
    
    tests = [
        test_basic_trajectory,
        test_trajectory_from_benchmark_result,
        test_invalid_trajectory,
        test_export_to_jsonl,
        test_realistic_agent_trajectory
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ TEST FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
