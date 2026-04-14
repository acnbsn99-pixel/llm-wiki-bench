"""Trajectory exporter module for SFT (Supervised Fine-Tuning) data.

This module converts agent trajectories to JSONL format compatible with
OpenAI-style fine-tuning APIs.
"""

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional, Union

from ..data.models import Trajectory, BenchmarkResult


logger = logging.getLogger(__name__)


class TrajectoryExporter:
    """Exports agent trajectories to JSONL format for SFT.
    
    Converts trajectory data from the agent's internal format to OpenAI-compatible
    JSONL format for supervised fine-tuning.
    
    Expected input format (from Trajectory dataclass):
        - question_id: str
        - messages: List[dict] with role/content/timestamp
        - metadata: dict
    
    Output format (OpenAI JSONL):
        {"messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
        ]}
    """
    
    DEFAULT_SYSTEM_PROMPT = (
        "You are an AI assistant that answers questions by querying a knowledge wiki. "
        "Use the provided context to synthesize accurate, well-structured answers. "
        "Cite sources using [[PageName]] wikilink syntax when referencing specific pages."
    )
    
    def __init__(
        self,
        output_dir: Optional[Path] = None,
        system_prompt: Optional[str] = None
    ):
        """Initialize the trajectory exporter.
        
        Args:
            output_dir: Directory to save JSONL files. Defaults to project root / trajectories
            system_prompt: System prompt to prepend to conversations. If None, uses default.
        """
        self.repo_root = Path(__file__).parent.parent.parent
        self.output_dir = output_dir or self.repo_root / "trajectories"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        
    def _validate_trajectory(self, trajectory: Trajectory) -> bool:
        """Validate that a trajectory has the minimum required fields.
        
        Args:
            trajectory: Trajectory dataclass instance
            
        Returns:
            True if valid, False otherwise
        """
        # Check required fields
        if not trajectory.question_id:
            logger.warning(f"Invalid trajectory: missing question_id")
            return False
            
        if not trajectory.messages:
            logger.warning(f"Invalid trajectory {trajectory.question_id}: no messages")
            return False
            
        # Check that messages have required fields
        for i, msg in enumerate(trajectory.messages):
            if not isinstance(msg, dict):
                logger.warning(
                    f"Invalid trajectory {trajectory.question_id}: "
                    f"message {i} is not a dict"
                )
                return False
            if "content" not in msg:
                logger.warning(
                    f"Invalid trajectory {trajectory.question_id}: "
                    f"message {i} missing 'content' field"
                )
                return False
                
        return True
    
    def _convert_to_openai_format(
        self,
        trajectory: Trajectory,
        include_system_prompt: bool = True
    ) -> dict:
        """Convert a Trajectory dataclass to OpenAI JSONL format.
        
        The agent's trajectory messages may have roles like:
        - "user" with thought/prompt content
        - "assistant" with action content
        - "user" with observation content
        
        For SFT, we want to train on the final answer. We structure this as:
        - system: System prompt
        - user: The original question (from metadata or first message)
        - assistant: The final answer/observation
        
        Args:
            trajectory: Trajectory dataclass instance
            include_system_prompt: Whether to include system prompt
            
        Returns:
            Dict in OpenAI chat format
        """
        messages = []
        
        # Add system prompt if requested
        if include_system_prompt:
            messages.append({
                "role": "system",
                "content": self.system_prompt
            })
        
        # Extract user question and assistant answer from trajectory
        user_content = ""
        assistant_content = ""
        
        # Try to get question from metadata first
        if trajectory.metadata.get("question"):
            user_content = trajectory.metadata["question"]
        elif trajectory.metadata.get("original_question"):
            user_content = trajectory.metadata["original_question"]
        
        # Process messages to find the conversation flow
        # The agent logs: thought (user) -> action (assistant) -> observation (user)
        # For SFT, we want: user question -> assistant answer
        for msg in trajectory.messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            # Look for the final observation which contains the answer
            if role == "user" and content.startswith("[OBSERVATION]"):
                # Extract the actual content from observation
                assistant_content = content.replace("[OBSERVATION]", "").strip()
            elif role == "user" and not user_content:
                # Use first user message as question if not already set
                user_content = content
                
        # Fallback: if no clear answer found, use last assistant message or last observation
        if not assistant_content:
            for msg in reversed(trajectory.messages):
                content = msg.get("content", "")
                if content and not content.startswith("[ACTION]"):
                    if msg.get("role") == "assistant":
                        assistant_content = content
                        break
                    elif msg.get("role") == "user":
                        # Strip observation prefix if present
                        assistant_content = content.replace("[OBSERVATION]", "").strip()
                        break
        
        # Build the conversation
        if user_content:
            messages.append({
                "role": "user",
                "content": user_content
            })
            
        if assistant_content:
            messages.append({
                "role": "assistant",
                "content": assistant_content
            })
        
        return {"messages": messages}
    
    def convert_single(
        self,
        trajectory: Trajectory,
        validate: bool = True
    ) -> Optional[dict]:
        """Convert a single trajectory to OpenAI format.
        
        Args:
            trajectory: Trajectory dataclass instance
            validate: Whether to validate before converting
            
        Returns:
            Dict in OpenAI format, or None if invalid and validate=True
        """
        if validate and not self._validate_trajectory(trajectory):
            return None
            
        try:
            return self._convert_to_openai_format(trajectory)
        except Exception as e:
            logger.error(
                f"Error converting trajectory {trajectory.question_id}: {e}",
                exc_info=True
            )
            return None
    
    def convert_batch(
        self,
        trajectories: List[Trajectory],
        validate: bool = True
    ) -> List[dict]:
        """Convert multiple trajectories to OpenAI format.
        
        Args:
            trajectories: List of Trajectory dataclass instances
            validate: Whether to validate each trajectory
            
        Returns:
            List of dicts in OpenAI format (skips invalid ones)
        """
        converted = []
        for traj in trajectories:
            result = self.convert_single(traj, validate=validate)
            if result is not None:
                converted.append(result)
        return converted
    
    def export_single(
        self,
        trajectory: Trajectory,
        filename: Optional[str] = None,
        validate: bool = True
    ) -> Optional[Path]:
        """Export a single trajectory to a JSON file.
        
        Args:
            trajectory: Trajectory dataclass instance
            filename: Optional filename. If None, uses question_id + timestamp
            validate: Whether to validate before exporting
            
        Returns:
            Path to saved file, or None if invalid
        """
        converted = self.convert_single(trajectory, validate=validate)
        if converted is None:
            return None
            
        # Generate filename
        if filename is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{trajectory.question_id}_{timestamp}.json"
            
        filepath = self.output_dir / filename
        filepath.write_text(json.dumps(converted, indent=2))
        logger.info(f"Exported trajectory to {filepath}")
        return filepath
    
    def export_to_jsonl(
        self,
        trajectories: List[Trajectory],
        output_filename: str = "trajectories.jsonl",
        validate: bool = True
    ) -> Path:
        """Export multiple trajectories to a single JSONL file.
        
        Each line in the output file is a valid JSON object in OpenAI format.
        
        Args:
            trajectories: List of Trajectory dataclass instances
            output_filename: Name of the output JSONL file
            validate: Whether to validate each trajectory
            
        Returns:
            Path to saved JSONL file
        """
        converted = self.convert_batch(trajectories, validate=validate)
        
        output_path = self.output_dir / output_filename
        with open(output_path, "w", encoding="utf-8") as f:
            for item in converted:
                f.write(json.dumps(item) + "\n")
                
        logger.info(
            f"Exported {len(converted)} trajectories to {output_path} "
            f"(skipped {len(trajectories) - len(converted)} invalid)"
        )
        return output_path
    
    def export_from_benchmark_results(
        self,
        results: List[BenchmarkResult],
        output_filename: str = "trajectories.jsonl",
        validate: bool = True
    ) -> Path:
        """Export trajectories from BenchmarkResult objects.
        
        Extracts trajectory data from BenchmarkResult.trajectory field.
        
        Args:
            results: List of BenchmarkResult instances
            output_filename: Name of the output JSONL file
            validate: Whether to validate each trajectory
            
        Returns:
            Path to saved JSONL file
        """
        trajectories = []
        for result in results:
            if result.trajectory:
                traj = Trajectory(
                    question_id=result.question_id,
                    messages=result.trajectory.get("messages", []),
                    metadata=result.trajectory.get("metadata", {})
                )
                trajectories.append(traj)
                
        return self.export_to_jsonl(trajectories, output_filename, validate=validate)


def export_trajectories(
    trajectories: List[Trajectory],
    output_dir: Optional[Union[str, Path]] = None,
    output_filename: str = "trajectories.jsonl",
    system_prompt: Optional[str] = None,
    validate: bool = True
) -> Path:
    """Convenience function to export trajectories to JSONL.
    
    Args:
        trajectories: List of Trajectory dataclass instances
        output_dir: Directory to save output. If None, uses default
        output_filename: Name of the output JSONL file
        system_prompt: Custom system prompt. If None, uses default
        validate: Whether to validate trajectories
        
    Returns:
        Path to saved JSONL file
    """
    if output_dir:
        output_dir = Path(output_dir)
        
    exporter = TrajectoryExporter(output_dir=output_dir, system_prompt=system_prompt)
    return exporter.export_to_jsonl(trajectories, output_filename, validate=validate)
