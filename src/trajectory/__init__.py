"""Trajectory module for llm-vs-rag-bench.

Provides trajectory export functionality for SFT (Supervised Fine-Tuning).
"""

from .exporter import TrajectoryExporter, export_trajectories


__all__ = ["TrajectoryExporter", "export_trajectories"]
