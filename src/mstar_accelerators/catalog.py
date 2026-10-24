"""Discover the repository corpus without an independently maintained index."""
from pathlib import Path

from .sources import load_accelerations

accelerations = load_accelerations(Path(__file__).resolve().parents[2] / "accelerations")
