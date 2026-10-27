"""Optional ordinary Torch execution; no graph language or native provider."""
from .programs import TorchProgram, WorkspaceCapacity, prepare_torch_programs

__all__ = ["TorchProgram", "WorkspaceCapacity", "prepare_torch_programs"]
