"""Optional ordinary Torch execution; no graph language or native provider."""
from .programs import TorchProgram, prepare_torch_programs

__all__ = ["TorchProgram", "prepare_torch_programs"]
