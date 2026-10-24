"""An acceleration describes a reference, its supported circumstances and a program."""

from .definition import Acceleration, Implementation, PreparedImplementation, TensorForwardExecution
from .conditions import All, AllRows, Eq, In, Or, Range, When
from .registry import Registry

__all__ = ["Acceleration", "Implementation", "PreparedImplementation", "TensorForwardExecution", "All", "AllRows", "Eq", "In", "Or", "Range", "When", "Registry"]
