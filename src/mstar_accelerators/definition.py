from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from .conditions import Condition
from mstar.engine.execution import InvocationFacts
from mstar.engine.tensor_forward import CapturePolicy, CompilePolicy


@dataclass(frozen=True)
class TensorForwardExecution:
    """Request M*'s tensor-forward runner for a separately prepared program.

    The program owns its computation, packing and capture configurations. M*
    owns static storage, graph capture/replay and completion. Resource views
    are still requested before preparation through ``Implementation.resources``.
    """

    compile: CompilePolicy = field(default_factory=CompilePolicy)
    capture: CapturePolicy = field(default_factory=CapturePolicy)
    autocast_dtype: Any = None


@dataclass(frozen=True)
class PreparedImplementation:
    materialize: Callable
    primitive: Callable
    acquire: Callable[[InvocationFacts], Any | None]
    close: Callable[[], None]


@dataclass(frozen=True)
class Implementation:
    prepare: Callable
    # Physical access requirements for the logical resources this node uses.
    # Values are one typed request or a tuple of requests for compatible views.
    resources: Mapping[str, Any] | Callable = field(default_factory=dict)
    cost: Callable | None = None
    # None consumes PreparedImplementation. TensorForwardExecution consumes
    # an independent physical nn.Module; no logical shell mutation is needed.
    execution: TensorForwardExecution | None = None


@dataclass(frozen=True)
class Acceleration:
    reference: type
    when: Condition
    implementation: Implementation
