"""Torch-facing Pi backend.

Import registers the backend. ``initialize`` creates this process's native
context; call it in each spawned worker before constructing Pi tensors.
There is no import-time device open, implicit CPU fallback, or model server.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
import sys

import torch

_NAME = "mstar_pi"
_existing_name = torch._C._get_privateuse1_backend_name()
if _existing_name not in ("privateuseone", _NAME):
    raise RuntimeError(
        f"PrivateUse1 is already registered as {_existing_name!r}; "
        "use a separate worker process for mstar_pi"
    )
from . import _C

__version__ = "0.4.0"
# Completion events are first class; this backend does not expose timestamp events.
supports_timing_events = False
torch.utils.rename_privateuse1_backend(_NAME)
torch._register_device_module(_NAME, sys.modules[__name__])

FatalError = _C.FatalError
Rejected = _C.Rejected
PreparedCsd = _C.PreparedCsd
PreparedLinear = _C.PreparedLinear
BufferAddress = _C.BufferAddress


def initialize(
    *,
    copy_shader: str | Path | None = None,
    render_node: str = "/dev/dri/renderD128",
    wait_timeout_ms: int = 30_000,
    validation: bool = False,
) -> None:
    """Open one native context in this process.

    Initialize once and use ``multiprocessing``'s ``spawn`` start method.
    Reinitializing a closed/fatal context is intentionally unsupported: existing
    Torch streams carry IDs which cannot be silently reinterpreted in a new
    context. Start a fresh worker to establish a new context epoch.
    """
    if copy_shader is None:
        copy_shader = Path(__file__).with_name("transfer.spv")
    copy_shader = Path(copy_shader).expanduser().resolve(strict=True)
    if wait_timeout_ms <= 0:
        raise ValueError("wait_timeout_ms must be positive")
    _C.initialize(str(copy_shader), render_node, wait_timeout_ms, validation)


def shutdown() -> None:
    """Drain and close; rejects while tensor/staging Storage owners remain.

    A fatal context cannot be made healthy by this function. Native quarantine
    retains uncertain uses; recovery is a separate backend/worker operation.
    """
    _C.shutdown()


def is_available() -> bool:
    """Whether the default Pi target is discoverable or the context is healthy.

    Before initialization this performs a read-only DRM capability probe. It
    creates no Vulkan context, queues, weights or native work. Initialization
    still validates the Vulkan provider and must be called explicitly.
    """
    return _C.is_available()


def is_initialized() -> bool:
    """A native context is retained, including after an incomplete close.

    New work additionally requires open admission; ``is_available`` and
    ``health`` expose that distinction. A retained context still provides the
    completion/drain path after a nonfatal host-side close exception.
    """
    return _C.is_initialized()


def _is_in_bad_fork() -> bool:
    return _C.is_in_bad_fork()


def _lazy_init() -> None:
    if not is_initialized():
        raise RuntimeError("call mstar_pi.initialize() in this spawned worker first")


def device_count() -> int:
    return 1 if is_available() else 0


def probe(render_node: str = "/dev/dri/renderD128") -> bool:
    """Read-only discovery for an explicitly selected render node."""
    return _C.probe(render_node)


def current_device() -> int:
    _lazy_init()
    return 0


def _device_argument(value=None) -> torch.device:
    parsed = torch.device(_NAME, value) if isinstance(value, int) else torch.device(value or _NAME)
    if parsed.type != _NAME or parsed.index not in (None, 0):
        raise ValueError("this backend instance exposes mstar_pi:0 only")
    return torch.device(_NAME, 0)


def Stream(device=None, *, priority: int = 0) -> torch.Stream:
    """Backend-module facade over Torch's actual generic stream object."""
    return torch.Stream(device=_device_argument(device), priority=priority)


def Event(*, enable_timing: bool = False) -> torch.Event:
    """Backend-module facade over a generic Torch event; timing is unsupported."""
    if enable_timing:
        raise NotImplementedError("mstar_pi does not expose event timing")
    return torch.Event(device=_NAME, enable_timing=False)


def stream(value: torch.Stream | None):
    """Select an actual Torch stream through its existing context manager."""
    if value is None:
        return contextlib.nullcontext()
    _device_argument(value.device)
    return value


def current_stream(device=None) -> torch.Stream:
    return torch.accelerator.current_stream(_device_argument(device))


def set_device(device: int | str | torch.device) -> None:
    _device_argument(device)
    _lazy_init()


@contextlib.contextmanager
def device(index: int | str | torch.device):
    set_device(index)
    yield


def synchronize(device: int | None = None) -> None:
    if device not in (None, 0):
        raise ValueError("this backend instance exposes mstar_pi:0 only")
    _C.synchronize()


def collect() -> None:
    """Retire completed native uses without waiting for pending work."""
    _C.collect()


def health() -> dict:
    """Authoritative context status, including after an ATen-translated error.

    Torch may translate a C++ native exception into RuntimeError. A worker must
    inspect this status after an execution error, rather than classify it by
    matching exception text.
    """
    return _C.health()


def capabilities() -> dict:
    return {
        "device": _NAME,
        "tensor_dtypes": (torch.float32, torch.int32, torch.int64),
        "copy_layout": "contiguous, same shape and dtype",
        "pinned_cpu_storage": True,
        "timing_events": False,
        "graph_capture": False,
        "inference_only": True,
        "prepared_ggml_linear": {
            "dtype": torch.float32,
            "layout": "contiguous 2-D",
            "output": "caller-owned Pi tensor",
            "completion": "current Torch stream",
        },
    }


def address(tensor: torch.Tensor, byte_offset: int = 0) -> BufferAddress:
    """A checked native address operand retaining its backing allocation.

    The argument must be a contiguous Pi tensor or this backend's pinned CPU
    tensor. This does not import arbitrary CPU pointers. A native program must
    declare additional indirectly addressed tensors through ``uses``.
    """
    return _C.address(tensor, byte_offset)


def prepare_csd(words: list[int]) -> PreparedCsd:
    """Upload an implementation-owned native program once.

    The program's correctness, bounds and supported domain remain the author's
    responsibility. Submission binds typed address operands and retains every
    declared use through its completion.
    """
    return _C.prepare_csd(words)


def prepare_linear(
    weights: torch.Tensor, *, token_counts: list[int] | tuple[int, ...], slots: int = 2,
) -> PreparedLinear:
    """Prepare real ggml Vulkan ``out = input @ weights.T`` executions.

    ``weights`` is a contiguous float32 Pi tensor of shape ``[outputs, inputs]``.
    Its native storage is retained directly, without another Torch Storage
    owner, repacking, or an implicit host copy. Every ``token_counts`` entry
    prepares a shape ``input[tokens, inputs] -> out[tokens, outputs]``; ``slots``
    bounds simultaneous mutable execution contexts for each prepared shape.

    ``program.submit(input, out)`` uses the current ordinary Torch stream. It
    borrows the checked input/output views, joins their native dependencies,
    and publishes a completion tail before returning. Immediate Torch clone,
    device copies and worker-style events therefore consume the same ordering.
    The caller owns output storage; a returned invocation never hides scratch
    storage behind an ordinary Tensor. Destructive input/output overlap rejects.

    Shape preparation and pipeline creation happen here. Lack of a free slot
    during submit is explicit rejection, never an implicit host completion wait.
    Completed slots are collected before each submission; consumers may also
    call ``collect()`` after waiting on an event. No GPU initialization occurs
    until the caller has explicitly initialized this worker's native context.
    """
    return _C.PreparedLinear(weights, list(token_counts), slots)


def manual_seed_all(seed: int) -> None:
    # torch.manual_seed invokes this hook for every initialized accelerator.
    # No random kernels or per-device RNG are supplied by this first slice.
    # Accepting a seed here must not imply a native sampling implementation.
    return None


def get_rng_state(device: int = 0):
    raise NotImplementedError("the first Pi backend slice has no random operators")


def set_rng_state(state, device: int = 0):
    raise NotImplementedError("the first Pi backend slice has no random operators")

