"""Environment inputs only: no executable plugins or acceptance/coverage knobs."""
from collections.abc import Mapping

_COMMON = {"checkpoint", "threads"}
_FIELDS = {
    "pi": _COMMON | {"audio", "language", "task", "wait_timeout_ms", "validation", "in_process"},
    "cpu": _COMMON | {"audio", "language", "task", "tp_size", "collective_timeout_seconds", "deadline_seconds"},
    "rocm": _COMMON | {"device", "dtype", "deployment", "in_process"},
}


def environment(kind, options):
    if kind not in _FIELDS:
        raise ValueError(f"unknown execution environment: {kind!r}")
    if not isinstance(options, Mapping):
        raise ValueError("environment options must be an object")
    unknown = set(options) - _FIELDS[kind]
    if unknown:
        raise ValueError(f"unsupported environment options: {sorted(unknown)}; "
                         "reference, coverage and comparison policy belong to the central harness")
    if kind == "rocm":
        import torch
        if torch.device(options.get("device", "cuda:0")).type != "cuda":
            raise ValueError("the ROCm environment requires a CUDA/HIP device")
    if "validation" in options and options["validation"] is not True:
        raise ValueError("qualification requires backend validation")
    return dict(options)


def environments(config):
    if not isinstance(config, Mapping):
        raise ValueError("qualification config must be an object")
    shared = {key: value for key, value in config.items() if key != "environments"}
    placements = config.get("environments", [{"kind": "pi"}])
    if not isinstance(placements, list) or not placements:
        raise ValueError("environments must be a nonempty list")
    result = []
    for placement in placements:
        if not isinstance(placement, Mapping) or "kind" not in placement:
            raise ValueError("each environment needs a kind and environment options")
        kind = placement["kind"]
        options = {**shared, **{key: value for key, value in placement.items() if key != "kind"}}
        result.append((kind, environment(kind, options)))
    return result
