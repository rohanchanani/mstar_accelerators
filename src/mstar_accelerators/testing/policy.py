"""Platform-owned qualification policy, independent of acceleration authors.

Environment hosts provide model/assets and M* execution context. Only this
module chooses reference adapters, coverage defaults and the comparator.
Changing acceptance or coverage requires a separate, justified policy change.
"""
from .harness import DifferentialConsumer
from .numerics import DEFAULT, differential_comparator

# Existing Qwen default domain; these are not environment/user overrides.
QWEN_BATCHES = (1, 2, 8)
QWEN_DECODE_STEPS = 3
POLICY_ID = "registry-differential-v1"


def consumer(model, *, session, device, options, corpus=None, features=None,
             stream=None, parallel_groups=None, scope=None):
    """Resolve platform-owned adapters from the logical model, not its optimizer."""
    from mstar.model.whisper.config import WhisperModelConfig
    from mstar.model.qwen3_omni.config import Qwen3OmniModelConfig

    if isinstance(model.config, WhisperModelConfig):
        from .models.whisper import adapters
        references = adapters(model, features, session=session, device=device,
                              corpus=corpus, stream=stream, parallel_groups=parallel_groups,
                              language=options.get("language", "en"),
                              task=options.get("task", "transcribe"), scope=scope)
    elif isinstance(model.config, Qwen3OmniModelConfig):
        from .models.qwen3_omni import adapters
        references = adapters(model, session=session, device=device, options=options)
    else:
        references = {}  # Missing executable reference remains SKIP, never PASS.
    return DifferentialConsumer(references, differential_comparator)


def identity():
    return {"id": POLICY_ID, "numerics": DEFAULT.name,
            "case_selection": "declared-alternatives-v1",
            "qwen_batches": list(QWEN_BATCHES), "qwen_decode_steps": QWEN_DECODE_STEPS}
