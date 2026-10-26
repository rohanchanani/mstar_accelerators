"""Platform-owned qualification policy, independent of acceleration authors.

The platform corpus provides logical models/assets; hosts provide M* execution context. Only this
module chooses reference adapters, coverage defaults and the comparator.
Changing acceptance or coverage requires a separate, justified policy change.
"""
from .harness import DifferentialConsumer
from .numerics import DEFAULT, differential_comparator

# Existing Qwen default domain; these are not environment/user overrides.
QWEN_BATCHES = (1, 2, 8)
QWEN_DECODE_STEPS = 3
POLICY_ID = "registry-differential-v2"


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
            "generation": "generation-reciprocal-top5-v1",
            "conditioning": "hidden-cosine-norm-v1",
            "qwen_reference": {"Talker": "stock HF, matched precision/device and token history",
                               "Thinker": "retained MStar BF16 through the engine",
                               "audio_encoder": "retained HF through the engine"},
            "case_selection": "declared-alternatives-v1",
            "qwen_batches": list(QWEN_BATCHES), "qwen_decode_steps": QWEN_DECODE_STEPS,
            "placement_tp_sizes": list(TP_SIZES), "placement_sp_sizes": [1]}

# Bounded placement coverage shared by every logical model and implementation.
# Environment discovery reports feasibility; it does not choose test coverage.
TP_SIZES = (1, 2)


def dtype_for(model):
    import torch
    from mstar.model.whisper.config import WhisperModelConfig
    # Preserve the existing logical test representations, independent of device.
    return torch.float32 if isinstance(model.config, WhisperModelConfig) else model.get_autocast_dtype()


def context(model, *, fixtures, device, entries, stream, parallel_groups):
    """Model semantics only. Backend hosting never constructs a model or oracle."""
    from mstar.model.whisper.config import WhisperModelConfig
    from mstar.model.qwen3_omni.config import Qwen3OmniModelConfig
    from .harness import Unavailable
    if isinstance(model.config, WhisperModelConfig):
        from .whisper import QualificationSession, audio_inputs
        inputs, identity = audio_inputs(model, fixtures['audio'], language='en', task='transcribe')
        session = QualificationSession()
        return session, consumer(model, session=session, device=device, options={},
            features=inputs['audio_features'][0], corpus=entries, stream=stream,
            parallel_groups=parallel_groups), {'fixture': identity}
    if isinstance(model.config, Qwen3OmniModelConfig):
        if any(group.world_size != 1 for group in parallel_groups.node_to_tp_group.values()):
            raise Unavailable('case_domain', 'central Qwen trajectories currently cover TP1/SP1 only')
        from .models.qwen3_omni import QualificationSession
        session = QualificationSession()
        return session, consumer(model, session=session, device=device,
            options={'dtype': str(dtype_for(model)).removeprefix('torch.')}), {}
    raise Unavailable('reference_adapter', 'no central case constructor for this logical model')
