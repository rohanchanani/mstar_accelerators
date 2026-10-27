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
POLICY_ID = "registry-layered-model-validation-v2"


def consumer(model, *, session, device, options, corpus=None, features=None,
             stream=None, parallel_groups=None, scope=None):
    """Resolve platform-owned adapters from the logical model, not its optimizer."""
    from mstar.model.whisper.config import WhisperModelConfig

    if isinstance(model.config, WhisperModelConfig):
        from .models.whisper import adapters
        references = adapters(model, features, session=session, device=device,
                              corpus=corpus, stream=stream, parallel_groups=parallel_groups,
                              model_config=options.get("model_config", {}), quality=options.get("quality"), language=options.get("language", "en"),
                              task=options.get("task", "transcribe"), scope=scope)
    else:
        from mstar.model.qwen3_omni.config import Qwen3OmniModelConfig
        if not isinstance(model.config, Qwen3OmniModelConfig):
            return DifferentialConsumer({}, differential_comparator)
        from .models.qwen3_omni import adapters
        references = adapters(model, session=session, device=device, options=options)
    from mstar.model.fidelity import resolve_fidelity
    from .quality import quality_comparator
    fidelity = resolve_fidelity(options.get("model_config", {}))
    return DifferentialConsumer(references, quality_comparator if fidelity.mode == "quality" else differential_comparator)


def identity():
    return {"id": POLICY_ID, "numerics": DEFAULT.name,
            "fidelity": {"producer": "deployment", "numerical_default": DEFAULT.name,
                         "quality": "model-declared baseline; central metric and M* continuation"},
            "generation": "vllm-first-divergence-top5-v2; exact sampler semantics on all tokens; later conditional comparisons diagnostic",
            "speech": "paired-speech-content-v1; each clip and its opening sentence; no deployment opt-in",
            "conditioning": DEFAULT.name,
            "qwen_reference": {"Talker": "stock HF through M*, matched precision and token history",
                               "Thinker": "stock HF through M*, matched precision and token history",
                               "audio_encoder": "stock HF through M*, matched precision"},
            "promoted_reference": {"trigger": "ordinary numerical allclose failure only",
                                   "execution": "M*; FP64; CPU permitted"},
            "case_selection": "declared-alternatives-v1",
            "qwen_batches": list(QWEN_BATCHES), "qwen_decode_steps": QWEN_DECODE_STEPS,
            "placement_tp_sizes": list(TP_SIZES), "placement_sp_sizes": [1]}

# Bounded placement coverage shared by every logical model and implementation.
# Environment discovery reports feasibility; it does not choose test coverage.
TP_SIZES = (1, 2)


def dtype_for(model, deployment=None):
    import torch
    from mstar.model.binding import resolve_autocast_dtype
    return resolve_autocast_dtype(model, deployment or {}) or torch.float32


def context(model, *, fixtures, device, entries, stream, parallel_groups, model_config=None, model_identity=None):
    """Model semantics only. Backend hosting never constructs a model or oracle."""
    from mstar.model.whisper.config import WhisperModelConfig
    from .harness import Unavailable
    if isinstance(model.config, WhisperModelConfig):
        from .whisper import QualificationSession, audio_inputs
        inputs, identity = audio_inputs(model, fixtures['audio'], language='en', task='transcribe')
        from .quality import resolve_quality
        quality = resolve_quality(model, model_config or {}, model_identity or {})
        session = QualificationSession()
        return session, consumer(model, session=session, device=device, options={"model_config": model_config or {}, "quality": quality},
            features=inputs['audio_features'][0], corpus=entries, stream=stream,
            parallel_groups=parallel_groups), {'fixture': identity, 'fidelity': (model_config or {}).get('fidelity', {'mode':'numerical'}), 'quality': quality}
    from mstar.model.qwen3_omni.config import Qwen3OmniModelConfig
    if isinstance(model.config, Qwen3OmniModelConfig):
        if any(group.world_size != 1 for group in parallel_groups.node_to_tp_group.values()):
            raise Unavailable('case_domain', 'central Qwen trajectories currently cover TP1/SP1 only')
        from .models.qwen3_omni import QualificationSession
        session = QualificationSession()
        return session, consumer(model, session=session, device=device,
            options={'dtype': str(dtype_for(model)).removeprefix('torch.')}), {}
    raise Unavailable('reference_adapter', 'no central case constructor for this logical model')
