"""Accelerator-owned audio encoder with request-local attention metadata."""
import inspect
import logging
from typing import Any, Optional
import torch
from torch import nn
from transformers import AutoConfig
from mstar.communication.tensors import NameToTensorList
from mstar.model.submodule_base import ModelInputsFromEngine
from mstar.model.components.hf_attention import build_hf_encoder
from mstar.model.qwen3_omni.components.attention import patch_hf_fa2_int_maxlen
from .audio_metadata import AudioEncoderWithLengths

logger = logging.getLogger(__name__)

def _accepts_kwarg(fn: Any, name: str) -> bool:
    """True if fn takes `name` as a keyword (explicitly or via **kwargs)."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return True
    param = params.get(name)
    return param is not None and param.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    )


class AudioProgram(nn.Module):
    def __init__(self, audio_encoder):
        super().__init__()
        self.audio_encoder = audio_encoder
        self._encoder_takes_return_dict = _accepts_kwarg(audio_encoder.forward, "return_dict")

    def preprocess(self, graph_walk, engine_inputs, inputs):
        if len(inputs) > 1:
            raise NotImplementedError(
                f"Batching not implemented for submodule {self.__class__.__name__}"
            )
        return {**inputs[0].tensor_inputs, **inputs[0].kwargs}

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        audio_features: torch.Tensor,
        audio_seqlens: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> NameToTensorList:
        """Run the audio encoder and return embeddings.

        Returns:
            {"audio_embeds": [tensor of shape (audio_tokens, hidden_size)]}
        """
        logger.debug(
            "Running AudioEncoder with audio_features shape=%s",
            audio_features.shape,
        )
        extra_kwargs = {"return_dict": True} if self._encoder_takes_return_dict else {}
        audio_out = self.audio_encoder(
            audio_features,
            feature_lens=audio_seqlens,
            **extra_kwargs,
        )
        audio_embeds = getattr(audio_out, "last_hidden_state", audio_out)

        # Flatten to (num_audio_tokens, hidden_size) if needed
        if audio_embeds.dim() == 3:
            audio_embeds = audio_embeds.squeeze(0)

        return {"audio_embeds": [audio_embeds]}


def prepare_audio(binding, resources):
    del resources
    # HF configuration describes the logical architecture, not an implementation factory.
    config = AutoConfig.from_pretrained(binding.model.local_dir, trust_remote_code=False).thinker_config.audio_config
    patch_hf_fa2_int_maxlen()
    encoder = build_hf_encoder(AudioEncoderWithLengths, config)
    source = binding.model.logical_weights()
    values = source.load(name for name in source.by_name if name.startswith(("thinker.audio_tower.",)))
    missing, _ = encoder.load_state_dict({
        name.removeprefix("thinker.audio_tower."): value.to(binding.device)
        for name, value in values.items()
    }, assign=True, strict=False)
    if missing:
        raise KeyError(f"Missing audio checkpoint keys: {missing}")
    # Match the native EngineManager's final parameter and buffer placement.
    encoder.to(device=binding.device, dtype=binding.dtype)
    return AudioProgram(encoder.eval()).eval()
