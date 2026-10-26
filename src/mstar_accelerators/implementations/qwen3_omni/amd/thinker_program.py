"""Physical Thinker program and capture recipes."""
from __future__ import annotations
import functools
import logging
from typing import Any
import torch
from torch import nn
from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.accelerator_graph_config import AcceleratorGraphConfig, BatchedAcceleratorGraphConfig, PackedAcceleratorGraphConfig
from mstar.engine.engine import ExecutingBatch
from mstar.engine.resources.attn.flashinfer import FlashInferManager
from mstar.engine.resources.sampler.resource import SamplerResource
from mstar.model.qwen3_omni.components.rope import compute_3d_cos_sin, compute_rope_freqs
from mstar.model.qwen3_omni.config import THINKER_ATTN, THINKER_MIXED, THINKER_SAMPLER
from mstar.model.submodule_base import ARNodeInputs, ModelInputsFromEngine, NodeInputs

logger = logging.getLogger(__name__)

class ThinkerProgram(nn.Module):
    MROPE_SECTION = [24, 20, 20]
    PREFILL_TOKEN_BUCKETS = [128, 256, 512, 1024, 2048]
    PREFILL_CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16, 32]
    PREFILL_VISION_TOKEN_BUCKETS = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    PREFILL_VISION_CAPTURE_BATCH_SIZES = [1, 4, 16, 32]
    DEEPSTACK_KEY = "deepstack"

    def __init__(self, model, config):
        super().__init__()
        self.model, self.config = model, config
        self._inv_freq = self._decode_thinker_mask = None
        self._audio_bos_embed = self._audio_eos_embed = None
        self._vision_bos_embed = self._vision_eos_embed = None

    def get_device(self):
        return next(self.parameters()).device

    def bind_resources(self, resources):
        for module in self.modules():
            if module is not self and (bind := getattr(module, "bind_resources", None)) is not None:
                bind(resources)

    def prepare_rows(self, inputs):
        from dataclasses import replace
        return tuple(replace(inp, input_embeds=self._embed_input(inp),
            tensor_inputs={key: value for key, value in inp.tensor_inputs.items()
                           if key != "embedding_kind" and not key.startswith("deepstack_")})
            for inp in inputs)

    def _embed_input(self, inp):
        kind = inp.tensor_inputs.get("embedding_kind")
        if kind == "text":
            return self.model.model.embed_tokens(inp.input_ids)
        if kind == "audio":
            return self._wrap_audio_input(inp.input_embeds)
        if kind == "vision":
            return self._wrap_vision_input(inp.input_embeds)
        # Capture templates already supply physical embeddings.
        return inp.input_embeds

    def _get_inv_freq(self, device: torch.device) -> torch.Tensor:
        """Lazy-initialize and cache inverse frequencies."""
        if self._inv_freq is None or self._inv_freq.device != device:
            self._inv_freq = compute_rope_freqs(
                self.config.thinker_head_dim,
                rope_theta=self.config.thinker_text.rope_theta,
                device=device,
            )
        return self._inv_freq


    def _wrap_audio_input(self, audio_embeds: torch.Tensor):
        # Wrap the audio span in ``<|audio_bos|>`` / ``<|audio_eos|>``
        # sentinel token embeddings so the Thinker sees the same
        # prompt layout the HF processor produces.
        device = self.get_device()
        if self._audio_bos_embed is None or self._audio_eos_embed is None:
            audio_start_id = self.config.thinker.audio_start_token_id
            audio_end_id = self.config.thinker.audio_end_token_id
            start_tok = torch.tensor(
                [audio_start_id], dtype=torch.long, device=device
            )
            end_tok = torch.tensor(
                [audio_end_id], dtype=torch.long, device=device
            )
            self._audio_bos_embed = self.model.model.embed_tokens(start_tok)
            self._audio_eos_embed = self.model.model.embed_tokens(end_tok)

        return torch.cat([
            self._audio_bos_embed,
            audio_embeds,
            self._audio_eos_embed
        ], dim=0)


    def _wrap_vision_input(self, vision_embeds: torch.Tensor):
        # Wrap the vision span in ``<|vision_bos|>`` / ``<|vision_eos|>``
        # sentinel token embeddings.
        if self._vision_bos_embed is None or self._vision_eos_embed is None:
            device = vision_embeds.device
            vision_start_id = self.config.thinker.vision_start_token_id
            vision_end_id = self.config.thinker.vision_end_token_id
            start_tok = torch.tensor(
                [vision_start_id], dtype=torch.long, device=device
            )
            end_tok = torch.tensor(
                [vision_end_id], dtype=torch.long, device=device
            )
            self._vision_bos_embed = self.model.model.embed_tokens(start_tok)
            self._vision_eos_embed = self.model.model.embed_tokens(end_tok)

        return torch.cat([
            self._vision_bos_embed,
            vision_embeds,
            self._vision_eos_embed
        ], dim=0)


    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[ARNodeInputs],
    ) -> dict[str, torch.Tensor | Any]: # input name to tensor
        device = self.get_device()
        # Concatenate across requests
        input_embeds = torch.cat([
            inp.input_embeds for inp in inputs
        ], dim=0)
        position_ids_3d = torch.cat([
            inp.custom_pos_ids for inp in inputs
        ], dim=1)  # (3, total_tokens)
        seq_lens = [
            inp.input_seq_len for inp in inputs
        ]

        # Compute cos/sin for 3D MRoPE.  Returned as separate tensor keys
        # (not a tuple) so the CUDA graph runner can detect them as static
        # inputs and copy them into the captured buffers at replay.
        inv_freq = self._get_inv_freq(device)
        cos_3d, sin_3d = compute_3d_cos_sin(
            position_ids_3d, inv_freq,
            mrope_section=self.MROPE_SECTION,
            target_dtype=input_embeds.dtype,
        )

        extra_inputs = {}
        if graph_walk == "prefill_vision" or any("deepstack" in inp.tensor_inputs for inp in inputs):
            # Q3(b): emit deepstack as separate keys ``deepstack_<i>`` so each
            # tensor gets its own static buffer in the captured config (the
            # runner's static-buffer interning is per-key and tensor-typed;
            # passing a list-of-tensors under one key would not get interned
            # and addresses captured into the graph would be stale at replay).
            # ``forward_batched``/``forward`` reassemble the list from kwargs.
            extra_inputs = self._deepstack_inputs(inputs, input_embeds.dtype, device)

        return {
            "input_embeds": input_embeds,
            "cos_3d": cos_3d,
            "sin_3d": sin_3d,
            "mrope_section": self.MROPE_SECTION,
            "seq_lens": seq_lens,
            "masks_for_talker": {
                rid: inp.tensor_inputs.get("masks_for_talker") \
                    for (rid, inp) in zip(engine_inputs.request_ids, inputs, strict=True)
            },
            **extra_inputs
        }


    def _deepstack_inputs(
        self, inputs: list[ARNodeInputs], dtype: torch.dtype, device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """Each deepstack layer packed over the step's rows; a row with no
        vision features (a text, audio or decode row beside a vision one)
        contributes zeros, which the splice adds as nothing."""
        num_deepstack = len(self.config.vision.deepstack_visual_indexes)
        layers: list[list[torch.Tensor]] = [[] for _ in range(num_deepstack)]
        for inp in inputs:
            rows = inp.tensor_inputs.get("deepstack")
            if rows is None or (isinstance(rows, torch.Tensor) and rows.numel() == 0):
                zeros = torch.zeros(
                    (inp.input_seq_len, self.config.thinker_hidden_size), dtype=dtype, device=device,
                )
                rows = [zeros] * num_deepstack
            assert len(rows) == num_deepstack, (
                f"deepstack list length ({len(rows)}) does not match "
                f"vision.deepstack_visual_indexes ({num_deepstack})."
            )
            for layer, t in zip(layers, rows, strict=True):
                layer.append(t)
        return {
            f"deepstack_{i}": layer[0] if len(layer) == 1 else torch.cat(layer)
            for i, layer in enumerate(layers)
        }


    def _collect_deepstack_kwargs(
        self, kwargs: dict[str, Any]
    ) -> list[torch.Tensor] | None:
        """Reassemble the deepstack list from per-layer ``deepstack_<i>`` keys.

        ``preprocess`` emits one key per ``vision.deepstack_visual_indexes``
        entry so each layer's visual feature gets its own static buffer in
        the captured prefill_vision config. Both the eager ``forward`` and
        the captured ``forward_batched`` call this helper to put the list
        back together before invoking ``Qwen3OmniThinkerModel.forward``.

        Returns None when no ``deepstack_*`` keys are present (e.g. for
        prefill_text / prefill_audio / thinker_decode), so the inner model
        receives ``deepstack_visual_embeds=None`` and skips the deepstack
        splice entirely.
        """
        num_deepstack = len(self.config.vision.deepstack_visual_indexes)
        if not any(f"deepstack_{i}" in kwargs for i in range(num_deepstack)):
            return None
        out: list[torch.Tensor] = []
        for i in range(num_deepstack):
            t = kwargs.get(f"deepstack_{i}")
            if t is None:
                return None
            out.append(t)
        return out


    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_embeds: torch.Tensor | None = None,
        cos_3d: torch.Tensor | None = None,
        sin_3d: torch.Tensor | None = None,
        mrope_section: list[int] | None = None,
        masks_for_talker: dict[str, torch.Tensor] | None = None,
        **kwargs,
    ) -> NameToTensorList:
        """Run Thinker transformer, produce logits (decode) and thinker_states.

        ``thinker_states`` is only emitted when audio output is requested
        (checked via ``request_info.step_metadata["audio_output"]``). This
        saves cross-partition bandwidth for text-only requests. Defaults to
        ``True`` for backwards compatibility with callers that do not set
        the flag (e.g. unit tests).

        ``deepstack`` (used by prefill_vision) is reassembled from per-layer
        ``deepstack_<i>`` kwargs — see ``_collect_deepstack_kwargs``.
        """
        deepstack = self._collect_deepstack_kwargs(kwargs)
        request_info: CurrentForwardPassInfo = engine_inputs.single_request_info
        audio_output = request_info.step_metadata.get(
            "audio_output", True,
        )

        cos_sin_3d = (cos_3d, sin_3d) if cos_3d is not None else None

        hidden, layer_0_embed, layer_n_hidden = self.model(
            input_embeds=input_embeds,
            cos_sin_3d=cos_sin_3d,
            mrope_section=mrope_section,
            deepstack_visual_embeds=deepstack,
            label="main",
        )

        result: NameToTensorList = {}

        # Decode: produce logits for text token sampling
        if graph_walk == "thinker_decode" or request_info.step_metadata.get("is_last_prefill", False):
            logits = self.model.lm_head(hidden[-1:, :])
            result["new_token"] = [engine_inputs.resources[THINKER_SAMPLER].sample(
                engine_inputs.request_ids, logits=logits
            )]

        # Pack thinker_states for Talker conditioning ONLY when audio output
        # is requested.  For text-only requests we skip this entirely to
        # avoid sending hidden states the Talker will never consume.
        if audio_output:
            # Concatenate layer-0 embeddings and layer-N hidden states along
            # last dim -> (tokens, 2 * hidden_size)
            if layer_n_hidden is not None:
                thinker_states = torch.cat(
                    [layer_0_embed, layer_n_hidden], dim=-1,
                )
            else:
                # Fallback: use layer_0_embed doubled (shouldn't happen in
                # practice)
                thinker_states = torch.cat(
                    [layer_0_embed, layer_0_embed], dim=-1,
                )
            result["thinker_states"] = [thinker_states]
            result["thinker_mask"] = [next(iter(masks_for_talker.values()))] \
                if masks_for_talker else []
        return result


    def can_batch(self, batch: ExecutingBatch, model_inputs: list[NodeInputs]) -> bool:
        return len(model_inputs) > 1


    def get_accelerator_graph_configs(
        self, device: torch.device, tp_world_size: int = 1,
    ) -> list[AcceleratorGraphConfig]:
        vision_input = functools.partial(self._vision_capture_input, device)
        return [
            BatchedAcceleratorGraphConfig(
                capture_graph_walk="thinker_decode",
                single_request_inputs=ARNodeInputs(
                    input_seq_len=1,
                    input_embeds=torch.zeros(
                        (1, self.config.thinker_hidden_size),
                        device=device, dtype=torch.bfloat16
                    ),
                    custom_pos_ids=torch.tensor(
                        [[0], [0], [0]],
                        dtype=torch.float,
                        device=device,
                    ),
                    tensor_inputs={
                        "masks_for_talker": self._get_decode_thinker_mask(device)
                    }
                ),
                capture_batch_sizes=[1, 2, 4, 8, 16, 32],
            ),
            PackedAcceleratorGraphConfig(
                capture_graph_walk="prefill_text",
                replay_graph_walks=["prefill_text", "prefill_audio", THINKER_MIXED],
                make_node_input=lambda n: ARNodeInputs(
                    input_seq_len=n,
                    input_ids=torch.zeros((n,), dtype=torch.long, device=device),
                    input_embeds=torch.zeros(
                        (n, self.config.thinker_hidden_size),
                        device=device, dtype=torch.bfloat16
                    ),
                    custom_pos_ids=torch.zeros(
                        (3, n),
                        dtype=torch.float,
                        device=device,
                    ),
                    tensor_inputs={
                        "masks_for_talker": torch.zeros(
                            (2, n),
                            dtype=torch.float,
                            device=device,
                        ),
                    }
                ),
                capture_token_lengths=self.PREFILL_TOKEN_BUCKETS,
                capture_batch_sizes=self.PREFILL_CAPTURE_BATCH_SIZES,
            ),
            # any step holding a vision row: the packed forward plus deepstack
            PackedAcceleratorGraphConfig(
                capture_graph_walk="prefill_vision",
                replay_graph_walks=["prefill_vision", THINKER_MIXED],
                additional_key_info=self.DEEPSTACK_KEY,
                capture_token_lengths=self.PREFILL_VISION_TOKEN_BUCKETS,
                capture_batch_sizes=self.PREFILL_VISION_CAPTURE_BATCH_SIZES,
                make_node_input=vision_input,
            ),
        ]


    def cg_key_info(
        self, graph_walk: str,
        per_request_info: dict[str, CurrentForwardPassInfo],
        per_request_input_metadata=None,
        **kwargs,
    ):
        """A step holding a vision row replays the deepstack captures."""
        del per_request_input_metadata, kwargs
        if graph_walk == "prefill_vision" or graph_walk == THINKER_MIXED and any(
            info.graph_walk == "prefill_vision" for info in per_request_info.values()
        ):
            return self.DEEPSTACK_KEY
        return None


    def _vision_capture_input(self, device: torch.device, n: int) -> ARNodeInputs:
        num_deepstack = len(self.config.vision.deepstack_visual_indexes)
        return ARNodeInputs(
            input_seq_len=n,
            input_embeds=torch.zeros(
                (n, self.config.thinker_hidden_size),
                device=device, dtype=torch.bfloat16,
            ),
            custom_pos_ids=torch.zeros(
                (3, n),
                dtype=torch.float,
                device=device,
            ),
            tensor_inputs={
                "masks_for_talker": torch.zeros(
                    (2, n), dtype=torch.float, device=device,
                ),
                # Use a list (matches eager prepare_inputs) — preprocess turns it
                # into per-layer ``deepstack_<i>`` keys.
                "deepstack": [
                    torch.zeros(
                        (n, self.config.thinker_hidden_size),
                        dtype=torch.bfloat16, device=device,
                    )
                    for _ in range(num_deepstack)
                ],
                "mrope_pos_advance": n,
            },
        )


    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_embeds: torch.Tensor | None = None,
        cos_3d: torch.Tensor | None = None,
        sin_3d: torch.Tensor | None = None,
        mrope_section: list[int] | None = None,
        masks_for_talker: dict[str, torch.Tensor] | None = None,
        **kwargs,
    ) -> dict[str, NameToTensorList]:
        """Batched Thinker forward shared between ``thinker_decode`` and the prefill walks.

        Decode path (1 token per request, ``hidden`` shape ``(bs, hidden)``):
        Always packs ``thinker_states`` + ``thinker_mask`` in every per-rid
        output dict so the captured CUDA graph has a static output shape
        regardless of request metadata. Per-rid filtering happens in
        ``unpack_packed_output``.
        """

        is_prefill = graph_walk in (
            "prefill_text", "prefill_audio", "prefill_vision", THINKER_MIXED,
        )

        # prefill_vision: reassemble per-layer deepstack tensors from
        # ``deepstack_<i>`` kwargs into the list shape ``model.forward``
        # expects. None for non-vision walks → model skips the splice.
        deepstack = self._collect_deepstack_kwargs(kwargs)

        cos_sin_3d = (cos_3d, sin_3d) if cos_3d is not None else None
        attn_manager: FlashInferManager = engine_inputs.resources[THINKER_ATTN]
        hidden, layer_0_embed, layer_n_hidden = self.model(
            input_embeds=input_embeds,
            cos_sin_3d=cos_sin_3d,
            mrope_section=mrope_section,
            deepstack_visual_embeds=deepstack,
            label="main",
        )

        request_ids = engine_inputs.request_ids
        sampler: SamplerResource = engine_inputs.resources[THINKER_SAMPLER]

        if is_prefill:
            last_hidden = attn_manager.select_last_hidden(hidden)
            logits = self.model.lm_head(last_hidden)  # (padded_bs, vocab)
            new_tokens = sampler.sample(request_ids, logits)
            if layer_n_hidden is not None:
                thinker_states = torch.cat(
                    [layer_0_embed, layer_n_hidden], dim=-1,
                )  # (total_tokens, 2*hidden)
            else:
                thinker_states = torch.cat(
                    [layer_0_embed, layer_0_embed], dim=-1,
                )

            return {
                "__batched_thinker_states__": thinker_states,
                **{
                    rid: {
                        "new_token": [new_tokens[i:i + 1]]
                    } for i, rid in enumerate(request_ids)
                }
            }

        # thinker_decode (existing behavior)
        logits = self.model.lm_head(hidden)  # (batch, vocab)
        new_tokens = sampler.sample(request_ids, logits)

        if layer_n_hidden is not None:
            thinker_states = torch.cat(
                [layer_0_embed, layer_n_hidden], dim=-1,
            )
        else:
            thinker_states = torch.cat(
                [layer_0_embed, layer_0_embed], dim=-1,
            )

        outputs: dict[str, NameToTensorList] = {}
        for i, rid in enumerate(request_ids):
            req_out: NameToTensorList = {
                "thinker_states": [thinker_states[i : i + 1]],
                "new_token": [new_tokens[i:i+1]]
            }
            if masks_for_talker is not None and rid in masks_for_talker:
                req_out["thinker_mask"] = [masks_for_talker[rid]]
            outputs[rid] = req_out

        return outputs


    def unpack_packed_outputs(
        self,
        static_output: dict,
        request_ids: list[str],
        real_seq_lens: list[int],
        inputs: list[ARNodeInputs],
        per_request_info: dict[str, "CurrentForwardPassInfo"],
    ) -> dict[str, dict[str, list[torch.Tensor]]]:
        """Slice the packed ``__batched_thinker_states__`` per real seq_len.

        Captured forward emits the full ``(total_tokens, 2*hidden)`` packed
        tensor; here we cut it at the real per-request token boundaries and
        reattach the per-request talker masks, which live on the original
        ARNodeInputs (the captured graph never saw them — masks vary in
        shape with text content). Drops per-rid emission for requests with
        ``audio_output=False``, mirroring ``filter_batched_output``'s gating
        for the decode path.
        """
        packed_states = static_output.get("__batched_thinker_states__")
        if packed_states is None:
            return {}

        out: dict[str, dict[str, list[torch.Tensor]]] = {}
        cum = 0
        for i, rid in enumerate(request_ids):
            sl = real_seq_lens[i]
            slice_start, slice_end = cum, cum + sl
            cum = slice_end

            info = per_request_info.get(rid) if per_request_info else None
            if info is not None and not info.step_metadata.get("audio_output", True):
                continue

            # TensorForwardRunner.collect_outputs takes the owned copy before replay reuse.
            ts_slice = packed_states[slice_start:slice_end]
            rid_out: dict[str, list[torch.Tensor]] = {
                "thinker_states": [ts_slice],
            }
            mask = inputs[i].tensor_inputs.get("masks_for_talker")
            if mask is not None:
                rid_out["thinker_mask"] = [mask]
            out[rid] = rid_out
        return out


    def _get_decode_thinker_mask(self, device: torch.device) -> torch.Tensor:
        """Return the constant ``[[0], [1]]`` decode mask (multimodal row =
        0, text-inclusion row = 1), lazily allocated per device."""
        if (
            self._decode_thinker_mask is None
            or self._decode_thinker_mask.device != device
        ):
            self._decode_thinker_mask = torch.tensor(
                [[0], [1]], dtype=torch.bool, device=device,
            )
        return self._decode_thinker_mask


    def filter_batched_output(
        self,
        request_info: CurrentForwardPassInfo,
        rid_output: dict[str, list[torch.Tensor]],
    ) -> dict[str, list[torch.Tensor]]:
        """Drop ``thinker_states`` + ``thinker_mask`` for text-only requests.

        ``forward_batched`` always emits these keys so the captured CUDA
        graph's output shape is static.  Here, outside the captured
        region, we gate them on the real request's ``audio_output`` flag
        so the Talker edge stays unrouted for text-only requests (matches
        the pre-capture eager-mode behaviour).
        """
        if request_info is None:
            return rid_output
        if request_info.step_metadata.get("audio_output", True):
            return rid_output
        return {
            k: v for k, v in rid_output.items()
            if k not in ("thinker_states", "thinker_mask")
        }


