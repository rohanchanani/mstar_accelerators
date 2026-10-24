"""Acceleration-owned Talker physical input, computation and capture recipe.

The program contains immutable weights and per-invocation scratch. Request
interpretation, stopping, resource admission and commit stay with MStar.
"""
from __future__ import annotations

from typing import Any
import torch
from torch import nn

from mstar.communication.tensors import NameToTensorList
from mstar.engine.accelerator_graph_config import (
    AcceleratorGraphConfig, BatchedAcceleratorGraphConfig, PackedAcceleratorGraphConfig,
)
from mstar.engine.resources.sampler.resource import SamplerResource
from mstar.model.qwen3_omni.config import CODE_PRED_SAMPLER, TALKER_SAMPLER, Qwen3OmniModelConfig
from mstar.model.qwen3_omni.logical import LAST_PREFILL_TOKENS_PER_REQUEST, codec_prefix_ids
from mstar.model.submodule_base import ARNodeInputs, BatchedModelOutput, ModelInputsFromEngine, NodeInputs


class TalkerProgram(nn.Module):
    MAX_BATCH_SIZE = 32
    TALKER_PREFILL_TOKEN_BUCKETS = [128, 256, 512, 1024]
    TALKER_PREFILL_CAPTURE_BATCH_SIZES = [1, 2, 4]
    TALKER_LAST_PREFILL_TOKENS_PER_REQ = LAST_PREFILL_TOKENS_PER_REQUEST
    TALKER_LAST_PREFILL_CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16, 32]

    def __init__(self, talker_model, code_predictor, raw_conditioning, config: Qwen3OmniModelConfig):
        super().__init__()
        self.model = talker_model
        self.code_predictor = code_predictor
        self.talker_code_emb = talker_model.model.codec_embedding
        self.config = config
        self.cp_cfg = config.code_predictor
        self.num_codes = self.cp_cfg.num_code_groups
        self.register_buffer("_tts_raw_embeddings", None, persistent=False)
        self._suppress_mask = None
        self._cp_kv_cache = None
        self._init_tts_raw_embeddings(raw_conditioning)

    def get_device(self):
        return next(self.parameters()).device

    def bind_resources(self, resources):
        for module in self.modules():
            if module is not self and (bind := getattr(module, "bind_resources", None)) is not None:
                bind(resources)

    def _get_last_prefill_codec_ids(self, speaker):
        return codec_prefix_ids(self.config, speaker, self.get_device())

    @torch.no_grad()
    def _init_tts_raw_embeddings(self, raw: torch.Tensor) -> None:
        if tuple(raw.shape) != (3, self.config.thinker_hidden_size):
            raise ValueError("TTS embeddings must contain pad, bos and eos rows")
        dtype = self.model.text_projection.linear_fc1.weight.dtype
        self._tts_raw_embeddings = raw.to(device=next(self.model.parameters()).device, dtype=dtype)
        # Preserve the donor implementation's single-row projection arithmetic.
        for i, name in enumerate(("pad", "bos", "eos")):
            projected = self.model.text_projection(self._tts_raw_embeddings[i:i + 1]).squeeze(0)
            setattr(self, f"_tts_{name}_embed_cached", projected)


    def _get_cp_kv_cache(self):
        # TODO: make this a resource
        if self._cp_kv_cache is None:
            self._cp_kv_cache = torch.zeros((
                    self.cp_cfg.num_hidden_layers,
                    self.MAX_BATCH_SIZE, 2, self.num_codes,
                    self.cp_cfg.num_key_value_heads,
                    self.cp_cfg.head_dim
                ), dtype=self.talker_code_emb.weight.dtype,
                device=self.get_device(),
            )
        return self._cp_kv_cache


    def _get_suppress_mask(self) -> torch.Tensor:
        """Return the bool mask of layer-0 logits to set to -inf.

        Matches HF's ``talker_supppressed_tokens`` list: suppress the top
        1024 IDs of the Talker vocab (the "special token" region) EXCEPT
        ``codec_eos_token_id``, which is the valid stop signal.
        """
        if  self._suppress_mask is None:
            device = next(self.model.parameters()).device
            vocab_size = self.config.talker_text.vocab_size
            mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
            start = vocab_size - 1024
            start = max(start, 0)
            mask[start:vocab_size] = True
            # Do not suppress codec_eos (the valid stop signal).
            eos = self.config.talker.codec_eos_token_id
            if 0 <= eos < vocab_size:
                mask[eos] = False
            self._suppress_mask = mask
        return self._suppress_mask


    def _get_talker_embeds(
        self, selected_hidden: torch.Tensor, multimodal_mask: torch.Tensor,
    ):
        # ``selected_hidden`` is layer-0 embed for text positions and
        # layer-N hidden for multimodal positions (the Thinker's
        # postprocess already did that selection and dropped positions
        # the Talker won't consume). ``multimodal_mask`` is 1D, aligned
        # to the kept tokens, and tells us which projection to apply.
        text_mask = ~multimodal_mask
        projected = torch.empty(
            selected_hidden.shape[0], self.config.talker_hidden_size,
            device=selected_hidden.device, dtype=selected_hidden.dtype,
        )
        if text_mask.any():
            projected[text_mask] = self.model.text_projection(
                selected_hidden[text_mask]
            )
        if multimodal_mask.any():
            projected[multimodal_mask] = self.model.hidden_projection(
                selected_hidden[multimodal_mask]
            )
        return projected


    def _get_last_prefill_talker_hidden(
        self, thinker_hidden: torch.Tensor
    ):
        # Six-token assistant prefix: pad*4, bos, first Thinker token.
        # Special values preserve the startup single-row projections.
        pad = self._tts_pad_embed_cached.expand(4, -1)  # 4 pad tokens
        bos_text = self._tts_bos_embed_cached.unsqueeze(0) # 1 bos token

        return torch.cat([
            pad,          # pad * 4   (4 tokens)
            bos_text,     # bos       (1 token)
            self.model.text_projection(thinker_hidden), #  (1 token)
        ], dim=0)  # (6, talker_hidden)


    def _materialize_input_embeds(self, graph_walk: str, inp: ARNodeInputs) -> torch.Tensor:
        raw = inp.tensor_inputs["thinker_states"].to(self.model.text_projection.linear_fc1.weight.dtype)
        if graph_walk == "talker_prefill":
            return self._get_talker_embeds(raw, inp.tensor_inputs["multimodal_mask"])
        if graph_walk == "talker_last_prefill":
            return self.talker_code_emb(inp.tensor_inputs["codec_ids"]) + self._get_last_prefill_talker_hidden(raw[-1:])
        dtype = self.model.text_projection.linear_fc1.weight.dtype
        kind = inp.kwargs.get("conditioning_kind", "text")
        if kind == "text":
            text_hidden = self.model.text_projection(raw.to(dtype))
        else:
            text_hidden = getattr(self, f"_tts_{kind}_embed_cached")
        # The incoming codec sum may be a shared graph output. Repeated input
        # preparation must not add conditioning into that source in place.
        return inp.input_embeds.to(dtype) + text_hidden


    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[ARNodeInputs],
    ) -> dict[str, torch.Tensor | Any]:
        seq_lens = [
            inp.input_seq_len for inp in inputs
        ]
        input_embeds = torch.cat([
            self._materialize_input_embeds(graph_walk, inp) for inp in inputs
        ], dim=0)
        device = self.get_device()

        extra_args = {}
        # codec scratch is decode-only; talker_prefill runs the LLM backbone
        # alone and interning these would clash (codec_emb_sum's hidden dim
        # aliases the token-bucket seq dim).
        if graph_walk != "talker_prefill":
            extra_args["suppress_mask"] = self._get_suppress_mask()
            extra_args["all_codes"] = torch.zeros(
                (len(inputs), self.num_codes), device=device, dtype=torch.long
            )
            extra_args["codec_emb_sum"] = torch.zeros(
                (len(inputs), self.cp_cfg.hidden_size), device=device
            )
            extra_args["pos_buf"] = torch.zeros(
                (len(inputs), 1), device=device, dtype=torch.long
            )
        if graph_walk == "talker_last_prefill":
            extra_args["last_token_indices"] = (
                torch.tensor(seq_lens, device=device, dtype=torch.long).cumsum(0) - 1
            )
        return {
            "input_embeds": input_embeds,
            **extra_args
        }


    def _forward_prefill(
        self, input_embeds: torch.Tensor,
    ):
        self.model(
            input_embeds=input_embeds, label="main",
        )
        return {}


    def _forward_decode_like(
        self, request_ids: list[str],
        talker_sampler: SamplerResource,
        code_pred_sampler: SamplerResource,
        suppress_mask: torch.Tensor,
        all_codes: torch.Tensor,
        codec_emb_sum: torch.Tensor,
        pos_buf: torch.Tensor,
        is_batched_decode: bool,
        input_embeds: torch.Tensor,
        last_token_indices: torch.Tensor | None = None,
        **kwargs
    ):
        """
        Run the Talker and depth predictor for walks that sample codec tokens.

        ``last_token_indices`` (when provided) is used to ``index_select`` the
        per-request last hidden out of a packed multi-token-per-request hidden
        — the batched ``talker_last_prefill`` path computes it in ``preprocess``
        as ``cumsum(seq_lens) - 1`` and passes it through here so the
        codec_head only sees one hidden per request. Mutually exclusive with
        the batched-decode (``hidden`` is already ``(bs, hidden)``) and
        non-batched (``hidden[-1:, :]``) branches.
        """
        hidden = self.model(
            input_embeds=input_embeds, label="main",
        )
        if last_token_indices is not None:
            last_hidden = hidden.index_select(0, last_token_indices)
        elif not is_batched_decode:
            last_hidden = hidden[-1:, :]
        else:
            last_hidden = hidden
        logits = self.model.codec_head(last_hidden)
        logits = logits.masked_fill(suppress_mask.unsqueeze(0), float("-inf"))
        # Apply the repetition penalty to the Talker's layer-0 codes (the code
        # predictor depth loop below samples through its own aux config and
        # stays penalty-free). The penalty is baked into the captured graph.
        layer0_codes = talker_sampler.sample(
            request_ids, logits
        )

        # code predictor section
        embed = self.talker_code_emb(layer0_codes)  # [bs, hidden]
        codec_emb_sum.add_(embed)
        all_codes[:, 0] = layer0_codes
        bs = all_codes.shape[0]

        kv_cache = self._get_cp_kv_cache()[:, :bs]

        # forward over [last_hidden] to update kv cache with the Talker's final hidden
        # state as context for the code prediction. This returns nothing because the
        # layer 0 code is already provided by the talker LLM
        cp = self.code_predictor
        codec_embedding = cp.model.codec_embedding
        cp.forward_depth_unrolled(
            last_hidden.unsqueeze(1), pos_buf, kv_cache, cache_pos=0,
        )
        pos_buf += 1

        for group_idx in range(1, self.num_codes):
            hidden = cp.forward_depth_unrolled(
                embed.unsqueeze(1), pos_buf, kv_cache, cache_pos=group_idx,
            ).squeeze(1)
            pos_buf += 1

            logits = torch.matmul(
                hidden, cp.lm_head_weight[group_idx - 1].t()
            )

            # Params come from the "code_predictor" aux sampler's static
            # buffers, so per-request values survive graph replay.
            tokens = code_pred_sampler.sample(
                request_ids, logits,
            )
            all_codes[:, group_idx] = tokens
            embed = codec_embedding[group_idx - 1](tokens)
            codec_emb_sum.add_(embed)

        return {
            "codec_embeds": [codec_emb_sum],
            "codec_tokens": [all_codes],
            "tokens": [layer0_codes]
        }


    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> NameToTensorList | BatchedModelOutput:
        if graph_walk == "talker_prefill":
            return self._forward_prefill(
                input_embeds=input_embeds
            )
        output = self._forward_decode_like(
            request_ids=engine_inputs.request_ids,
            talker_sampler=engine_inputs.resources[TALKER_SAMPLER],
            code_pred_sampler=engine_inputs.resources[CODE_PRED_SAMPLER],
            input_embeds=input_embeds,
            is_batched_decode=(graph_walk == "talker_decode"),
            **kwargs
        )
        return BatchedModelOutput(row_outputs={name: value[0] for name, value in output.items()})


    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_embeds: torch.Tensor | None = None,
        **kwargs,
    ):
        """Batched Talker forward shared between ``talker_decode``, ``talker_prefill``, and ``talker_last_prefill``.

        Decode path (1 token per request, ``hidden`` shape ``(bs, hidden)``):
          Runs the full LLM + codec_head + suppress_mask via _forward_decode_like,
          samples inline through the injected samplers, and emits batch rows
          for the logical codec_embeds, codec_tokens and tokens fields.

        Prefill path (``talker_prefill``; multi-token-per-request, ``hidden``
        shape ``(total_tokens, hidden)``): Runs only the LLM backbone, and
        populates the KV cache.

        Last-prefill path (``talker_last_prefill``; fixed 6 tokens per request,
        ``hidden`` shape ``(bs * 6, hidden)``)
        """
        assert graph_walk in (
            "talker_decode", "talker_prefill", "talker_last_prefill",
        )

        if graph_walk == "talker_prefill":
            hidden = self.model(
                input_embeds=input_embeds,
                label="main",
            )
            return {
                "__batched_talker_prefill_hidden__": hidden,
            }

        last_token_indices = kwargs.pop("last_token_indices", None)
        if graph_walk == "talker_last_prefill":
            assert last_token_indices is not None, (
                "talker_last_prefill forward_batched requires "
                "last_token_indices from preprocess; got None."
            )

        fwd_out = self._forward_decode_like(
            request_ids=engine_inputs.request_ids,
            talker_sampler=engine_inputs.resources[TALKER_SAMPLER],
            code_pred_sampler=engine_inputs.resources[CODE_PRED_SAMPLER],
            is_batched_decode=True,
            last_token_indices=last_token_indices,
            input_embeds=input_embeds,
            **kwargs
        )

        return BatchedModelOutput(row_outputs={name: value[0] for name, value in fwd_out.items()})


    def can_batch(self, batch: Any, model_inputs: list[NodeInputs]) -> bool:
        return batch.graph_walk in [
            "talker_decode", "talker_last_prefill"
        ] and len(model_inputs) <= self.MAX_BATCH_SIZE


    def max_batch_size(self, graph_walk):
        return self.MAX_BATCH_SIZE


    def get_accelerator_graph_configs(
        self, device: torch.device, tp_world_size: int = 1,
    ) -> list[AcceleratorGraphConfig]:
        dtype = self.model.text_projection.linear_fc1.weight.dtype
        return [
            BatchedAcceleratorGraphConfig(
                capture_graph_walk="talker_decode",
                single_request_inputs=ARNodeInputs(
                    input_embeds=torch.zeros(
                        (1, self.config.talker_hidden_size),
                        device=device, dtype=torch.float32,
                    ),
                    tensor_inputs={"thinker_states": torch.zeros(
                        (1, self.config.thinker_hidden_size), device=device, dtype=dtype,
                    )},
                    input_seq_len=1,
                ),
                capture_batch_sizes=[1, 2, 4, 8, 16, 32],
                compile=True
            ),
            PackedAcceleratorGraphConfig(
                capture_graph_walk="talker_prefill",
                replay_graph_walks=["talker_prefill"],
                make_node_input=lambda n: ARNodeInputs(
                    tensor_inputs={
                        "thinker_states": torch.zeros(
                            (n, self.config.thinker_hidden_size), device=device, dtype=dtype,
                        ),
                        "multimodal_mask": torch.zeros(n, device=device, dtype=torch.bool),
                    },
                    input_seq_len=n,
                ),
                capture_token_lengths=self.TALKER_PREFILL_TOKEN_BUCKETS,
                capture_batch_sizes=self.TALKER_PREFILL_CAPTURE_BATCH_SIZES,
                compile=True
            ),
            BatchedAcceleratorGraphConfig(
                capture_graph_walk="talker_last_prefill",
                single_request_inputs=ARNodeInputs(
                    tensor_inputs={
                        "codec_ids": self._get_last_prefill_codec_ids("Ethan"),
                        "thinker_states": torch.cat([
                            self._tts_raw_embeddings[0:1].expand(4, -1),
                            self._tts_raw_embeddings[1:2],
                            torch.zeros((1, self.config.thinker_hidden_size), device=device, dtype=dtype),
                        ]),
                    },
                    input_seq_len=self.TALKER_LAST_PREFILL_TOKENS_PER_REQ,
                ),
                capture_batch_sizes=self.TALKER_LAST_PREFILL_CAPTURE_BATCH_SIZES,
                compile=True
            ),
        ]


    def unpack_packed_outputs(self, static_output, request_ids, real_seq_lens,
                              inputs, per_request_info):
        # Prefill produces state effects, with no logical output tensor.
        return {}
