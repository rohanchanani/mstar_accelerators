"""Prepared CPU decoder over runtime-owned KV and sampling plans."""
import ctypes as C
from dataclasses import dataclass
import logging

import torch

from mstar.engine.resources.sampler.cpu import can_use_native_greedy

from mstar.engine.execution import Support
from mstar.engine.resources.sequence import DenseSequenceView, PagedSequenceView, PlainKVEncoding, Replicated
from mstar.engine.resources.kv.config import KVSpec
from mstar.model.whisper.config import ATTN, CROSS_ATTN, CROSS_KV_CACHE, KV_CACHE, POS
from mstar_accelerators import PreparedImplementation
from mstar_accelerators.backends.torch import TorchProgram, WorkspaceCapacity, prepare_torch_programs
from mstar_accelerators.compiler import bind_default_call
from mstar_accelerators.conditions import Constant
from .cpu_ggml import Weight, native_threads
from .cpu_ggml_decoder_abi import Config, StoragePlan, Plane, TokenPlan, TokenCall, checked, size, library
from .cpu_ggml_readout_weights import prepared_weights

logger = logging.getLogger(__name__)


def resource_requests(binding):
    return {spec.resource_key: DenseSequenceView(
                partition=Replicated(), encoding=PlainKVEncoding(torch.float16))
            for spec in binding.resource_specs
            if isinstance(spec, KVSpec) and binding.node_name in spec.nodes}


@dataclass(frozen=True)
class CPUTokenDecoderAvailable:
    readout_format: str = "q8_0"
    def evaluate(self, facts):
        binding = facts.get('binding')
        if binding is None:
            return Support.UNKNOWN
        return Support.YES if library(binding, readout_format=self.readout_format) is not None else Support.NO

    def specialize(self, facts):
        value = self.evaluate(facts)
        return self if value is Support.UNKNOWN else Constant(value)


class Decoder:
    """One physical owner; KV planes remain borrowed from M* resources."""

    def __init__(self, binding, weights, resources, packed, suppressed, shape, *, readout_format="q8_0"):
        self.handle = None
        self.workspace = WorkspaceCapacity(slots=1)
        self.users = 0
        self.planes = self.views = self.logits = None
        self.embedding = self.positions = self.packed = self.suppressed = None
        self.config = binding.submodule.config
        self.library = library(binding, readout_format=readout_format)
        if self.library is None:
            raise ValueError('configured artifact lacks native token decoder capability')
        self._validate_readout(packed, suppressed, shape, readout_format)
        self.packed, self.suppressed, self.rows = packed, suppressed, shape[0]
        self.embedding = checked(weights['decoder.embed_tokens.weight'], (self.rows, self.config.d_model))
        self.positions = checked(weights['decoder.embed_positions.weight'],
                                 (self.config.max_target_positions, self.config.d_model))
        self.views = self._bind_views(resources)
        try:
            self._create_native(binding, weights)
            self._attach_readout(readout_format, shape)
            self._attach_tokens()
            self.logits = torch.empty(self.rows, dtype=torch.float32)
            logger.info("CPU_GGML_DECODER_READY readout=%s borrowed_kv=True logits_bytes=%d",
                        readout_format, size(self.logits))
        except BaseException:
            self.close()
            raise

    def _validate_readout(self, packed, suppressed, shape, readout_format):
        if readout_format not in ("q8_0", "q4_0", "q4_0_kai"):
            raise ValueError("unsupported physical readout format")
        block_bytes = 34 if readout_format == "q8_0" else 18
        rows, width = shape
        if (packed.device.type != 'cpu' or packed.dtype != torch.int32 or not packed.is_contiguous()
                or width != self.config.d_model or rows != self.config.vocab_size
                or size(packed) != rows * (width // 32) * block_bytes or width % 32):
            raise ValueError('graph readout packed projection differs from logical shape')
        for mask in suppressed.values():
            if mask.device.type != 'cpu' or mask.dtype != torch.int64 or not mask.is_contiguous():
                raise ValueError('graph readout suppression storage differs')

    def _dense_views(self, resources):
        self.cache_dtypes, self.cache_capacities = {}, {}
        formats, capacities, views = [], [], {}
        ids = {torch.float32: 1, torch.float16: 2}  # decoder_storage.h ABI v1
        for key in (KV_CACHE, CROSS_KV_CACHE):
            resource = resources[key]
            encoding = resource.config.encoding
            if not isinstance(encoding, PlainKVEncoding) or encoding.dtype not in ids:
                raise ValueError('native storage ABI v1 requires resolved plain F32/F16 encoding')
            view = resource.view(DenseSequenceView(encoding=encoding))
            logical_capacity = (self.config.max_target_positions if key == KV_CACHE
                                else self.config.max_source_positions)
            if (view.config.max_seq_len != logical_capacity or
                    (view.config.num_layers, view.config.num_kv_heads, view.config.head_dim) !=
                    (self.config.decoder_layers, self.config.decoder_attention_heads,
                     self.config.d_model // self.config.decoder_attention_heads)):
                raise ValueError('native storage ABI v1 requires full unpadded replicated logical geometry')
            self.cache_capacities[key] = view.config.max_seq_len
            self.cache_dtypes[key] = encoding.dtype
            formats.append(ids[encoding.dtype])
            capacities.append(view.config.max_seq_len)
            views[key] = view
        self.storage_plan = StoragePlan(*formats, *capacities)
        return views

    def _bind_views(self, resources):
        dense = self._dense_views(resources)
        views, self.planes = {}, {}
        for key, selected in dense.items():
            config = selected.config
            if not config.dense:
                raise ValueError('native token entry needs the existing resolved dense owner')
            # Paged access is compatible with the same full-capacity dense
            # representation; it returns admitted plans, without constructing
            # transient per-layer Torch views or attention mask tensors.
            views[key] = resources[key].view(PagedSequenceView(
                page_size=config.page_size, encoding=selected.encoding))
            self.planes[key] = tuple(checked(views[key].layer(i),
                (config.max_num_pages, 2, config.max_seq_len, config.num_kv_heads, config.head_dim),
                self.cache_dtypes[key]) for i in range(config.num_layers))
        return views

    def _create_native(self, binding, weights):
        c = self.config
        threads = native_threads(binding)
        physical = Config(c.d_model, c.decoder_layers, c.decoder_attention_heads, c.decoder_ffn_dim,
                          c.max_source_positions, c.max_target_positions, threads)
        body = {n: t.detach().to(device='cpu', dtype=torch.float32).contiguous()
                for n, t in weights.items() if n not in ('decoder.embed_tokens.weight',
                    'decoder.proj_out.weight', 'decoder.embed_positions.weight')}
        descriptors = (Weight * len(body))()
        for i, (name, tensor) in enumerate(sorted(body.items())):
            descriptors[i] = Weight(name.encode(), tensor.data_ptr(), size(tensor), tensor.ndim,
                                    (C.c_int64 * 3)(*reversed(tensor.shape)))
        error = C.create_string_buffer(1024)
        self.handle = self.library.cpu_decoder_q8_storage_create(
            C.byref(physical), C.byref(self.storage_plan), descriptors, len(descriptors), error, len(error))
        if not self.handle:
            raise RuntimeError('CPU decoder preparation: ' + error.value.decode())
        cross_threads = binding.model_config.get('runtime_extension', {}).get('cpu_ggml_cross_threads')
        if cross_threads is not None:
            if type(cross_threads) is not int or not 1 <= cross_threads <= 64:
                raise ValueError('cross preparation threads must be an integer in [1, 64]')
            api = self.library.cpu_decoder_set_cross_threads
            api.argtypes = [C.c_void_p, C.c_int, C.c_void_p, C.c_size_t]
            api.restype = C.c_int
            if api(self.handle, cross_threads, error, len(error)):
                raise RuntimeError('cross preparation: ' + error.value.decode())
        capture = binding.model_config.get('runtime_extension', {}).get('cpu_ggml_attention_capture')
        if capture is not None:
            from pathlib import Path
            if not isinstance(capture, str) or not Path(capture).is_absolute():
                raise ValueError('attention capture must be an absolute fresh file prefix')
            paths = [Path(capture + suffix) for suffix in ('.json', '.q.f32', '.k.f32', '.v.f32', '.mask.f32')]
            present = [path.exists() for path in paths]
            if any(present) and not all(present):
                raise ValueError('partial attention capture exists; preserve it and choose a fresh prefix')
            if not all(present):
                api = self.library.cpu_decoder_capture_attention
                api.argtypes = [C.c_void_p, C.c_char_p, C.c_void_p, C.c_size_t]
                api.restype = C.c_int
                if api(self.handle, capture.encode(), error, len(error)):
                    raise RuntimeError('attention capture: ' + error.value.decode())


    def _attach_readout(self, readout_format, shape):
        api, packed = self.library, self.packed
        rows, width = shape
        error = C.create_string_buffer(1024)
        setter = getattr(api, {"q8_0": "cpu_decoder_set_readout_q8", "q4_0": "cpu_decoder_set_readout_q4",
                               "q4_0_kai": "cpu_decoder_set_readout_q4_kai"}[readout_format])
        if setter(self.handle, packed.data_ptr(), size(packed), rows, width, error, len(error)):
            raise RuntimeError('CPU graph readout preparation: ' + error.value.decode())

    def _attach_tokens(self):
        api = self.library
        layers = (Plane * 4)(*(Plane(
            self.planes[KV_CACHE][i].data_ptr(), size(self.planes[KV_CACHE][i]),
            self.planes[CROSS_KV_CACHE][i].data_ptr(), size(self.planes[CROSS_KV_CACHE][i])) for i in range(4)))
        plan = TokenPlan(self.embedding.data_ptr(), size(self.embedding),
            self.positions.data_ptr(), size(self.positions),
            self.views[KV_CACHE].config.max_num_pages, self.views[CROSS_KV_CACHE].config.max_num_pages,
            self.config.d_model ** .5 if self.config.scale_embedding else 1., layers)
        error = C.create_string_buffer(1024)
        if api.cpu_decoder_set_token_inputs(self.handle, C.byref(plan), error, len(error)):
            raise RuntimeError('native token preparation: ' + error.value.decode())

    @staticmethod
    def context(access, use, q):
        plan = access.for_attention(use)
        if len(plan.contexts) != 1 or len(plan.request_ids) != 1 or use.query_lengths != (q,):
            raise ValueError('native token entry needs one admitted query/context row')
        view = plan.contexts[0]
        if len(view.page_idxs) != 1 or view.start != 0:
            raise ValueError('native token entry requires one complete resident owner block')
        return plan, view

    @torch.no_grad()
    def run(self, call):
        if self.handle is None:
            raise RuntimeError('CPU decoder body is closed')
        q = 4 if call.graph_walk == 'prefill' else 1
        ids, = call.inputs.input_ids
        checked(ids, (q,), torch.int64)
        positions = checked(getattr(call.plan_results, POS)['main'], (q,), torch.int64)
        uses = call.plan_results
        self_use, cross_use = getattr(uses, ATTN)['main'], getattr(uses, CROSS_ATTN)['main']
        read, self_view = self.context(self.views[KV_CACHE], self_use, q)
        cross, cross_view = self.context(self.views[CROSS_KV_CACHE], cross_use, q)
        write = self.views[KV_CACHE].for_step(getattr(uses, KV_CACHE)['main'])
        if len(write.views) != 1:
            raise ValueError('native token entry needs one admitted write row')
        writable = write.views[0]
        if (not read.causal or cross.causal or writable.request_id != read.request_ids[0]
                or cross.request_ids != read.request_ids or writable.start != 0
                or writable.page_idxs != self_view.page_idxs or writable.length != self_view.length
                or writable.to_compute != q):
            raise ValueError('admitted decoder write/read identities or masks differ')
        encoder = None
        if q == 4:
            encoder, = call.inputs.encoder_states
            encoder = checked(encoder, (self.config.max_source_positions, self.config.d_model))
            write_cross = self.views[CROSS_KV_CACHE].for_step(getattr(uses, CROSS_KV_CACHE)['main'])
            if len(write_cross.views) != 1:
                raise ValueError('native token prefill needs one cross write row')
            cw = write_cross.views[0]
            if (cw.request_id != cross.request_ids[0] or cw.start != 0 or cw.page_idxs != cross_view.page_idxs
                    or cw.length != cross_view.length or cw.length != self.config.max_source_positions
                    or cw.to_compute != self.config.max_source_positions):
                raise ValueError('admitted cross write/read identities differ')
        token = checked(call.outputs.tokens, (1,), torch.int64)
        mask = self.suppressed[call.graph_walk]
        actual = TokenCall(q, self_view.page_idxs[0], cross_view.page_idxs[0], self_view.length,
            cross_view.length, writable.length - writable.to_compute,
            ids.data_ptr(), size(ids), positions.data_ptr(), size(positions),
            encoder.data_ptr() if encoder is not None else None, size(encoder) if encoder is not None else 0,
            self.logits.data_ptr(), size(self.logits), token.data_ptr(), size(token),
            mask.data_ptr(), mask.numel(), size(mask))
        error = C.create_string_buffer(1024)
        if self.library.cpu_decoder_token_run(self.handle, C.byref(actual), error, len(error)):
            raise RuntimeError('native token execution: ' + error.value.decode())
        return self.logits, token

    def close(self):
        self.workspace.close()
        if self.handle is not None:
            error = C.create_string_buffer(1024)
            if self.library.cpu_decoder_close(self.handle, error, len(error)):
                raise RuntimeError('CPU decoder close: ' + error.value.decode())
            self.handle = None
        # Drop borrowed backing only after native destruction succeeds.
        self.planes = self.views = self.logits = None
        self.embedding = self.positions = self.packed = self.suppressed = None


def prepare_body(binding, weights, resources, packed, suppressed, shape, *, readout_format="q8_0"):
    key = (prepare_body, tuple(resources[name].config.encoding for name in (KV_CACHE, CROSS_KV_CACHE)))
    if readout_format != "q8_0":
        key += (readout_format,)
    if key not in binding.prepared:
        binding.prepared[key] = Decoder(binding, weights, resources, packed, suppressed, shape,
                                          readout_format=readout_format)
    owner = binding.prepared[key]
    owner.users += 1
    released = False

    def close():
        nonlocal released
        if released:
            return
        if owner.users == 1:
            owner.close()
            binding.prepared.pop(key)
        owner.users -= 1
        released = True
    return owner, close

def prepare(binding, resources, *, sample_tokens=None, readout_format="q8_0"):
    from .decoder_call import Call, Inputs, Plans
    defaults = bind_default_call(binding, resources, input_type=Inputs, call_type=Call, plan_type=Plans)
    source_body, weights, packed, suppressed, shape = prepared_weights(binding, defaults.contract.weight_prefix, readout_format=readout_format)
    body_weights = source_body | {'decoder.embed_tokens.weight': weights['decoder.embed_tokens.weight']}
    owner, close_body = prepare_body(binding, body_weights, resources, packed, suppressed, shape, readout_format=readout_format)

    def run(call):
        logits, token = owner.run(call)
        plan = call.plan_results.sampler
        return {'tokens': token if sample_tokens is None or can_use_native_greedy(plan)
                else sample_tokens(logits, plan)}

    try:
        program = prepare_torch_programs(binding, resources,
            {walk: TorchProgram(run) for walk in ('prefill', 'decode')},
            call_binding=defaults, workspace_capacity=owner.workspace)
    except BaseException:
        close_body()
        raise

    def close():
        program.close()
        close_body()
    return PreparedImplementation(program.materialize, program.primitive, program.acquire, close)
