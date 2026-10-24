"""Drive generated implementations through M*'s real host engine in tests."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.distributed.communication import WorkerParallelGroups
from mstar.engine.engine import ExecutingBatch
from mstar.engine.resources.sampler.config import SamplingReqConfig
from mstar.engine.resources.step import StepContext
from mstar.model.whisper.config import CROSS_KV_CACHE, KV_CACHE, SAMPLER
from mstar.worker.engine_manager import EngineManager
from mstar_accelerators.registry import Registry

from .cpu_runtime import host_entries
from .provider import HostGraphProvider, Submission


@dataclass
class Executed:
    batch: ExecutingBatch
    output: dict
    observation: Submission


class EngineSession:
    def __init__(self, model, *, requests=(0,), source=None):
        self.model = model
        self.provider = HostGraphProvider()
        self.registry = Registry(host_entries(provider=self.provider) if source is None else
                                 host_entries(source, provider=self.provider))
        settings = {"autocast_dtype": "float32", "resources": {
            KV_CACHE: {"max_bytes": 16 * 1024**2, "prefix_cache": False},
            CROSS_KV_CACHE: {"max_bytes": 56 * 1024**2, "prefix_cache": False},
        }}
        self.manager = EngineManager()
        self.manager.load(
            {"audio_encoder", "decoder"}, torch.device("cpu"), settings,
            WorkerParallelGroups(1, 0), None, model, None, runtime_extension=self.registry,
        )
        self.request_ids = tuple(requests)
        for rid in requests:
            self.manager.add_request(rid, {SAMPLER: SamplingReqConfig(temperature=0)})
        self.executions = []

    @property
    def resources(self):
        return self.manager.engine._resources

    @torch.no_grad()
    def run(self, rid, node, walk, inputs, *, iteration=0):
        info = CurrentForwardPassInfo(
            f"request-{rid}", walk, 0, 0, self.model.get_max_output_tokens(), rid_handle=rid,
            resource_configs={SAMPLER: SamplingReqConfig(temperature=0)},
        )
        info.dynamic_loop_iter_counts["decode_loop"] = iteration
        batch = ExecutingBatch(node, {rid: info}, StepContext([rid], walk, 0, False),
                               per_request_input_tensors={rid: inputs})
        self.manager.engine.prepare_inputs(batch)
        output = self.manager.engine.exec_and_postprocess(batch)
        assert batch.admit_error is None
        assert batch.commit_done.is_set()
        observation = self.provider.last_submission
        result = Executed(batch, output.per_rid_outputs[rid], observation)
        self.executions.append((rid, node, walk))
        return result

    def close(self):
        for rid in self.request_ids:
            self.manager.remove_request(rid)
        for key in (KV_CACHE, CROSS_KV_CACHE):
            self.resources[key].assert_pages_conserved()
        self.manager.shutdown()
        self.provider.last_submission = None
        assert all(program.active == 0 and program.closed for program in self.provider.prepared)
