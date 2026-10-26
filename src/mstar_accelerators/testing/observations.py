"""Read-only observations of the ordinary runtime sampler boundary.

Installed only by qualification, before warmup. Clones captured in a graph are
updated by replay; no sampling choice, input, or state is replaced. These runs
are instrumented correctness runs, never performance measurements.
"""
import torch


class SamplerObservation:
    def __init__(self, resource):
        self.resource = resource
        self._plan, self._sample = resource.plan, resource.sample
        self.snapshots = {}
        self.key = None
        self.cursor = 0
        resource.plan, resource.sample = self.plan, self.sample

    def plan(self, step, ctx):
        result = self._plan(step, ctx)
        if not ctx.is_preplan:
            self.key = ctx.slot_lease
            self.cursor = 0
            if ctx.capture or self.key is None:
                self.snapshots[self.key] = []
        return result

    @torch.compiler.disable
    def sample(self, request_ids, logits, **kwargs):
        # Clone before sampling: some samplers transform their input in place.
        saved = logits.detach().clone()
        tokens = self._sample(request_ids, logits, **kwargs)
        pair = (saved, tokens.detach().clone())
        records = self.snapshots.setdefault(self.key, [])
        if self.cursor == len(records):
            records.append(pair)
        else:
            records[self.cursor] = pair
        self.cursor += 1
        return tokens

    def read(self, lease):
        return [(logits.detach().cpu().clone(), tokens.detach().cpu().clone())
                for logits, tokens in self.snapshots.get(lease, ())]

    def close(self):
        self.resource.plan, self.resource.sample = self._plan, self._sample
        self.snapshots.clear()
