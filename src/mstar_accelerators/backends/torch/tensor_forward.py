"""Optional tensor-forward authoring helpers over M* execution services."""
from dataclasses import dataclass, replace

from mstar.engine.tensor_forward import CapturePolicy, CompilePolicy


@dataclass(frozen=True)
class ResourceMethods:
    """Select numerical resource methods, retaining M*'s state owner.

    The kind selects an explicit runtime implementation. Attention resolution
    is an authored policy; it does not inherit a model's backend override.
    """

    kind: str
    attention_backend: str = "auto"
    flashinfer_backend: str = "auto"

    def build(self, spec, info):
        if self.kind == "attention":
            from mstar.engine.resources.attn.base import AttentionManager
            from mstar.engine.resources.attn.config import AttnBackend
            selected = replace(spec, config=replace(
                spec.config, backend=AttnBackend(self.attention_backend),
                flashinfer_backend=self.flashinfer_backend,
            ))
            return AttentionManager.build(selected, info)
        if self.kind == "cross_attention":
            from mstar.engine.resources.attn.cross import CrossAttentionManager
            from mstar.engine.resources.attn.config import AttnBackend
            selected = replace(spec, config=replace(
                spec.config, backend=AttnBackend(self.attention_backend),
                flashinfer_backend=self.flashinfer_backend,
            ))
            return CrossAttentionManager.build(selected, info)
        if self.kind == "position":
            from mstar.engine.resources.position.manager import PositionManager
            return PositionManager.build(spec, info)
        if self.kind == "sampler":
            from mstar.engine.resources.sampler.resource import SamplerResource
            return SamplerResource.build(spec, info)
        raise ValueError(f"unknown numerical resource methods: {self.kind}")


def tensor_resource_requests(binding):
    """Editable paged/device-plan and numerical-resource choices.

    Callers may replace individual requests or use other physical resource
    implementations. This helper grants no allocation or commit authority.
    """
    from mstar.engine.resources.attn.config import AttentionSpec, CrossAttentionSpec
    from mstar.engine.resources.kv.config import KVSpec
    from mstar.engine.resources.position.config import PositionSpec
    from mstar.engine.resources.sampler.config import SamplerSpec
    from mstar.engine.resources.sequence import HeadSharded, PagedSequenceView

    requests = {}
    for spec in binding.resource_specs:
        if binding.node_name not in spec.nodes:
            continue
        if isinstance(spec, KVSpec):
            request = PagedSequenceView(partition=HeadSharded(), device_plan=True)
        elif isinstance(spec, AttentionSpec):
            request = ResourceMethods("attention")
        elif isinstance(spec, CrossAttentionSpec):
            request = ResourceMethods("cross_attention")
        elif isinstance(spec, PositionSpec):
            request = ResourceMethods("position")
        elif isinstance(spec, SamplerSpec):
            request = ResourceMethods("sampler")
        else:
            # Other resource kinds keep the ordinary bound construction unless
            # the author requests a concrete alternative.
            continue
        requests[spec.resource_key] = request
    return requests


__all__ = ["CompilePolicy", "CapturePolicy", "ResourceMethods", "tensor_resource_requests"]
