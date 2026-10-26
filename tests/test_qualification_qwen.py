"""CPU rehearsal of the real Qwen program through EngineManager and consumer.

The declaration is explicitly CPU/FP32 with compilation and capture disabled.
Only GPU-only attention, RoPE, sampling and resource construction are supplied
as host diagnostic oracles. The selected numerical program, registry, engine,
resource owners, typed result adapter and differential consumer are real.
This is not execution or qualification of the production AMD declaration.
"""
from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch

from mstar.engine.resources.sequence import PagedSequenceView
from mstar.model.qwen3_omni.logical import TalkerSubmodule
from mstar.model.qwen3_omni.qwen3_omni_model import Qwen3OmniModel
from mstar_accelerators import Acceleration, Eq, Implementation, Or, Range, TensorForwardExecution, When
from mstar_accelerators.backends.torch.tensor_forward import tensor_resource_requests
from mstar_accelerators.implementations.qwen3_omni.amd.preparation import prepare_program
from mstar_accelerators.sources import load_accelerations
from mstar_accelerators.testing.harness import DifferentialConsumer, Status, run_registry
from mstar_accelerators.testing.models.qwen3_omni import QualificationSession, TalkerCaseAdapter
from mstar_accelerators.testing.numerics import differential_comparator
from mstar_accelerators.testing.policy import consumer as central_consumer


@pytest.fixture
def tiny_model(tmp_path):
    from test.fixtures.qwen3_omni.tiny_checkpoint import write_tiny_checkpoint
    write_tiny_checkpoint(tmp_path, num_code_groups=16)
    return Qwen3OmniModel.describe(model_path_hf=str(tmp_path))


def _cpu_declaration():
    def requests(binding):
        return {key: replace(value, device_plan=False) if isinstance(value, PagedSequenceView) else value
                for key, value in tensor_resource_requests(binding).items()}

    return Acceleration(
        reference=TalkerSubmodule,
        when=When(Eq("node_name", "Talker"), Eq("device_type", "cpu"),
                  Eq("dtype", torch.float32), Eq("tp_size", 1), Eq("sp_size", 1),
                  Range("batch_size", 1, 8),
                  Or(*(Eq("graph_walk", walk) for walk in
                       ("talker_prefill", "talker_last_prefill", "talker_decode")))),
        implementation=Implementation(prepare_program, resources=requests,
                                      execution=TensorForwardExecution()),
    )


def test_actual_program_and_engine_consumed_for_every_talker_branch(tiny_model, monkeypatch):
    import mstar.engine.engine as engine_module
    from test.modular.test_qwen3_omni_binding import _cpu_resources, _install_cpu_oracles

    _install_cpu_oracles(monkeypatch)
    resources = {}
    built = []

    def host_resource(spec, info, requests=(), **kwargs):
        assert info.device.type == "cpu"
        if not resources:
            resources.update(_cpu_resources(tiny_model.create_submodule("Talker"),
                                            joint_comm_group=info.joint_comm_group))
        built.append(spec.resource_key)
        return resources[spec.resource_key]

    monkeypatch.setattr(engine_module, "build_bound_resource", host_resource)
    session = QualificationSession()
    consumer = central_consumer(tiny_model, session=session, device="cpu",
                                options={"dtype": "float32"})
    try:
        report = run_registry({"cpu-rehearsal": _cpu_declaration()}, consumer)
        failures = [check for row in report.results for check in row.checks if not check["passed"]]
        assert report.status is Status.PASS, failures[:10]
        assert report.complete and len(report.results) == 1
        result, = report.results
        assert result.details["coverage"]["branches_enumerated"] == 3
        assert all(case["executed"] for case in result.details["cases"])
        trajectories = result.details["trajectories"]
        assert [row["batch_size"] for row in trajectories] == [1, 2, 8]
        assert all(row["executed"] for row in trajectories)
        assert all(step["path"] == "eager" for row in trajectories for step in row["steps"])
        largest = next(row for row in trajectories if row["batch_size"] == 8)
        survivors = next(step for step in largest["steps"] if step["label"] == "survivors-after-removal")
        assert survivors["rows"] == 7
        check_names = {check["name"] for check in result.checks}
        for suffix in ("/codec_tokens", "/codec_sum", "/kv_length", "/position",
                       "/sampler_draws", "/seen_tokens", "/retained_output",
                       "/all_pages_released", "/logical_conditioning_released"):
            assert any(name.endswith(suffix) for name in check_names), suffix
        for layer in range(tiny_model.config.talker_text.num_hidden_layers):
            for component in ("k", "v"):
                assert any(name.endswith(f"/layer{layer}/{component}") for name in check_names)
        assert set(built) == set(resources)
        manager, = session.managers
        shell = manager.engine._submodules["Talker"].submodule
        assert not hasattr(shell, "model") and list(shell.parameters()) == []
        option, = shell.execution_capabilities.options
        assert type(option.runner.program).__name__ == "TalkerProgram"
        assert option.runner.program.model.__class__.__module__.startswith("mstar_accelerators.implementations.")
        print(json.dumps({
            "scope": "CPU diagnostic oracles; actual registered Talker program and EngineManager",
            "comparisons": len(result.checks),
            "branches": result.details["coverage"]["branches_enumerated"],
            "batches": [row["batch_size"] for row in trajectories],
            "engine_steps": sum(len(row["steps"]) for row in trajectories),
            "gpu_execution": False,
            "captured_execution": False,
        }, sort_keys=True))
    finally:
        session.close()
    assert session.managers == []


def test_production_amd_talker_declaration_is_honest_skip_on_cpu(tiny_model):
    book = Path(__file__).parents[1] / "accelerations/qwen3_omni/amd"
    entries = load_accelerations(str(book / "Talker.py"))
    session = QualificationSession()
    adapter = TalkerCaseAdapter(tiny_model, session=session, device="cpu",
                               options={"dtype": "bfloat16"})
    consumer = DifferentialConsumer({TalkerSubmodule: adapter}, differential_comparator)
    report = run_registry(entries, consumer)
    assert report.status is Status.SKIP and not report.complete
    assert all(not row.details["executed"] and row.checks == () for row in report.results)
    assert adapter.oracle is None and session.managers == []


def test_qualification_offset_observes_deferred_reuse_without_mutating_sampler():
    from mstar.engine.resources.sampler.resource import SamplerResource
    from mstar_accelerators.testing.models.qwen3_omni import _offset

    owner = SamplerResource(64, False, torch.device("cpu"))
    owner.build_accelerator_graph_buffers(slots=[], max_bs=1, max_seq_len=1)
    buffers = owner._cg_buffers
    owner.ingest_request(10)
    slot = buffers._rid_to_slot[10]
    assert _offset(owner, 10) == 0
    assert slot in buffers._pending_init
    offsets = buffers.offsets_for([10])
    buffers.advance_offsets([10], offsets, 15)
    assert _offset(owner, 10) == 15
    assert owner._sampler._step_offset[10] == 0  # device master is authoritative
    owner.remove_request(10)

    owner.ingest_request(20)
    assert buffers._rid_to_slot[20] == slot
    assert buffers.offset.master[slot].item() == 15  # old physical row until reset
    assert _offset(owner, 20) == 0
    assert slot in buffers._pending_init
    assert buffers.offset.master[slot].item() == 15

    # Pre-planning can install scalar parameters before the GPU-thread reset.
    buffers._init_slot(slot, 20)
    assert slot not in buffers._pending_init and slot in buffers._pending_offset_reset
    assert _offset(owner, 20) == 0
    assert slot in buffers._pending_offset_reset
    assert buffers.offset.master[slot].item() == 15

    offsets = buffers.offsets_for([20])
    assert offsets.tolist() == [0]
    buffers.advance_offsets([20], offsets, 1)
    assert _offset(owner, 20) == 1
    owner.remove_request(20)
