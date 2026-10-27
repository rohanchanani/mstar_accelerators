from types import SimpleNamespace

import pytest

from mstar.engine.execution import ExecutionCapabilities, ExecutionOption, InvocationFacts, Support
from mstar.model.binding import NodeAssembly
from mstar_accelerators.backends.torch.runtime import TorchRuntime
from mstar_accelerators.testing.distributed_controls import ControlledTorchRuntime


def controlled(monkeypatch, mode, rank, events):
    def option(name):
        def acquire(facts):
            events.append(("acquire", name))
            return name

        def primitive(call, execution):
            events.append(("primitive", name, call))
            return call

        return ExecutionOption(name, lambda facts: Support.YES, acquire,
                               lambda invocation, execution: invocation, primitive)

    original = ExecutionCapabilities((option("first"), option("second")), {"prefill": 1})
    monkeypatch.setattr(TorchRuntime, "assemble", lambda self, binding: NodeAssembly(lambda resources: original))
    runtime = ControlledTorchRuntime({"accelerations": "unused:declarations", "distributed_control": mode})
    binding = SimpleNamespace(node_name="decoder", tp_group=SimpleNamespace(rank=rank, world_size=2))
    return runtime.assemble(binding).bind({})


@pytest.mark.parametrize("rank", [0, 1])
def test_refusal_is_one_local_workspace_attempt(monkeypatch, rank):
    events = []
    options = controlled(monkeypatch, "refuse_once", rank, events).options
    facts = InvocationFacts("prefill", 1, (4,))
    assert options[0].acquire(facts) == (None if rank == 1 else "first")
    assert options[1].acquire(facts) == "second"
    assert options[0].acquire(facts) == "first"
    assert events == ([] if rank == 1 else [("acquire", "first")]) + [
        ("acquire", "second"), ("acquire", "first")]


@pytest.mark.parametrize("rank", [0, 1])
def test_failure_occurs_only_after_the_actual_primitive(monkeypatch, rank):
    events = []
    option = controlled(monkeypatch, "fail_after_execution", rank, events).options[0]
    if rank == 1:
        with pytest.raises(RuntimeError, match="after the real TP2 primitive returned"):
            option.primitive("first-call", None)
    else:
        assert option.primitive("first-call", None) == "first-call"
    assert events == [("primitive", "first", "first-call")]


def test_cpu_lifecycle_receipts_preserve_worker_and_process_identity():
    from tools.whisper.demo import cpu_lifecycle

    log = """INFO [worker-a] runtime: TORCH_RUNTIME_READY pid=120 device=cpu threads=1
INFO [worker-b] runtime: TORCH_RUNTIME_READY pid=121 device=cpu threads=1
INFO [worker-a] comm: Process groups closed: rank=0 failed=False
INFO [worker-b] comm: Process groups closed: rank=1 failed=False
INFO [worker-a] runtime: TORCH_RUNTIME_RELEASED pid=120
INFO [worker-b] runtime: TORCH_RUNTIME_RELEASED pid=121
"""
    result = cpu_lifecycle(log)
    assert result["ready"] == result["released"] == [
        {"worker": "worker-a", "pid": 120}, {"worker": "worker-b", "pid": 121}]
    assert result["process_groups"] == [{"rank": 0, "failed": False}, {"rank": 1, "failed": False}]


def test_cpu_placement_keeps_encoder_worker_separate_from_two_decoder_roles():
    from tools.whisper.demo import cpu_worker_placement

    config = {"node_groups": [
        {"node_names": ["audio_encoder"], "ranks": [2]},
        {"node_names": ["decoder"], "ranks": [0, 1], "tp_size": 2},
    ]}
    assert cpu_worker_placement(config) == ({0, 1, 2}, {"worker_0", "worker_1"})
    config["node_groups"][0]["ranks"] = [0]
    assert cpu_worker_placement(config) == ({0, 1}, {"worker_0", "worker_1"})
    config["node_groups"][1]["tp_size"] = 1
    with pytest.raises(ValueError, match="decoder TP1 or TP2"):
        cpu_worker_placement(config)


def test_cpu_decoder_sequence_checks_require_both_actual_placed_roles():
    from tools.whisper.demo import cpu_decoder_sequences

    sequence = [{"walk": "prefill", "implementation": "decoder_all_reduce", "rows": 1}]
    lifecycle = {"decoder_executions": {"worker_0": sequence, "worker_1": sequence, "worker_2": []}}
    workers = {"worker_0", "worker_1"}
    assert cpu_decoder_sequences(lifecycle, workers) == [sequence, sequence]
    for executions in ({"worker_0": sequence},
                       {"worker_0": sequence, "worker_1": sequence, "worker_2": sequence},
                       {"worker_0": sequence, "worker_1": sequence + sequence}):
        with pytest.raises(RuntimeError):
            cpu_decoder_sequences({"decoder_executions": executions}, workers)


def test_cpu_tp1_requires_one_actual_decoder_and_no_process_groups():
    from tools.whisper.demo import cpu_decoder_sequences, cpu_groups_closed, cpu_worker_placement

    config = {"node_groups": [
        {"node_names": ["audio_encoder"], "ranks": [1]},
        {"node_names": ["decoder"], "ranks": [0], "tp_size": 1},
    ]}
    assert cpu_worker_placement(config) == ({0, 1}, {"worker_0"})
    sequence = [{"walk": "prefill", "implementation": "whisper/cpu_tp1/decoder", "rows": 1}]
    lifecycle = {"decoder_executions": {"worker_0": sequence, "worker_1": []}, "process_groups": []}
    assert cpu_decoder_sequences(lifecycle, {"worker_0"}) == [sequence]
    assert cpu_groups_closed(config, lifecycle)
    lifecycle["decoder_executions"]["worker_1"] = sequence
    with pytest.raises(RuntimeError):
        cpu_decoder_sequences(lifecycle, {"worker_0"})
    lifecycle["process_groups"] = [{"rank": 0, "failed": False}]
    assert not cpu_groups_closed(config, lifecycle)


def test_cpu_tp2_still_requires_all_placed_process_groups_to_close():
    from tools.whisper.demo import cpu_groups_closed

    config = {"node_groups": [
        {"node_names": ["audio_encoder"], "ranks": [2]},
        {"node_names": ["decoder"], "ranks": [0, 1], "tp_size": 2},
    ]}
    assert not cpu_groups_closed(config, {"process_groups": []})
    assert not cpu_groups_closed(config, {"process_groups": [{"rank": 0, "failed": False},
                                                            {"rank": 1, "failed": False}]})
    assert cpu_groups_closed(config, {"process_groups": [{"rank": rank, "failed": False}
                                                        for rank in (0, 1, 2)]})
