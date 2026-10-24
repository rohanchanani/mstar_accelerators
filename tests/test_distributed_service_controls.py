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
