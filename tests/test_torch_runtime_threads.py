from types import SimpleNamespace

import pytest

from mstar_accelerators.backends.torch import runtime as module


@pytest.fixture
def environment(monkeypatch):
    state = {"threads": 7, "sets": [], "assembled": []}

    def set_threads(count):
        state["threads"] = count
        state["sets"].append(count)

    class Registry:
        def __init__(self, declarations):
            pass

        def assemble(self, binding):
            state["assembled"].append((binding.node_name, state["threads"]))
            return binding.node_name

    monkeypatch.setattr(module.torch, "set_num_threads", set_threads)
    monkeypatch.setattr(module.torch, "get_num_threads", lambda: state["threads"])
    monkeypatch.setattr(module, "Registry", Registry)
    monkeypatch.setattr(module, "load_accelerations", lambda source: {})
    return state


def runtime(**policy):
    result = module.TorchRuntime({"accelerations": "unused:declarations", **policy})
    result.initialize(SimpleNamespace(type="cpu"))
    return result


def test_default_changes_threads_only_at_initialize(environment):
    instance = runtime(threads=1)
    for node in ("audio_encoder", "decoder"):
        instance.assemble(SimpleNamespace(node_name=node))
    assert environment["sets"] == [1]
    assert environment["assembled"] == [("audio_encoder", 1), ("decoder", 1)]
    instance.close()
    assert instance.registry is None


def test_encoder_policy_is_set_once_before_registry_assembly(environment, caplog):
    with caplog.at_level("INFO", logger=module.__name__):
        instance = runtime(threads=1, node_threads={"audio_encoder": 2})
        for _ in range(2):
            instance.assemble(SimpleNamespace(node_name="audio_encoder"))
    assert environment["sets"] == [1, 2]
    assert environment["assembled"] == [("audio_encoder", 2), ("audio_encoder", 2)]
    assert "node=audio_encoder threads=2" in caplog.text


def test_unmapped_decoder_uses_default_without_changing_it(environment):
    instance = runtime(threads=1, node_threads={"audio_encoder": 2})
    instance.assemble(SimpleNamespace(node_name="decoder"))
    assert environment["sets"] == [1]
    assert environment["assembled"] == [("decoder", 1)]


@pytest.mark.parametrize("order", [("audio_encoder", "decoder"), ("decoder", "audio_encoder")])
def test_conflicting_local_nodes_refuse_before_second_assembly(environment, order):
    instance = runtime(threads=1, node_threads={"audio_encoder": 2})
    instance.assemble(SimpleNamespace(node_name=order[0]))
    previous_sets = list(environment["sets"])
    with pytest.raises(ValueError, match="place them on separate workers"):
        instance.assemble(SimpleNamespace(node_name=order[1]))
    assert environment["sets"] == previous_sets
    assert len(environment["assembled"]) == 1


def test_same_policy_can_cover_multiple_local_nodes(environment):
    instance = runtime(threads=1, node_threads={"audio_encoder": 2, "decoder": 2})
    for node in ("audio_encoder", "decoder"):
        instance.assemble(SimpleNamespace(node_name=node))
    assert environment["sets"] == [1, 2]
    assert environment["assembled"] == [("audio_encoder", 2), ("decoder", 2)]


@pytest.mark.parametrize("node_threads", [None, [], {"audio_encoder": 0}, {"audio_encoder": True},
                                         {"audio_encoder": 1.5}, {"": 2}, {2: 2}])
def test_invalid_node_policy_is_rejected(node_threads):
    with pytest.raises(ValueError, match="node_threads"):
        module.TorchRuntime({"accelerations": "unused:declarations", "node_threads": node_threads})


def test_policy_is_copied_from_deployment_config(environment):
    policy = {"audio_encoder": 2}
    instance = runtime(node_threads=policy)
    policy["audio_encoder"] = 3
    instance.assemble(SimpleNamespace(node_name="audio_encoder"))
    assert environment["threads"] == 2
