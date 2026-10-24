"""Filesystem authoring against M*'s real logical Whisper description."""
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from mstar.engine.execution import (
    Execution,
    InvocationFacts,
    LogicalInvocation,
    Support,
)
from mstar.model.binding import NodeBinding
from mstar.model.submodule_base import ModelInputsFromEngine, NodeInputs
from mstar.model.whisper.config import WhisperModelConfig
from mstar.model.whisper.whisper_model import WhisperModel

from mstar_accelerators.authoring import create_accelerations, render_acceleration
from mstar_accelerators import TensorForwardExecution
from mstar_accelerators.compiler import bind_tensor_result
from mstar_accelerators.registry import Registry
from mstar_accelerators.sources import load_accelerations
from mstar_accelerators.testing.harness import Result, Status, run_registry


@pytest.fixture
def deployment(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    config = WhisperModelConfig(d_model=8, decoder_layers=1, encoder_layers=1,
                                decoder_attention_heads=2, encoder_attention_heads=2,
                                decoder_ffn_dim=16, encoder_ffn_dim=16, num_mel_bins=4,
                                vocab_size=32, max_target_positions=8, max_source_positions=3,
                                decoder_start_token_id=1, eos_token_id=2)
    (model / "config.json").write_text(json.dumps(asdict(config)))
    spec = {
        "model": "whisper", "model_kwargs": {"model_path_hf": str(model)},
        "device_type": "cpu", "autocast_dtype": "float32", "max_concurrent_requests": 2,
        "node_groups": [{"node_names": ["audio_encoder", "decoder"], "ranks": [0]}],
        "runtime_extension": {"factory": "not_imported:Runtime"},
    }
    path = tmp_path / "deployment.yaml"
    path.write_text(yaml.safe_dump(spec))
    return path


@pytest.mark.parametrize("execution_template", ["callable", "tensor-forward"])
def test_real_cli_is_read_only_deterministic_and_emits_one_file_per_logical_node(
    deployment, tmp_path, execution_template,
):
    original = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    outputs = []
    for name in ("first", "second"):
        result = subprocess.run(
            [sys.executable, "-B", "-m", "mstar_accelerators.authoring", "create",
             "--deployment", str(deployment), "--out-dir", str(tmp_path / name),
             "--execution", execution_template],
            check=False, text=True, capture_output=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        assert result.returncode == 0, result.stderr
        outputs.append([Path(path) for path in json.loads(result.stdout)["created"]])
    assert [path.name for path in outputs[0]] == ["audio_encoder.py", "decoder.py"]
    assert [path.read_bytes() for path in outputs[0]] == [path.read_bytes() for path in outputs[1]]
    assert all(path.read_bytes() == content for path, content in original.items())
    entries = load_accelerations(tmp_path / "first")
    decoder = entries["decoder"]
    facts = {"node_name": "decoder", "device_type": "cpu", "dtype": torch.float32, "tp_size": 1, "sp_size": 1}
    assert decoder.when.evaluate(facts | {"graph_walk": "prefill"}) is Support.YES
    assert decoder.when.evaluate(facts | {"graph_walk": "decode"}) is Support.YES
    assert decoder.when.evaluate(facts | {"graph_walk": "other"}) is Support.NO
    assert sorted(path.name for path in (tmp_path / "first").iterdir()) == ["audio_encoder.py", "decoder.py"]


@pytest.mark.parametrize("device_type", ["cpu", "cuda"])
def test_execution_template_is_author_selected_independently_of_device(deployment, tmp_path, device_type):
    spec = yaml.safe_load(deployment.read_text())
    spec["device_type"] = device_type
    deployment.write_text(yaml.safe_dump(spec))
    create_accelerations(deployment, tmp_path / "default")
    create_accelerations(deployment, tmp_path / "tensor", execution="tensor-forward")
    callback = load_accelerations(tmp_path / "default")["audio_encoder"]
    tensor = load_accelerations(tmp_path / "tensor")["audio_encoder"]
    assert callback.implementation.execution is None
    assert isinstance(tensor.implementation.execution, TensorForwardExecution)
    assert not tensor.implementation.execution.compile.enabled
    assert not tensor.implementation.execution.capture.enabled
    assert tensor.reference is callback.reference
    assert tensor.when == callback.when


def test_unknown_execution_template_rejected_before_reading_or_writing(tmp_path):
    with pytest.raises(ValueError, match="unknown execution template"):
        create_accelerations(tmp_path / "missing.yaml", tmp_path / "books", execution="automatic-gpu")
    assert not (tmp_path / "books").exists()


def test_existing_output_refused_before_any_new_source_is_written(deployment, tmp_path):
    output = tmp_path / "books"
    output.mkdir()
    previous = output / "decoder.py"
    previous.write_text("the agent's edited implementation\n")
    with pytest.raises(FileExistsError, match="decoder.py"):
        create_accelerations(deployment, output)
    assert list(output.iterdir()) == [previous]
    assert previous.read_text() == "the agent's edited implementation\n"


def test_deployment_node_selection_and_explicit_override(deployment, tmp_path):
    spec = yaml.safe_load(deployment.read_text())
    spec["runtime_extension"]["nodes"] = ["decoder"]
    deployment.write_text(yaml.safe_dump(spec))
    paths = create_accelerations(deployment, tmp_path / "selected")
    assert [path.name for path in paths] == ["decoder.py"]
    paths = create_accelerations(deployment, tmp_path / "override", nodes=["audio_encoder"])
    assert [path.name for path in paths] == ["audio_encoder.py"]
    with pytest.raises(ValueError, match="absent from the deployment"):
        create_accelerations(deployment, tmp_path / "unknown", nodes=["missing"])
    assert not (tmp_path / "unknown").exists()


def test_disaggregated_walks_still_produce_one_decoder_file(deployment, tmp_path):
    spec = yaml.safe_load(deployment.read_text())
    spec["node_groups"] = [
        {"node_names": ["audio_encoder", "decoder"], "ranks": [0], "graph_walks": ["prefill"]},
        {"node_names": ["decoder"], "ranks": [1, 2], "graph_walks": ["decode"], "tp_size": 2},
    ]
    deployment.write_text(yaml.safe_dump(spec))
    paths = create_accelerations(deployment, tmp_path / "books")
    assert [path.name for path in paths] == ["audio_encoder.py", "decoder.py"]
    decoder = load_accelerations(tmp_path / "books")["decoder"]
    facts = {"node_name": "decoder", "device_type": "cpu", "dtype": torch.float32, "sp_size": 1}
    for walk, size, support in (("prefill", 1, Support.YES), ("decode", 2, Support.YES),
                                ("prefill", 2, Support.NO), ("decode", 1, Support.NO)):
        assert decoder.when.evaluate(facts | {"graph_walk": walk, "tp_size": size}) is support


def test_unspecified_target_remains_open_instead_of_guessing_a_device(deployment, tmp_path):
    spec = yaml.safe_load(deployment.read_text())
    del spec["device_type"]
    deployment.write_text(yaml.safe_dump(spec))
    create_accelerations(deployment, tmp_path / "books")
    condition = load_accelerations(tmp_path / "books")["decoder"].when
    facts = {"node_name": "decoder", "dtype": torch.float32, "sp_size": 1, "tp_size": 1, "graph_walk": "decode"}
    assert condition.evaluate(facts) is Support.YES


def test_plain_file_copy_adds_identity_without_registration_edits(deployment, tmp_path):
    create_accelerations(deployment, tmp_path / "books")
    source = tmp_path / "books" / "decoder.py"
    shutil.copyfile(source, source.with_name("decoder_specialized.py"))
    entries = load_accelerations(tmp_path / "books")
    assert set(entries) == {"audio_encoder", "decoder", "decoder_specialized"}
    assert entries["decoder"].reference is entries["decoder_specialized"].reference
    assert entries["decoder"].when == entries["decoder_specialized"].when


def test_loading_observes_same_length_source_edits_without_bytecode_staleness(deployment, tmp_path):
    create_accelerations(deployment, tmp_path / "books")
    source = tmp_path / "books" / "decoder.py"
    first = load_accelerations(source)["decoder"]
    content = source.read_text()
    changed = content.replace("'cpu'", "'xpu'")
    assert len(changed) == len(content) and changed != content
    source.write_text(changed)
    second = load_accelerations(source)["decoder"]
    assert first.when.specialize({"device_type": "cpu"}).evaluate({}) is Support.UNKNOWN
    assert second.when.specialize({"device_type": "cpu"}).evaluate({}) is Support.NO


def test_two_logical_nodes_with_one_submodule_class_do_not_collide(deployment, tmp_path):
    config = yaml.safe_load(deployment.read_text())
    model = WhisperModel.describe(**config["model_kwargs"])
    shell = model.create_submodule("audio_encoder")
    directory = tmp_path / "shared_class"
    directory.mkdir()
    for name in ("left", "right"):
        (directory / (name + ".py")).write_text(render_acceleration(
            shell, shell.logical_contract(), (), node_name=name,
            contexts={("prefill", 1, 1)}, device_type="cpu", dtype="float32",
        ))
    entries = load_accelerations(directory)
    assert entries["left"].reference is entries["right"].reference
    graph = SimpleNamespace(get_nodes=lambda: {"left": None, "right": None})
    model = SimpleNamespace(get_graph_walk_graphs=lambda: {"prefill": graph})
    group = SimpleNamespace(world_size=1, rank=0)
    for name in ("left", "right"):
        binding = NodeBinding(model, shell, name, torch.device("cpu"), torch.float32,
                              group, group, {"max_concurrent_requests": 2}, ())
        capabilities = Registry(entries).assemble(binding).bind({})
        assert [option.name for option in capabilities.options] == [name]
        capabilities.close()


@pytest.mark.parametrize("policy", list(Status))
def test_generated_placeholder_executes_and_harness_owns_its_verdict(deployment, tmp_path, policy):
    create_accelerations(deployment, tmp_path / "books")
    entries = load_accelerations(tmp_path / "books")
    config = yaml.safe_load(deployment.read_text())
    model = WhisperModel.describe(**config["model_kwargs"])
    shell = model.create_submodule("audio_encoder")
    group = SimpleNamespace(world_size=1, rank=0)
    binding = NodeBinding(model, shell, "audio_encoder", torch.device("cpu"), torch.float32,
                          group, group, config, ())
    prepared = entries["audio_encoder"].implementation.prepare(binding, {})
    execution = Execution(None, prepared.acquire(InvocationFacts("prefill", 1)))
    logical = NodeInputs(tensor_inputs={"audio_features": torch.ones(4, 6)}, input_seq_len=6)
    invocation = LogicalInvocation("prefill", [0], [logical], ModelInputsFromEngine([0], {}, {}), {})
    call = prepared.materialize(invocation, execution)
    assert type(call).__name__ == "Call" and type(call.inputs).__name__ == "Inputs"
    result = prepared.primitive(call, execution)
    assert result.encoder_states[0].shape == (3, 8)
    assert torch.count_nonzero(result.encoder_states[0]) == 0
    execution.workspace.close()
    prepared.close()
    report = run_registry(entries, lambda name, entry: Result(name, policy, "test consumer policy"))
    assert all(result.status is policy for result in report.results)


def test_tensor_template_prepares_an_independent_program_and_returns_logical_results(deployment, tmp_path):
    create_accelerations(deployment, tmp_path / "books", execution="tensor-forward")
    entries = load_accelerations(tmp_path / "books")
    config = yaml.safe_load(deployment.read_text())
    model = WhisperModel.describe(**config["model_kwargs"])
    shell = model.create_submodule("audio_encoder")
    original_shell_members = {name: id(value) for name, value in vars(shell).items()}
    group = SimpleNamespace(world_size=1, rank=0)
    binding = NodeBinding(model, shell, "audio_encoder", torch.device("cpu"), torch.float32,
                          group, group, config, ())
    program = entries["audio_encoder"].implementation.prepare(binding, {})
    assert isinstance(program, torch.nn.Module) and program is not shell
    assert {name: id(value) for name, value in vars(shell).items()} == original_shell_members
    rows = [NodeInputs(tensor_inputs={"audio_features": torch.ones(4, 6)}, input_seq_len=6)
            for _ in range(2)]
    engine_inputs = ModelInputsFromEngine([0, 1], {}, {})
    invocation = LogicalInvocation("prefill", [0, 1], rows, engine_inputs, {})
    kwargs = program.preprocess("prefill", engine_inputs, rows)
    physical = program.forward_batched("prefill", engine_inputs, **kwargs)
    result = bind_tensor_result(binding)(physical, invocation)
    assert type(result) is shell.logical_contract().result_type
    assert len(result.encoder_states) == 2
    assert all(tensor.shape == (3, 8) and torch.count_nonzero(tensor) == 0
               for tensor in result.encoder_states)
    assert set(physical.row_outputs) == {"encoder_states"}
    single_inputs = ModelInputsFromEngine([0], {}, {})
    single = LogicalInvocation("prefill", [0], rows[:1], single_inputs, {})
    kwargs = program.preprocess("prefill", single_inputs, rows[:1])
    single_result = bind_tensor_result(binding)(program.forward("prefill", single_inputs, **kwargs), single)
    assert len(single_result.encoder_states) == 1
    assert single_result.encoder_states[0].shape == (3, 8)
    assert program.get_accelerator_graph_configs(torch.device("cpu")) == []
    program.close()
