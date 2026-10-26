"""Asset identity, deployment rendering, and fail-closed benchmark reporting."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_tool(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"tools/qwen3_omni/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("algorithm", ["sha256", "git_blob_sha1"])
def test_asset_verification_rejects_same_size_corruption(tmp_path, algorithm):
    tool = load_tool("prepare_assets")
    path = tmp_path / "tensor"
    content = b"checkpoint bytes"
    path.write_bytes(content)
    digest = (hashlib.sha256(content).hexdigest() if algorithm == "sha256" else
              hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest())
    record = {"bytes": len(content), algorithm: digest}
    assert tool.verify_file(path, record)[algorithm] == digest
    path.write_bytes(b"X" + content[1:])
    with pytest.raises(ValueError, match="mismatch"):
        tool.verify_file(path, record)


def test_asset_verification_rejects_truncated_download(tmp_path):
    path = tmp_path / "tensor"
    path.write_bytes(b"short")
    with pytest.raises(ValueError, match="wrong size"):
        load_tool("prepare_assets").verify_file(path, {"bytes": 12, "sha256": "unused"})


def test_asset_manifest_is_complete_for_the_pinned_checkpoint():
    manifest = json.loads((ROOT / "requirements/qwen3-omni-assets.json").read_text())
    assert manifest["revision"] == "26291f793822fb6be9555850f06dfe95f2d7e695"
    names = [row["path"] for row in manifest["files"]]
    assert len(names) == len(set(names))
    assert sum(name.endswith(".safetensors") for name in names) == 15
    assert {"config.json", "model.safetensors.index.json", "tokenizer_config.json",
            "chat_template.json", "preprocessor_config.json", "merges.txt", "vocab.json"} <= set(names)
    assert all(row["bytes"] > 0 and ("sha256" in row or "git_blob_sha1" in row)
               for row in manifest["files"])


def test_rendered_deployment_preserves_policy_and_selects_only_talker(tmp_path):
    tool = load_tool("prepare_assets")
    template = ROOT / "deployments/qwen3_omni_amd.yaml"
    checkpoint = tmp_path / "model with spaces"
    destination = tmp_path / "rendered.yaml"
    source = yaml.safe_load(template.read_text())
    config = tool.render_deployment(template, checkpoint, destination)
    assert config["runtime_extension"]["nodes"] == ["Talker"]
    assert config["runtime_extension"]["factory"] == \
        "mstar_accelerators.backends.torch.device_runtime:DeviceTorchRuntime"
    assert Path(config["runtime_extension"]["accelerations"]) == ROOT / "accelerations/qwen3_omni/amd"
    assert Path(config["runtime_extension"]["accelerations"]).is_dir()
    assert config["model_kwargs"]["model_path_hf"] == str(checkpoint.resolve())
    assert config["resources"] == source["resources"]
    assert config["node_groups"] == source["node_groups"]
    assert config["max_concurrent_requests"] == 8
    assert yaml.safe_load(destination.read_text()) == config


def benchmark_report():
    row = {"error": None, "audio_s": 2.0, "ttfa_s": .1}
    return {
        "errors_total": 0, "num_sentences": 1, "concurrency": 1,
        "repeats_detail": [{"errors": 0, "requests": 1, "results": [row]}],
        "median_over_repeats": {"ttfa_p50_ms": 100, "audio_s_per_wall_s": 4, "rtf_mean": .25},
    }


def test_baseline_summary_counts_completed_requests():
    summary = load_tool("run_demo").check_benchmark(benchmark_report())
    assert summary == {"concurrency": 1, "completed": 1, "errors": 0,
                       "ttfa_p50_ms": 100, "audio_s_per_wall_s": 4, "rtf_mean": .25}
    assert "ttfa_p95_ms" not in summary


def test_baseline_summary_requires_the_requested_concurrency_and_request_count():
    tool = load_tool("run_demo")
    with pytest.raises(RuntimeError, match="concurrency"):
        tool.check_benchmark(benchmark_report(), expected_concurrency=8, expected_requests=1)
    with pytest.raises(RuntimeError, match="request count"):
        tool.check_benchmark(benchmark_report(), expected_concurrency=1, expected_requests=8)


@pytest.mark.parametrize("fault", ["errors", "missing", "truncated", "missing_row", "empty_audio", "no_first_audio", "row_error"])
def test_baseline_summary_refuses_failed_or_incomplete_requests(fault):
    report = benchmark_report()
    if fault == "errors":
        report["errors_total"] = 1
    elif fault == "missing":
        report["repeats_detail"] = []
    elif fault == "truncated":
        report["repeats_detail"][0]["requests"] = 0
    elif fault == "missing_row":
        report["repeats_detail"][0]["results"] = []
    elif fault == "empty_audio":
        report["repeats_detail"][0]["results"][0]["audio_s"] = 0
    elif fault == "no_first_audio":
        report["repeats_detail"][0]["results"][0]["ttfa_s"] = None
    elif fault == "row_error":
        report["repeats_detail"][0]["results"][0]["error"] = "failed stream"
    with pytest.raises(RuntimeError):
        load_tool("run_demo").check_benchmark(report)


def test_shutdown_requires_each_started_worker_to_release():
    tool = load_tool("run_demo")
    ready = "DEVICE_TORCH_RUNTIME_READY pid=100\nDEVICE_TORCH_RUNTIME_READY pid=101\n"
    with pytest.raises(RuntimeError):
        tool.check_shutdown(ready + "DEVICE_TORCH_RUNTIME_RELEASED pid=100\n")
    receipts = tool.check_shutdown(ready + "DEVICE_TORCH_RUNTIME_RELEASED pid=100\nDEVICE_TORCH_RUNTIME_RELEASED pid=101\n")
    assert receipts["ready_pids"] == receipts["released_pids"] == ["100", "101"]


@pytest.mark.parametrize("receipt", [
    "", "ENGINE_EXECUTION node=Talker implementation=Talker walk=talker_decode path=eager rows=8 runner=tensor_forward",
    "ENGINE_EXECUTION node=Talker implementation=Talker walk=talker_decode path=captured rows=8",
    "ENGINE_EXECUTION node=Talker implementation=Talker walk=talker_prefill path=captured rows=8 runner=tensor_forward",
    "ENGINE_EXECUTION node=Talker implementation=Talker walk=talker_decode path=captured rows=0 runner=tensor_forward",
])
def test_demo_requires_actual_tensor_forward_decode_replay(receipt):
    with pytest.raises(RuntimeError, match="captured tensor-forward Talker decode"):
        load_tool("run_demo").check_execution_receipts(receipt)


def test_demo_accepts_the_actual_runner_decode_receipt():
    receipt = "ENGINE_EXECUTION node=Talker implementation=Talker walk=talker_decode path=captured rows=8 runner=tensor_forward"
    assert load_tool("run_demo").check_execution_receipts(receipt) == [receipt]


@pytest.mark.parametrize("body", ["", '<testcase name="gpu"><skipped message="no GPU"/></testcase>',
                                  '<testcase name="gpu"><failure/></testcase>'])
def test_preflight_cannot_count_skips_or_empty_reports_as_execution(tmp_path, body):
    report = tmp_path / "kernels.xml"
    report.write_text(f"<testsuites><testsuite>{body}</testsuite></testsuites>")
    with pytest.raises(RuntimeError):
        load_tool("preflight").check_kernel_report(report)


def test_preflight_records_only_executed_passing_kernels(tmp_path):
    report = tmp_path / "kernels.xml"
    report.write_text('<testsuites><testsuite><testcase name="gpu"/></testsuite></testsuites>')
    assert load_tool("preflight").check_kernel_report(report) == {"passed": 1, "skipped": 0, "failed": 0}


@pytest.mark.parametrize("wrong_module", ["mstar", "mstar_accelerators"])
def test_preflight_rejects_imports_from_another_checkout(tmp_path, wrong_module):
    tool = load_tool("preflight")
    mstar_root = tmp_path / "mstar"
    imports = {
        "mstar": str(mstar_root / "mstar/__init__.py"),
        "mstar_accelerators": str(ROOT / "src/mstar_accelerators/__init__.py"),
    }
    tool.check_source_imports(imports, mstar_root)
    imports[wrong_module] = str(tmp_path / "other_checkout" / wrong_module / "__init__.py")
    with pytest.raises(RuntimeError, match=f"wrong {wrong_module} import"):
        tool.check_source_imports(imports, mstar_root)


def test_preflight_includes_the_real_device_capture_adapter():
    import mstar
    expected = "test/modular/test_tensor_forward_execution.py" + \
        "::test_separate_program_real_engine_and_stable_logical_outputs[gpu]"
    assert expected in load_tool("preflight").KERNEL_TESTS
    assert (Path(mstar.__file__).resolve().parents[1] / expected.split("::", 1)[0]).is_file()


@pytest.mark.parametrize("fault", ["skip", "fail", "incomplete", "empty", "partial"])
def test_campaign_cannot_advance_after_incomplete_qualification(fault):
    report = {"status": "PASS", "complete": True, "counts": {"PASS": 1, "FAIL": 0, "SKIP": 0},
              "accelerations": [{"status": "PASS"}]}
    if fault in ("skip", "fail"):
        report["status"] = fault.upper()
    elif fault == "incomplete":
        report["complete"] = False
    elif fault == "empty":
        report["accelerations"] = []
    else:
        report["accelerations"].append({"status": "SKIP"})
    with pytest.raises(RuntimeError, match="complete PASS"):
        load_tool("run_foundation").check_qualification_report(report)


def test_campaign_stage_retains_actual_child_failure_and_stops(tmp_path):
    tool = load_tool("run_foundation")
    record = {"stages": []}
    command = [sys.executable, "-c", "import sys; print('retained child failure'); sys.exit(7)"]
    with pytest.raises(RuntimeError, match="exit 7"):
        tool.run_stage("first", command, tmp_path, record, cwd=tmp_path)
    saved = json.loads((tmp_path / "foundation.json").read_text())
    assert len(saved["stages"]) == 1 and saved["stages"][0]["returncode"] == 7
    assert "retained child failure" in (tmp_path / "first.log").read_text()


def test_launch_configuration_supplies_environment_not_test_case_policy():
    config = {"model_kwargs": {"model_path_hf": "/checkpoint"},
              "runtime_extension": {"nodes": ["Talker"]}}
    assert load_tool("run_foundation").qualification_options(config) == \
        {"checkpoint": "/checkpoint", "device": "cuda:0", "threads": 4}


def test_source_archive_identity_uses_its_own_files_and_changes_with_source(tmp_path):
    # A delivered source tree may be inside an unrelated parent Git checkout.
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    archive = tmp_path / "source"
    archive.mkdir()
    source = archive / "program.py"
    source.write_text("VALUE = 1\n")
    (archive / "local.env").write_text("excluded environment content\n")
    (archive / "build").mkdir()
    (archive / "build/generated.py").write_text("generated build output\n")
    manifest = tmp_path / "SOURCE_MANIFEST.json"
    manifest.write_text('{"format": "source export"}\n')
    tool = load_tool("run_demo")
    first = tool.source_identity(archive)
    assert first["kind"] == "source_archive"
    assert set(first["files_sha256"]) == {"program.py"}
    assert first["export_manifest"]["sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    source.write_text("VALUE = 2\n")
    assert tool.source_identity(archive)["source_tree_sha256"] != first["source_tree_sha256"]
