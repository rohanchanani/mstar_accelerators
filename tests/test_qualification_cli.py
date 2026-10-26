import json
import pytest

from mstar_accelerators.testing.__main__ import exit_status, main


def test_skip_is_diagnostic_but_requested_coverage_is_explicit():
    skip = {"status": "SKIP", "complete": False}
    assert exit_status(skip) == 0
    assert exit_status(skip, require_all_pass=True) == 1
    mixed = {"status": "PASS", "complete": False}
    assert exit_status(mixed) == 0
    assert exit_status(mixed, require_all_pass=True) == 1
    assert exit_status({"status": "FAIL", "complete": True}) == 1


def test_cli_has_no_driver_override(tmp_path):
    with pytest.raises(SystemExit) as error:
        main(["--driver", "arbitrary:run", "--report", str(tmp_path / "report.json")])
    assert error.value.code == 2


@pytest.mark.parametrize("key,value", [
    ("driver", "arbitrary:run"), ("batches", [1]), ("decode_steps", 1),
    ("comparator", "arbitrary:compare"), ("atol", 100), ("max_trials", 1),
])
def test_environment_cannot_override_testing_policy(tmp_path, key, value):
    config = tmp_path / "config.json"
    report = tmp_path / "report.json"
    config.write_text(json.dumps({"environments": [{"kind": "rocm", key: value}]}))
    with pytest.raises(ValueError, match="central harness"):
        main(["--config", str(config), "--report", str(report)])
    assert not report.exists()


def test_report_identifies_central_policy(tmp_path):
    report = tmp_path / "report.json"
    assert main(["--report", str(report)]) == 0
    result = json.loads(report.read_text())
    assert result["qualification_policy"]["qwen_batches"] == [1, 2, 8]
    assert result["qualification_policy"]["qwen_decode_steps"] == 3
    assert "driver" not in result


def test_default_catalog_can_be_discovered_without_model_assets_or_native_initialization(tmp_path):
    from mstar_accelerators.catalog import accelerations

    report = tmp_path / "discovery.json"
    assert main(["--report", str(report)]) == 0
    result = json.loads(report.read_text())
    assert result["status"] == "SKIP" and not result["complete"]
    assert result["counts"] == {"PASS": 0, "FAIL": 0, "SKIP": len(accelerations)}
    assert [item["name"] for item in result["accelerations"]] == list(accelerations)
    assert result["environments"]["0:pi"]["native_owner_reserved"] is False


def test_empty_corpus_is_diagnostic_and_strict_coverage_remains_a_harness_choice(tmp_path):
    corpus = tmp_path / "accelerations"
    corpus.mkdir()
    report = tmp_path / "empty.json"
    args = ["--registry", str(corpus), "--report", str(report)]
    assert main(args) == 0
    result = json.loads(report.read_text())
    assert result["status"] == "SKIP" and not result["complete"]
    assert result["counts"] == {"PASS": 0, "FAIL": 0, "SKIP": 0}
    assert result["accelerations"] == []
    assert result["environments"]["0:pi"]["native_owner_reserved"] is False
    assert main([*args, "--require-all-pass"]) == 1
