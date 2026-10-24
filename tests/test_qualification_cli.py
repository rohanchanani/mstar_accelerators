import json
import sys
from types import ModuleType

from mstar_accelerators.testing.__main__ import exit_status, main


def test_skip_is_diagnostic_but_requested_coverage_is_explicit():
    skip = {"status": "SKIP", "complete": False}
    assert exit_status(skip) == 0
    assert exit_status(skip, require_all_pass=True) == 1
    mixed = {"status": "PASS", "complete": False}
    assert exit_status(mixed) == 0
    assert exit_status(mixed, require_all_pass=True) == 1
    assert exit_status({"status": "FAIL", "complete": True}) == 1


def test_cli_can_run_an_independent_driver_without_pi_or_model_flags(tmp_path, monkeypatch):
    calls = []
    driver = ModuleType("_qualification_test_driver")

    def run(registry, options):
        calls.append((registry, options))
        return {"status": "PASS", "complete": True, "counts": {"PASS": 1, "FAIL": 0, "SKIP": 0}}

    driver.run = run
    monkeypatch.setitem(sys.modules, driver.__name__, driver)
    config, report = tmp_path / "config.json", tmp_path / "report.json"
    config.write_text(json.dumps({"workload": "custom"}))
    assert main(["--registry", "other.module:corpus", "--driver", driver.__name__ + ":run",
                 "--config", str(config), "--report", str(report), "--require-all-pass"]) == 0
    assert calls == [("other.module:corpus", {"workload": "custom"})]
    assert json.loads(report.read_text())["driver"] == driver.__name__ + ":run"


def test_default_catalog_can_be_discovered_without_model_assets_or_native_initialization(tmp_path):
    from mstar_accelerators.catalog import accelerations

    report = tmp_path / "discovery.json"
    assert main(["--report", str(report)]) == 0
    result = json.loads(report.read_text())
    assert result["status"] == "SKIP" and not result["complete"]
    assert result["counts"] == {"PASS": 0, "FAIL": 0, "SKIP": len(accelerations)}
    assert [item["name"] for item in result["accelerations"]] == list(accelerations)
    assert result["native_owner_reserved"] is False


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
    assert result["native_owner_reserved"] is False
    assert main([*args, "--require-all-pass"]) == 1
