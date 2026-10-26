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


@pytest.mark.parametrize('flag', ['--config', '--driver'])
def test_no_user_config_or_driver_surface(tmp_path, flag):
    with pytest.raises(SystemExit) as error:
        main([flag, 'unused', '--report', str(tmp_path / 'report.json')])
    assert error.value.code == 2


def test_report_contains_inferred_plan_and_central_policy(tmp_path, monkeypatch):
    from mstar_accelerators.testing import __main__ as cli
    calls = []
    def run(registry):
        calls.append(registry)
        return {'status': 'SKIP', 'complete': False,
                'execution_plan': {'jobs': []},
                'qualification_policy': {'id': 'registry-differential-v1'}}
    monkeypatch.setattr(cli, 'run', run)
    path = tmp_path / 'report.json'
    assert main(['--registry', 'chosen:entries', '--report', str(path)]) == 0
    report = json.loads(path.read_text())
    assert calls == ['chosen:entries'] and report['execution_plan'] == {'jobs': []}
    assert 'config' not in report


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
    assert result["execution_plan"]["jobs"] == []
    assert main([*args, "--require-all-pass"]) == 1
