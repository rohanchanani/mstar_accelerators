import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "_trace_summary", Path(__file__).parents[1] / "tools/qualification/summarize_trace.py")
trace_summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(trace_summary)


def trace_file(tmp_path, lines):
    path = tmp_path / "recorded.trace"
    path.write_text("\n".join(lines) + "\n")
    return path


def test_interleaved_resumptions_keep_result_and_start_identity(tmp_path):
    path = trace_file(tmp_path, [
        "11  01:02:03.000001 ioctl(5</dev/dri/renderD128<char 226:128>>, DRM_IOCTL_V3D_SUBMIT_CSD, 0x1 <unfinished ...>",
        "12  01:02:03.000002 ioctl(6</dev/dri/renderD128<char 226:128>>, _IOC(_IOC_READ|_IOC_WRITE, 0x64, 0xc1, 0x18), 0x2 <unfinished ...>",
        "12  01:02:03.000003 <... ioctl resumed>) = -1 EINVAL (Invalid argument)",
        "11  01:02:03.000004 <... ioctl resumed>) = 0",
        "12  01:02:03.000005 ioctl(6, _IOC(_IOC_READ|_IOC_WRITE, 0x64, 0xc1, 0x18) , 0x2) = 0",
        "11  01:02:03.000006 ioctl(9, SYNC_IOC_MERGE, 0x3) = 0",
        "11  01:02:03.000007 openat(AT_FDCWD, \"/lib/libvulkan_broadcom.so\", O_RDONLY) = 7</lib/libvulkan_broadcom.so>",
        "11  01:02:03.000008 openat(AT_FDCWD, \"/missing/libggml.so\", O_RDONLY) = -1 ENOENT (No such file)",
        "11  01:02:03.000009 +++ exited with 0 +++",
    ])
    result = trace_summary.summarize(path)
    assert result["parser_status"] == "PASS"
    assert result["trace"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert result["requests"]["DRM_IOCTL_V3D_SUBMIT_CSD"] == {
        "calls": 1, "accepted": 1, "rejected": 0, "unresolved": 0}
    assert result["requests"][trace_summary.EXPORT] == {
        "calls": 2, "accepted": 1, "rejected": 1, "unresolved": 0}
    assert result["sync_export_einval"]["count"] == 1
    rejection, = result["rejected_requests"]
    assert rejection["pid"] == 12 and rejection["start_line"] == 2 and rejection["end_line"] == 3
    assert [row["outcome"] for row in result["provider_opens"]] == ["accepted", "rejected"]
    assert not result["unknown_ioctls"]
    assert result["events"] == {"process_exit": 1}


@pytest.mark.parametrize("unfinished,resumed", [
    ("ioctl(5, DRM_IOCTL_V3D_SUBMIT_CSD, 0x1 <unfinished ...>", "<... openat resumed>) = 0"),
    (None, "<... ioctl resumed>) = 0"),
    ("ioctl(5, DRM_IOCTL_V3D_SUBMIT_CSD, 0x1 <unfinished ...>", None),
])
def test_missing_or_mismatched_tail_never_counts_as_accepted(tmp_path, unfinished, resumed):
    lines = [f"11  01:02:03.00000{i} {body}" for i, body in enumerate((unfinished, resumed)) if body]
    path = trace_file(tmp_path, lines)
    result = trace_summary.summarize(path)
    assert result["parser_status"] == "INCOMPLETE"
    assert result["requests"]["DRM_IOCTL_V3D_SUBMIT_CSD"]["accepted"] == 0
    assert bool(result["unmatched_unfinished"]) is bool(unfinished)
    assert bool(result["unmatched_resumed"]) is bool(resumed)
    output = tmp_path / "summary.json"
    assert trace_summary.main(["--trace", str(path), "--output", str(output)]) == 1
    assert json.loads(output.read_text())["parser_status"] == "INCOMPLETE"


def test_other_engines_rejections_and_unknowns_are_reported_without_workload_assertions(tmp_path):
    path = trace_file(tmp_path, [
        "11  01:02:03.000001 ioctl(5, DRM_IOCTL_V3D_SUBMIT_CPU, 0x1) = 0",
        "11  01:02:03.000002 ioctl(5, DRM_IOCTL_V3D_SUBMIT_CL, 0x1) = -1 EINVAL (Invalid argument)",
        # Same nr but another payload size is deliberately NOT classified.
        "11  01:02:03.000003 ioctl(5, _IOC(_IOC_READ|_IOC_WRITE, 0x64, 0xc1, 0x10), 0x2) = 0",
        "11  01:02:03.000004 openat(AT_FDCWD, \"/lib/libggml.so\", O_RDONLY) = 7</lib/libggml.so>",
    ])
    result = trace_summary.summarize(path)
    assert result["parser_status"] == "PASS"
    assert result["requests"]["DRM_IOCTL_V3D_SUBMIT_CPU"]["accepted"] == 1
    assert result["requests"]["DRM_IOCTL_V3D_SUBMIT_CL"]["rejected"] == 1
    assert result["requests"][trace_summary.EXPORT]["calls"] == 0
    assert len(result["unknown_ioctls"]) == 1
    assert not result["ioctl_classification_complete"]
    assert result["provider_opens"][0]["outcome"] == "accepted"


def test_unparsed_lines_and_unknown_returns_make_accounting_incomplete(tmp_path):
    path = trace_file(tmp_path, [
        "unrecognized header",
        "11  01:02:03.000001 ioctl(5, DRM_IOCTL_V3D_SUBMIT_CSD, 0x1) = ? ERESTARTSYS (restart)",
    ])
    result = trace_summary.summarize(path)
    assert result["parser_status"] == "INCOMPLETE"
    assert len(result["unparsed"]) == 1 and len(result["unparsed_results"]) == 1
    assert result["requests"]["DRM_IOCTL_V3D_SUBMIT_CSD"]["unresolved"] == 1


def test_argumentless_ioctl_and_negative_fd_discovery_probe(tmp_path):
    path = trace_file(tmp_path, [
        "11  01:02:03.000001 ioctl(3</source/script.py>, FIOCLEX) = 0",
        "11  01:02:03.000002 ioctl(-1, DRM_IOCTL_V3D_MMAP_BO or DRM_IOCTL_VC4_CREATE_BO, 0x1) = -1 EBADF (Bad file descriptor)",
    ])
    result = trace_summary.summarize(path)
    assert result["parser_status"] == "PASS"
    assert result["requests"]["FIOCLEX"]["accepted"] == 1
    assert result["unknown_ioctls"][0]["errno"] == "EBADF"
    assert result["unknown_ioctls"][0]["fd"] == -1
    assert not result["ioctl_classification_complete"]
