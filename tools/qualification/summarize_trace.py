#!/usr/bin/env python3
"""Account for an existing `strace -f -tt -yy -e trace=ioctl,openat` file.

The report describes accepted/rejected syscalls in this trace. It establishes
neither device completion nor arithmetic correctness, overlap, or performance.
Unknown requests remain visible; observing another engine is not itself a
failure or proof of a CPU arithmetic fallback.
"""
import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

ENGINES = (
    "DRM_IOCTL_V3D_SUBMIT_CSD", "DRM_IOCTL_V3D_SUBMIT_CL",
    "DRM_IOCTL_V3D_SUBMIT_TFU", "DRM_IOCTL_V3D_SUBMIT_CPU",
)
EXPORT = "DRM_IOCTL_SYNCOBJ_HANDLE_TO_FD"
IMPORT = "DRM_IOCTL_SYNCOBJ_FD_TO_HANDLE"
# Saved target drm.h: DRM_IOCTL_BASE='d'; DRM_IOWR(0xC1/0xC2,
# struct drm_syncobj_handle). The saved struct is 24 bytes (point included).
# Header SHA256: 9cc46d8e17a2b383bf0178dab54c7ffb19b58901cec7391b8bc08d6b025fa118.
# Decode only these exact tuples: older/different-sized structs stay unknown.
SYNCOBJ_CODES = {(0x64, 0xC1, 0x18): EXPORT, (0x64, 0xC2, 0x18): IMPORT}
PROVIDER_NAMES = (
    "mstar_pi/_C.", "libvulkan_broadcom.so", "libvulkan.so", "libdrm.so", "libggml",
)
LINE = re.compile(r"\s*(\d+)\s+(\S+)\s+(.*)")
CALL = re.compile(r"(\w+)\(")
RESUMED = re.compile(r"<\.\.\. (\w+) resumed>(.*)")
RESULT = re.compile(r"\)\s+=\s+(-?\d+)(?:<.*?>)?(?:\s+([A-Z][A-Z0-9_]+))?")
IOCTL = re.compile(r"ioctl\((-?\d+)(?:<.*?>)?,\s*(_IOC\([^)]*\)|[^,)]+)\s*(?:,|\))")
OPAQUE = re.compile(r"_IOC\(([^,]+),\s*(0x[0-9a-fA-F]+),\s*(0x[0-9a-fA-F]+),\s*(0x[0-9a-fA-F]+)\)")


def _requests():
    return {"calls": 0, "accepted": 0, "rejected": 0, "unresolved": 0}


def _outcome(row):
    value = row["return_value"]
    return "unresolved" if value is None else "accepted" if value >= 0 else "rejected"


def _request_name(raw):
    match = OPAQUE.fullmatch(raw)
    if match and set(match[1].replace(" ", "").split("|")) == {"_IOC_READ", "_IOC_WRITE"}:
        code = tuple(int(value, 16) for value in match.groups()[1:])
        return SYNCOBJ_CODES.get(code, raw)
    return raw


def summarize(trace):
    raw = trace.read_bytes()
    pending, complete = {}, []
    unmatched_unfinished, unmatched_resumed, unparsed = [], [], []
    events = Counter()
    for number, text in enumerate(raw.decode("utf-8").splitlines(), 1):
        match = LINE.fullmatch(text)
        if not match:
            unparsed.append({"line": number, "text": text})
            continue
        pid, stamp, body = match.groups()
        pid = int(pid)
        row = {"pid": pid, "start_line": number, "end_line": number,
               "start_time": stamp, "end_time": stamp, "body": body}
        if body.startswith(("--- ", "+++ ")):
            events["signal" if body.startswith("--- ") else "process_exit"] += 1
            continue
        if body.endswith("<unfinished ...>"):
            if pid in pending:
                unmatched_unfinished.append(pending[pid])
            row["body"] = body.removesuffix("<unfinished ...>")
            pending[pid] = row
            continue
        resumed = RESUMED.fullmatch(body)
        if resumed:
            prior = pending.get(pid)
            started = CALL.match(prior["body"]) if prior else None
            if not started or started[1] != resumed[1]:
                unmatched_resumed.append(row)
                continue
            pending.pop(pid)
            row = {**prior, "end_line": number, "end_time": stamp,
                   "body": prior["body"] + resumed[2]}
        call = CALL.match(row["body"])
        if not call:
            unparsed.append(row)
            continue
        # Return parsing is anchored to the last ") = ..." occurrence, so a
        # decoded pathname/argument cannot supply the result of the syscall.
        result = list(RESULT.finditer(row["body"]))
        row.update(syscall=call[1], return_value=int(result[-1][1]) if result else None,
                   errno=result[-1][2] if result else None)
        complete.append(row)
    unmatched_unfinished.extend(pending.values())

    requests = {name: _requests() for name in (*ENGINES, "SYNC_IOC_MERGE", EXPORT, IMPORT)}
    unknown_ioctls, provider_opens, rejected_requests = [], [], []
    unparsed_results, engine_groups = [], {}
    other_syscalls = Counter()
    for row in complete:
        if row["return_value"] is None:
            unparsed_results.append(row)
        if row["syscall"] == "ioctl":
            match = IOCTL.match(row["body"])
            if not match:
                unparsed.append(row)
                continue
            fd, raw_request = match.groups()
            name = _request_name(raw_request.strip())
            row.update(fd=int(fd), request=name)
            counts = requests.setdefault(name, _requests())
            counts["calls"] += 1
            counts[_outcome(row)] += 1
            if not re.fullmatch(r"[A-Z][A-Z0-9_]+", name):
                unknown_ioctls.append(row)
            if _outcome(row) == "rejected":
                rejected_requests.append(row)
            if name in ENGINES:
                group = engine_groups.setdefault((row["pid"], int(fd), name), _requests())
                group["calls"] += 1
                group[_outcome(row)] += 1
        elif row["syscall"] == "openat":
            if any(name in row["body"] for name in PROVIDER_NAMES):
                provider_opens.append({**row, "outcome": _outcome(row)})
        else:
            other_syscalls[row["syscall"]] += 1
    export_einval = [row for row in rejected_requests if row["request"] == EXPORT
                    and row["errno"] == "EINVAL"]
    accounting_complete = not (unmatched_unfinished or unmatched_resumed or unparsed or unparsed_results)
    return {
        "scope": __doc__.strip(),
        "trace": {"path": str(trace.resolve()), "sha256": hashlib.sha256(raw).hexdigest(),
                  "bytes": len(raw)},
        "parser_status": "PASS" if accounting_complete else "INCOMPLETE",
        "ioctl_classification_complete": not unknown_ioctls,
        "completed_syscalls": len(complete), "events": dict(events),
        "requests": requests,
        "engine_calls_by_pid_fd": [dict(pid=pid, fd=fd, engine=name, **counts)
                                   for (pid, fd, name), counts in sorted(engine_groups.items())],
        "engine_accounting": "Counts include unambiguous engine request names only. "
            "Opaque requests and strace's alternative-name lists remain in unknown_ioctls.",
        "provider_opens": provider_opens,
        "rejected_requests": rejected_requests,
        "sync_export_einval": {
            "count": len(export_einval),
            "interpretation": "Rejected sync-object exports, not successful dependencies. "
                "Earlier target receipts observed EINVAL during pending availability polling; "
                "this count alone does not establish the cause or eventual completion.",
        },
        "unknown_ioctls": unknown_ioctls,
        "unmatched_unfinished": unmatched_unfinished, "unmatched_resumed": unmatched_resumed,
        "unparsed": unparsed, "unparsed_results": unparsed_results,
        "other_syscalls": dict(other_syscalls),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = summarize(args.trace)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"parser_status": report["parser_status"], "report": str(args.output),
                      "trace_sha256": report["trace"]["sha256"],
                      "requests": report["requests"],
                      "unknown_ioctl_records": len(report["unknown_ioctls"])}))
    return 0 if report["parser_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
