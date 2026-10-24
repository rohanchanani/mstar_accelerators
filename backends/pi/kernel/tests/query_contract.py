#!/usr/bin/env python3
"""Read-only Pi/aarch64 V3D ABI gate; does not allocate or submit GPU work."""
from __future__ import annotations

import argparse
import errno
import fcntl
import json
import os
from pathlib import Path
import platform
import struct
import sys

# Linux generic ioctl encoding on the r2 arm64 Pi. drm_v3d_get_param is IIQ.
# DRM_IOWR(DRM_COMMAND_BASE + DRM_V3D_GET_PARAM, struct drm_v3d_get_param)
GET_PARAM = 0xC0106444
CONTRACT_PARAM = 0x50490001
STATUS_PARAM = 0x50490002
GLOBAL_RESET_PARAM = 15
CONTEXT_RESET_PARAM = 16


def get_param(fd: int, param: int) -> int:
    data = bytearray(struct.pack("=IIQ", param, 0, 0))
    fcntl.ioctl(fd, GET_PARAM, data, True)
    returned_param, pad, value = struct.unpack("=IIQ", data)
    if returned_param != param or pad:
        raise RuntimeError("Unexpected GET_PARAM reply layout")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("device", help="V3D render node, e.g. /dev/dri/renderD128")
    args = parser.parse_args()
    if platform.machine() != "aarch64" or sys.byteorder != "little":
        parser.error("This source-pinned ioctl probe is for little-endian aarch64 Pi Linux")
    node = Path(args.device).resolve(strict=True)
    driver = Path("/sys/class/drm") / node.name / "device/driver"
    if driver.resolve(strict=True).name != "v3d":
        parser.error(f"{node} is not backed by the v3d platform driver")
    fd = os.open(node, os.O_RDWR | os.O_CLOEXEC)
    try:
        try:
            contract = get_param(fd, CONTRACT_PARAM)
        except OSError as error:
            if error.errno == errno.EINVAL:
                print(json.dumps({"device": str(node), "accepted": False,
                                  "reason": "CSD completion contract absent"}))
                return 4
            raise
        if contract != 1:
            print(json.dumps({"device": str(node), "accepted": False,
                              "contract": contract, "reason": "Unsupported contract version"}))
            return 4
        status = get_param(fd, STATUS_PARAM)
        result = {"device": str(node), "contract": contract,
                  "fatal_errno": status,
                  "global_reset_counter": get_param(fd, GLOBAL_RESET_PARAM),
                  "context_reset_counter": get_param(fd, CONTEXT_RESET_PARAM),
                  "accepted": status == 0, "gpu_work_submitted": False}
        print(json.dumps(result, indent=2))
        return 0 if status == 0 else 5
    finally:
        os.close(fd)


if __name__ == "__main__":
    raise SystemExit(main())
