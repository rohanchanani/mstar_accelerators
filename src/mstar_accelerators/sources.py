"""Load trusted acceleration source files in the process that consumes them.

A file exports ``acceleration``. Directory entries are named by their relative
paths, so copying a file creates a separate registration without editing it.
Files use ordinary absolute imports for shared code; they need no package or
catalog. Module-and-symbol registries remain another source form.
"""
from __future__ import annotations

import hashlib
import importlib
import sys
from pathlib import Path
from types import ModuleType


def normalize_source(source: str | Path, *, relative_to: Path | None = None) -> str:
    text = str(source)
    if ":" in text and not Path(text).exists():
        return text
    path = Path(text)
    if not path.is_absolute() and relative_to is not None:
        path = relative_to / path
    return str(path.resolve())


def acceleration_files(source: str | Path) -> dict[str, Path]:
    """The same filesystem identities used by loading and source receipts."""
    path = Path(normalize_source(source))
    if path.is_file():
        root, files = path.parent, [path]
    elif path.is_dir():
        root, files = path, sorted(path.rglob("*.py"))
    else:
        raise FileNotFoundError(path)
    return {file.relative_to(root).with_suffix("").as_posix(): file for file in files}


def load_accelerations(source: str | Path):
    source = normalize_source(source)
    if ":" in source and not Path(source).exists():
        module_name, symbol = source.split(":", 1)
        entries = getattr(importlib.import_module(module_name), symbol)
        return dict(entries.accelerations if hasattr(entries, "accelerations") else entries)
    entries = {}
    for name, file in acceleration_files(source).items():
        module_name = "_mstar_acceleration_" + hashlib.sha256(str(file).encode()).hexdigest()
        module = ModuleType(module_name)
        module.__file__ = str(file)
        module.__package__ = ""
        sys.modules[module_name] = module
        try:
            # Read the edited source directly: same-size edits in one timestamp
            # interval must not reuse Python's timestamp-based bytecode cache.
            exec(compile(file.read_bytes(), str(file), "exec"), module.__dict__)  # noqa: S102 -- trusted Python module
            entries[name] = module.acceleration
        except BaseException:
            sys.modules.pop(module_name, None)
            raise
    return entries
