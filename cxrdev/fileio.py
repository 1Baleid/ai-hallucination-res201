"""File helpers: JSONL run logs with resume, deterministic CSV, SHA-256, column checks, UTC time.

The module imports only the standard library when loaded (numpy is imported lazily to
serialise numpy scalars, and ``write_csv`` / ``require_columns`` only call methods of the frame
they are given), so the pod runner can use the JSONL helpers on a machine without pandas.

JSONL conventions: one JSON object per line, UTF-8 (non-ASCII written as is), ``\\n`` line ends.
Non-finite floats are written as the tokens ``NaN`` / ``Infinity`` / ``-Infinity``, which
Python's json module reads back as floats (S7 ``logp_abstain`` is NaN when there are no
abstain ids).
"""
from __future__ import annotations

import codecs
import hashlib
import json
import os
import warnings
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path, PurePath
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

_CHUNK = 1 << 20


# ----------------------------------------------------------------------------- hashing
def sha256_bytes(b: bytes) -> str:
    """Hex SHA-256 of a bytes object."""
    if not isinstance(b, (bytes, bytearray, memoryview)):
        raise TypeError(f"expected bytes, got {type(b).__name__}")
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Hex SHA-256 of a file's bytes, read in 1 MiB chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ----------------------------------------------------------------------------- JSONL
def _json_default(obj: Any) -> Any:
    """json.dumps fallback for numpy scalars and arrays and for paths."""
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy is present wherever numpy objects exist
        np = None
    if np is not None:
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    if isinstance(obj, PurePath):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _ends_with_newline(path: Path) -> bool:
    """True when the file is missing, empty, or ends with a newline byte."""
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            if fh.tell() == 0:
                return True
            fh.seek(-1, os.SEEK_END)
            return fh.read(1) == b"\n"
    except FileNotFoundError:
        return True


def append_jsonl(path: str | Path, record: dict) -> None:
    """Append one record as a single UTF-8 JSON line, then flush and fsync.

    Creates the parent directory if needed. If the file does not end with a newline (a write
    interrupted by a crash), a newline is written first, so the new record never merges with
    the partial one; ``read_jsonl`` then skips the partial line with a warning.
    """
    if not isinstance(record, dict):
        raise TypeError(f"record must be a dict, got {type(record).__name__}")
    line = json.dumps(record, ensure_ascii=False, default=_json_default)
    data = (line + "\n").encode("utf-8")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    prefix = b"" if _ends_with_newline(p) else b"\n"
    with open(p, "ab") as fh:
        fh.write(prefix + data)
        fh.flush()
        os.fsync(fh.fileno())


def read_jsonl(path: str | Path, *, strict: bool = False) -> list[dict]:
    """Read a JSONL file into a list of dicts (blank lines ignored, a UTF-8 BOM tolerated).

    Lines are split on ``\\n`` only (so U+2028 inside a string is safe). A line that is not a
    JSON object, typically the partial last line of a write interrupted by a crash, is skipped
    with a RuntimeWarning naming its line number; with ``strict=True`` (optional extension) it
    raises ValueError instead.
    """
    raw = Path(path).read_bytes()
    if raw.startswith(codecs.BOM_UTF8):
        raw = raw[len(codecs.BOM_UTF8):]
    records: list[dict] = []
    bad: list[int] = []
    for lineno, line in enumerate(raw.split(b"\n"), start=1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            rec = None
        if isinstance(rec, dict):
            records.append(rec)
        else:
            bad.append(lineno)
    if bad:
        msg = f"{path}: {len(bad)} malformed line(s) skipped (line numbers {bad[:10]})"
        if strict:
            raise ValueError(msg)
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
    return records


def done_keys(path: str | Path, key_fields: Sequence[str]) -> set[tuple]:
    """Keys already present in a JSONL file, for resuming a run.

    Returns ``{tuple(record[f] for f in key_fields)}`` over the records (an empty set when the
    file does not exist). Raises ValueError when a record lacks a key field.
    """
    if isinstance(key_fields, str):
        raise TypeError("key_fields must be a sequence of field names, not a single string")
    fields = list(key_fields)
    if not fields:
        raise ValueError("key_fields must not be empty")
    p = Path(path)
    if not p.exists():
        return set()
    keys: set[tuple] = set()
    for i, rec in enumerate(read_jsonl(p), start=1):
        missing = [f for f in fields if f not in rec]
        if missing:
            raise ValueError(f"{p}: record {i} lacks key field(s) {missing}")
        keys.add(tuple(rec[f] for f in fields))
    return keys


# ----------------------------------------------------------------------------- CSV and columns
def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write bytes to a temporary file beside ``path``, fsync, then replace ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def write_csv(df: "pd.DataFrame", path: str | Path) -> str:
    """Write a frame as UTF-8 CSV without the index, ``\\n`` line endings, columns in frame order.

    The same frame always gives the same bytes. The file is replaced atomically and the parent
    directory is created if needed. Returns the hex SHA-256 of the written bytes.
    """
    data = df.to_csv(index=False, lineterminator="\n").encode("utf-8")
    _atomic_write_bytes(Path(path), data)
    return sha256_bytes(data)


def require_columns(df: "pd.DataFrame", columns: Sequence[str], name: str) -> None:
    """Raise ValueError naming every column of ``columns`` that ``df`` lacks (in the given order)."""
    if isinstance(columns, str):
        columns = [columns]
    present = set(df.columns)
    missing: list[str] = []
    for col in columns:
        if col not in present and col not in missing:
            missing.append(col)
    if missing:
        raise ValueError(f"{name} is missing required column(s): {', '.join(map(str, missing))}")


# ----------------------------------------------------------------------------- time
def utc_now() -> str:
    """Current UTC time as ISO 8601 with a "Z" suffix, to the second (e.g. 2026-11-02T10:15:30Z)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
