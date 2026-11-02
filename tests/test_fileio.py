"""Tests for cxrdev.fileio (JSONL resume, deterministic CSV, hashing, column checks, time)."""
from __future__ import annotations

import hashlib
import math
import re
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cxrdev import fileio

ROOT = Path(__file__).resolve().parents[1]


def _record(item_id: str, prompt_id: str, condition: str = "image", **extra) -> dict:
    rec = {"run_id": "synthetic", "item_id": item_id, "prompt_id": prompt_id, "condition": condition,
           "raw_output": "Yes", "parsed": "yes", "z": 1.25, "yes_ids": [9454, 7414]}
    rec.update(extra)
    return rec


# ----------------------------------------------------------------------------- hashing
def test_sha256_known_vectors_and_file_agreement(tmp_path):
    assert fileio.sha256_bytes(b"") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    assert fileio.sha256_bytes(b"abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    payload = bytes(range(256)) * 9000  # > 2 MiB: several read chunks
    path = tmp_path / "blob.bin"
    path.write_bytes(payload)
    assert fileio.sha256_file(path) == fileio.sha256_bytes(payload) == hashlib.sha256(payload).hexdigest()
    assert fileio.sha256_file(str(path)) == fileio.sha256_bytes(bytearray(payload))
    empty = tmp_path / "empty.bin"
    empty.write_bytes(b"")
    assert fileio.sha256_file(empty) == fileio.sha256_bytes(b"")
    with pytest.raises(TypeError):
        fileio.sha256_bytes("abc")


# ----------------------------------------------------------------------------- JSONL
def test_jsonl_round_trip_unicode_and_special_values(tmp_path):
    path = tmp_path / "runs" / "sub" / "run.jsonl"  # parent directories are created
    records = [
        _record("a1b2c3d4e5", "P0", raw_output="I don’t know — café 中文 \U0001F600"),
        _record("a1b2c3d4e5", "P1", raw_output="line1\nline2\r\n\ttab \"quoted\" \\ backslash"),
        _record("BLIND-TEXT", "P2hi", condition="BLIND-text", raw_output="sep para end",
                logp_abstain=float("nan"), z=float("-inf"), ended_with_eos=True, nih_filename=""),
    ]
    for rec in records:
        fileio.append_jsonl(path, rec)
    raw = path.read_bytes()
    assert raw.count(b"\n") == 3 and raw.endswith(b"\n")       # one line per record
    assert "café".encode("utf-8") in raw                   # UTF-8 written as is, not \u escapes
    back = fileio.read_jsonl(path)
    assert len(back) == 3
    assert back[0] == records[0] and back[1] == records[1]
    assert back[2]["raw_output"] == "sep para end"    # U+2028 does not split the line
    assert math.isnan(back[2]["logp_abstain"]) and back[2]["z"] == -math.inf
    assert back[2]["ended_with_eos"] is True


def test_append_jsonl_serialises_numpy_values(tmp_path):
    path = tmp_path / "np.jsonl"
    fileio.append_jsonl(path, {"n": np.int64(16), "f": np.float32(0.5), "b": np.bool_(False),
                               "ids": np.array([1, 2, 3]), "p": Path("x") / "y.png"})
    back = fileio.read_jsonl(path)[0]
    assert back == {"n": 16, "f": 0.5, "b": False, "ids": [1, 2, 3], "p": str(Path("x") / "y.png")}
    assert type(back["n"]) is int and type(back["b"]) is bool
    with pytest.raises(TypeError):
        fileio.append_jsonl(path, {"bad": object()})
    with pytest.raises(TypeError):
        fileio.append_jsonl(path, ["not", "a", "dict"])


def test_done_keys_for_resume(tmp_path):
    path = tmp_path / "run.jsonl"
    keys = ["item_id", "prompt_id", "condition"]
    assert fileio.done_keys(path, keys) == set()  # missing file: nothing done
    fileio.append_jsonl(path, _record("i1", "P0"))
    fileio.append_jsonl(path, _record("i1", "P1"))
    fileio.append_jsonl(path, _record("BLIND-GREY", "P0", condition="BLIND-grey"))
    fileio.append_jsonl(path, _record("i1", "P0"))  # a duplicate (re-run) collapses in the set
    done = fileio.done_keys(path, keys)
    assert done == {("i1", "P0", "image"), ("i1", "P1", "image"), ("BLIND-GREY", "P0", "BLIND-grey")}
    pending = [(i, p, "image") for i in ("i1", "i2") for p in ("P0", "P1", "P2hi")]
    assert [k for k in pending if k not in done] == [("i1", "P2hi", "image"), ("i2", "P0", "image"),
                                                 ("i2", "P1", "image"), ("i2", "P2hi", "image")]
    assert fileio.done_keys(path, ["item_id"]) == {("i1",), ("BLIND-GREY",)}
    with pytest.raises(ValueError, match="seed"):
        fileio.done_keys(path, ["item_id", "seed"])
    with pytest.raises(TypeError):
        fileio.done_keys(path, "item_id")


def test_partial_last_line_is_skipped_and_repaired_on_append(tmp_path):
    path = tmp_path / "crashed.jsonl"
    fileio.append_jsonl(path, _record("i1", "P0"))
    with open(path, "ab") as fh:
        fh.write(b'{"run_id": "synthetic", "item_id": "i2", "prom')  # write cut short by a crash
    with pytest.warns(RuntimeWarning, match="line numbers \\[2\\]"):
        recs = fileio.read_jsonl(path)
    assert [r["item_id"] for r in recs] == ["i1"]
    with pytest.raises(ValueError):
        fileio.read_jsonl(path, strict=True)
    with pytest.warns(RuntimeWarning):
        assert fileio.done_keys(path, ["item_id", "prompt_id"]) == {("i1", "P0")}
    # resume: the next append starts on a fresh line instead of extending the fragment
    fileio.append_jsonl(path, _record("i2", "P0"))
    with pytest.warns(RuntimeWarning):
        recs = fileio.read_jsonl(path)
    assert [(r["item_id"], r["prompt_id"]) for r in recs] == [("i1", "P0"), ("i2", "P0")]


def test_read_jsonl_tolerates_blank_lines_crlf_and_bom(tmp_path):
    path = tmp_path / "edited.jsonl"
    path.write_bytes(b'\xef\xbb\xbf{"a": 1}\r\n\r\n   \n{"a": 2}\n')
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no malformed-line warning expected
        assert fileio.read_jsonl(path) == [{"a": 1}, {"a": 2}]
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"a": 1}\n[1, 2]\n"text"\n{"b": 2}\n', encoding="utf-8")
    with pytest.warns(RuntimeWarning, match="2 malformed"):
        assert fileio.read_jsonl(bad) == [{"a": 1}, {"b": 2}]


# ----------------------------------------------------------------------------- CSV
def _frame() -> pd.DataFrame:
    return pd.DataFrame({
        "item_id": ["0f3a9c21be", "7d1e44a0c2", "c3b2a1f0e9"],
        "nih_filename": ["00000001_000.png", "00000002_001.png", "00000003_000.png"],
        "z": [0.1 + 0.2, -1.5, float("nan")],
        "n": [1, 2, 3],
        "flag": [True, False, True],
        "note": ["café", "comma, inside", 'quote " inside'],
    })


def test_write_csv_deterministic_bytes(tmp_path):
    df = _frame()
    p1, p2 = tmp_path / "a.csv", tmp_path / "nested" / "b.csv"
    h1 = fileio.write_csv(df, p1)
    h2 = fileio.write_csv(df.copy(), p2)
    b1, b2 = p1.read_bytes(), p2.read_bytes()
    assert b1 == b2 and h1 == h2 == fileio.sha256_file(p1)
    assert b"\r\n" not in b1 and b1.endswith(b"\n")
    assert b1.splitlines()[0] == b"item_id,nih_filename,z,n,flag,note"   # frame order, no index
    assert "café".encode("utf-8") in b1                              # UTF-8, no BOM
    assert not b1.startswith(b"\xef\xbb\xbf")
    back = pd.read_csv(p1)
    assert list(back.columns) == list(df.columns)
    assert back["note"].tolist() == df["note"].tolist()
    assert back["z"].iloc[0] == df["z"].iloc[0]                           # float round trip
    assert not list(tmp_path.glob("*.tmp")) and not list((tmp_path / "nested").glob("*.tmp"))
    # overwriting replaces the file; a different column order gives different bytes
    fileio.write_csv(df[list(reversed(df.columns))], p1)
    assert p1.read_bytes() != b2


def test_write_csv_respects_given_column_order(tmp_path):
    df = _frame()[["note", "item_id", "n"]]
    path = tmp_path / "order.csv"
    fileio.write_csv(df, path)
    assert path.read_text(encoding="utf-8").splitlines()[0] == "note,item_id,n"


def test_require_columns(tmp_path):
    df = _frame()
    fileio.require_columns(df, ["item_id", "nih_filename"], "pod_items.csv")  # no error
    with pytest.raises(ValueError) as exc:
        fileio.require_columns(df, ["item_id", "prompt_set", "smoke", "prompt_set"], "pod_items.csv")
    msg = str(exc.value)
    assert "pod_items.csv" in msg and "prompt_set" in msg and "smoke" in msg
    assert "item_id" not in msg.split(":", 1)[1]
    with pytest.raises(ValueError, match="arm"):
        fileio.require_columns(df, "arm", "items.csv")


# ----------------------------------------------------------------------------- time
def test_utc_now_format():
    stamp = fileio.utc_now()
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", stamp)
    parsed = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    assert parsed.tzinfo is not None
    assert abs((datetime.now(timezone.utc) - parsed).total_seconds()) < 60


# ----------------------------------------------------------------------------- pod constraint
def test_pod_modules_import_without_pandas_or_scipy():
    """prompts, parse, logits and the fileio JSONL helpers must work on a pod without pandas/scipy."""
    code = (
        "import sys\n"
        "class Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] in ('pandas', 'scipy', 'sklearn', 'matplotlib'):\n"
        "            raise ImportError('blocked ' + name)\n"
        "sys.meta_path.insert(0, Block())\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "from cxrdev import prompts, parse, logits, fileio, constants\n"
        "assert parse.parse_answer('Yes') == 'yes'\n"
        "assert prompts.prompt_text('P0').startswith('Is an endotracheal tube')\n"
        "import numpy as np\n"
        "sig = logits.first_token_signal(logits.log_softmax(np.zeros(4)), [0], [1])\n"
        "assert abs(sig['coverage'] - 0.5) < 1e-12\n"
        "print('ok')\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "ok"
