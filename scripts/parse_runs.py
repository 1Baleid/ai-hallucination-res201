"""Turn pod run logs (S7 JSONL) into ``parsed_results.csv`` (S8).

Every record is re-parsed with the current parser (``cxrdev.parse``), ``parser_version`` is
updated, ``truncated`` is added (no end-of-sequence within ``max_new_tokens``) and the token id
lists are dropped. Records are de-duplicated on (model_key, item_id, prompt_id, condition): the
last one read wins (files in sorted path order, lines in file order) and the number dropped is
reported. Malformed JSONL lines (for example a partial last line after a crash) are skipped with
a warning by ``fileio.read_jsonl``.

A run directory from ``pod/queue.sh`` also holds the smoke-test records (``smoke/<model>.smoke.jsonl``,
run id ``<run_id>-smoke``, always the first-token signal) and rotated earlier copies
(``*.prev.jsonl``). They repeat the dev-pool smoke items of the main run under the same key and,
read last, would replace the main run's records, so a directory scan skips them (they are listed in
the summary as ``skipped_files``). Name such a file explicitly, or pass ``--include-smoke``, to
parse it anyway.

Usage:
  python scripts/parse_runs.py --runs runs/<run_id> [runs/<run_id2> ...] --out runs/parsed_results.csv
                               [--summary runs/parse_summary.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from cxrdev import fileio, parse  # noqa: E402
from cxrdev.metrics import S7_FIELDS, S8_COLUMNS, coerce_parsed  # noqa: E402

KEY_FIELDS: tuple[str, ...] = ("model_key", "item_id", "prompt_id", "condition")


def is_side_file(path: Path, root: Path) -> bool:
    """True for a smoke-test or rotated JSONL file inside a run directory (skipped by a directory scan)."""
    rel = path.relative_to(root).parts if path.is_relative_to(root) else path.parts
    name = path.name.lower()
    return "smoke" in (part.lower() for part in rel[:-1]) or name.endswith(".smoke.jsonl") or name.endswith(".prev.jsonl")


def run_files(paths: Iterable[str | Path], include_smoke: bool = False,
              skipped: list[Path] | None = None) -> list[Path]:
    """The JSONL files named by ``paths`` (a file, or every ``*.jsonl`` under a directory), sorted, unique.

    A directory scan leaves out smoke-test and rotated files (``is_side_file``) unless
    ``include_smoke``; the files left out are appended to ``skipped`` when given.
    """
    found: list[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            for q in sorted(q for q in p.rglob("*.jsonl") if q.is_file()):
                if not include_smoke and is_side_file(q, p):
                    if skipped is not None:
                        skipped.append(q.resolve())
                    continue
                found.append(q)
        elif p.is_file():
            found.append(p)
        else:
            raise FileNotFoundError(f"run path not found: {p}")
    return list(dict.fromkeys(q.resolve() for q in found))


def parse_records(records: Sequence[dict[str, Any]], source: str = "records") -> tuple[pd.DataFrame, dict[str, Any]]:
    """S8 frame from S7 records, plus a summary (counts, parser changes, duplicates).

    Raises ValueError when a record lacks an S7 field (the message names the fields and the first
    record number).
    """
    rows: list[dict[str, Any]] = []
    changed = Counter()
    extra_fields: set[str] = set()
    for i, rec in enumerate(records, start=1):
        missing = [f for f in S7_FIELDS if f not in rec]
        if missing:
            raise ValueError(f"{source}: record {i} lacks S7 field(s) {missing}")
        extra_fields.update(set(rec) - set(S7_FIELDS))
        row = {f: rec[f] for f in S7_FIELDS if f not in ("yes_ids", "no_ids")}
        new = parse.parse_answer(rec["raw_output"])
        if str(rec["parsed"]) != new:
            changed[f"{rec['parsed']}->{new}"] += 1
        row["parsed"] = new
        row["parser_version"] = parse.PARSER_VERSION
        row["truncated"] = parse.is_truncated(int(rec["n_generated_tokens"]), int(rec["max_new_tokens"]), rec["ended_with_eos"])
        for f in ("logp_yes", "logp_no", "logp_abstain", "z", "coverage", "wall_s"):
            if row[f] is None:
                row[f] = float("nan")
        rows.append(row)
    df = pd.DataFrame(rows, columns=list(S8_COLUMNS))
    n_before = len(df)
    if n_before:
        df = df.drop_duplicates(subset=list(KEY_FIELDS), keep="last").reset_index(drop=True)
        df = coerce_parsed(df)
    summary = {
        "n_records": n_before,
        "n_duplicates_dropped": n_before - len(df),
        "n_rows": int(len(df)),
        "parser_version": parse.PARSER_VERSION,
        "parse_changes": dict(sorted(changed.items())),
        "extra_fields_dropped": sorted(extra_fields),
        "counts": {},
    }
    if len(df):
        grp = df.groupby(["model_key", "condition", "prompt_id"], sort=True)
        for (mk, cond, pid), g in grp:
            summary["counts"][f"{mk}|{cond}|{pid}"] = {
                "n": int(len(g)),
                "parsed": {k: int(v) for k, v in sorted(Counter(g["parsed"]).items())},
                "truncated": int(g["truncated"].sum()),
            }
    return df, summary


def parse_runs(run_paths: Iterable[str | Path], out: str | Path, summary_path: str | Path | None = None,
               include_smoke: bool = False) -> dict[str, Any]:
    """Read every JSONL run file, write S8 to ``out`` and return the summary (also written if asked)."""
    skipped: list[Path] = []
    files = run_files(run_paths, include_smoke=include_smoke, skipped=skipped)
    if not files:
        raise FileNotFoundError("no *.jsonl run files found")
    records: list[dict[str, Any]] = []
    for f in files:
        records.extend(fileio.read_jsonl(f))
    df, summary = parse_records(records, source=", ".join(str(f) for f in files))
    summary["files"] = [str(f) for f in files]
    summary["skipped_files"] = [str(f) for f in skipped]
    summary["sha256"] = fileio.write_csv(df, out)
    summary["out"] = str(out)
    if summary_path is not None:
        p = Path(summary_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="run directories or JSONL files")
    ap.add_argument("--out", required=True, help="output CSV (S8), e.g. runs/parsed_results.csv")
    ap.add_argument("--summary", default=None, help="optional JSON summary path")
    ap.add_argument("--include-smoke", action="store_true",
                    help="also parse smoke/ and *.prev.jsonl files found in a run directory")
    a = ap.parse_args(argv)
    s = parse_runs(a.runs, a.out, a.summary, include_smoke=a.include_smoke)
    print(f"{s['n_records']} records from {len(s['files'])} file(s); {s['n_duplicates_dropped']} duplicates dropped; "
          f"{s['n_rows']} rows written to {s['out']} (sha256 {s['sha256'][:12]})")
    if s["skipped_files"]:
        print(f"skipped {len(s['skipped_files'])} smoke-test or rotated file(s): "
              + ", ".join(Path(f).name for f in s["skipped_files"]))
    if s["parse_changes"]:
        print(f"parser v{s['parser_version']} changed answers: {s['parse_changes']}")
    for key, c in s["counts"].items():
        print(f"  {key}: n={c['n']} parsed={c['parsed']} truncated={c['truncated']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
