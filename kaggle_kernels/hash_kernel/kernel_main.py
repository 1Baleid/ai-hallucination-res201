"""res201-hash: yield table, dimension-key test and NIH perceptual hashes (private Kaggle CPU kernel).

Inputs, found by globbing under ``--input-root`` (default ``/kaggle/input``; no mount names are
hard-coded):

* RANZCR CLiP ``train.csv`` (the CSV whose header has StudyInstanceUID and PatientID) and the JPEGs in
  the ``train/`` folder next to it (competition ``ranzcr-clip-catheter-line-classification``);
* NIH ChestX-ray14 ``Data_Entry_2017*.csv`` and ``images_*/images/*.png`` (dataset
  ``nih-chest-xrays/data``; NIH Clinical Center; Wang et al., CVPR 2017; the NIH Box release).

Outputs, in ``--working`` (default ``/kaggle/working``):

* ``yield_table.csv``: aggregate RANZCR CLiP label counts (metric, value), from train.csv alone;
* ``dimkey_test.json``: header sizes of 200 JPEGs (evenly spaced over the sorted list) against the NIH
  original sizes: share found, candidates per size;
* ``nih_hashes_<folder>.csv``: nih_filename, folder, ph (256-bit pHash, hex), dh (256-bit dHash, hex);
  one file per NIH folder, written as soon as that folder finishes;
* ``run_summary.json``: status of every step and folder, counts, timings and library versions.

Every step and every NIH folder runs inside try/except and the script always exits with code 0, so a
Kaggle commit saves whatever finished. The yield table and the dimension-key test are aggregates; no
per-image RANZCR file is written here. Run locally against a fake tree with
``python kernel_main.py --input-root <dir> --working <dir> --temp <dir> --workers 1``.
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing
import os
import platform
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

# BUNDLE-STRIP-START
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
# BUNDLE-STRIP-END
from cxrdev.imghash import bits_to_hex, dhash_bits, load_gray, phash_bits
from cxrdev.nih import find_inputs, find_nih_image_folders, find_nih_meta_csv, load_nih_meta
from cxrdev.ranzcr import is_train_csv, presence, read_train_csv, yield_table

KERNEL_NAME = "res201-hash"
DIMKEY_SAMPLE = 200
HASH_SIZE = 16
HIGHFREQ_FACTOR = 4
FOLDERS_OVERRIDE = None  # build_kernels.py --hash-folders replaces this line with a list of folder names


# --------------------------------------------------------------------------- small helpers
def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _versions() -> dict:
    import scipy
    import PIL
    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "pillow": PIL.__version__, "scipy": scipy.__version__}


def _write_json(path: Path, obj: dict) -> None:
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=False, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _write_csv(path: Path, header: list, rows: list) -> None:
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)
    os.replace(tmp, path)


def _mp_context():
    methods = multiprocessing.get_all_start_methods()
    return multiprocessing.get_context("fork" if "fork" in methods else "spawn")


def _pool_map(func, items: list, workers: int, chunksize: int = 32, label: str = "") -> list:
    """Map ``func`` over ``items`` with a process pool (in-process when workers <= 1)."""
    out = []
    every = 2000
    if workers <= 1 or len(items) < 2:
        for i, item in enumerate(items, 1):
            out.append(func(item))
            if label and i % every == 0:
                print(f"{label}: {i}/{len(items)}", flush=True)
        return out
    with _mp_context().Pool(processes=int(workers)) as pool:
        for i, res in enumerate(pool.imap(func, items, chunksize=chunksize), 1):
            out.append(res)
            if label and i % every == 0:
                print(f"{label}: {i}/{len(items)}", flush=True)
    return out


def _systematic_sample(items: list, n: int) -> list:
    """``n`` items evenly spaced over ``items`` (all of them when there are at most ``n``)."""
    if len(items) <= n:
        return list(items)
    idx = np.unique(np.round(np.linspace(0, len(items) - 1, n)).astype(int))
    return [items[i] for i in idx]


def find_train_csv(input_root: Path) -> Path | None:
    """The RANZCR train.csv under ``input_root`` (header with StudyInstanceUID and PatientID)."""
    for p in find_inputs(input_root, "train.csv"):
        if is_train_csv(p):
            return p
    return None


# --------------------------------------------------------------------------- steps
def dimkey_test(jpeg_paths: list, nih_meta: pd.DataFrame, n: int = DIMKEY_SAMPLE) -> dict:
    """Compare JPEG header sizes with NIH original sizes (aggregate report, no file names)."""
    sample = _systematic_sample(sorted(jpeg_paths), n)
    sizes = []
    unreadable = 0
    for p in sample:
        try:
            with Image.open(p) as im:
                sizes.append((int(im.size[0]), int(im.size[1])))
        except Exception:
            unreadable += 1
    nih_sizes = Counter(zip(nih_meta["orig_w"].astype(int).tolist(), nih_meta["orig_h"].astype(int).tolist()))
    found = [s for s in sizes if s in nih_sizes]
    transposed = [s for s in sizes if s not in nih_sizes and (s[1], s[0]) in nih_sizes]
    cands = np.array([nih_sizes[s] for s in found], dtype=float)
    per_size = sorted(Counter(sizes).items(), key=lambda kv: (-kv[1], kv[0]))
    return {
        "n_jpegs_total": len(jpeg_paths),
        "n_sampled": len(sample),
        "sampling": "systematic: evenly spaced over the JPEG list sorted by file name",
        "n_read": len(sizes),
        "n_unreadable": unreadable,
        "n_found": len(found),
        "share_found": (len(found) / len(sizes)) if sizes else None,
        "n_found_transposed_only": len(transposed),
        "n_nih_images": int(len(nih_meta)),
        "n_nih_unique_sizes": len(nih_sizes),
        "candidates_per_found_jpeg": ({"min": float(cands.min()), "median": float(np.median(cands)),
                                       "mean": float(cands.mean()), "max": float(cands.max())}
                                      if cands.size else None),
        "per_size": [{"jpg_w": w, "jpg_h": h, "n_jpegs": c, "in_nih": (w, h) in nih_sizes,
                      "n_nih_candidates": int(nih_sizes.get((w, h), 0))} for (w, h), c in per_size],
        "nih_largest_buckets": [{"orig_w": w, "orig_h": h, "n_nih": c}
                                for (w, h), c in sorted(nih_sizes.items(), key=lambda kv: (-kv[1], kv[0]))[:10]],
    }


def hash_one(path_str: str) -> tuple:
    """(nih_filename, pHash hex, dHash hex, error) for one NIH PNG; errors are returned, not raised."""
    name = os.path.basename(path_str)
    try:
        gray = load_gray(path_str)
        ph = bits_to_hex(phash_bits(gray, HASH_SIZE, HIGHFREQ_FACTOR))
        dh = bits_to_hex(dhash_bits(gray, HASH_SIZE))
        return name, ph, dh, ""
    except Exception as exc:
        return name, "", "", f"{type(exc).__name__}: {exc}"


def hash_folder(name: str, folder: Path, workers: int) -> tuple[list, list]:
    """Hash every PNG of one NIH folder; returns (rows for the CSV, per-image errors)."""
    paths = sorted(str(p) for p in Path(folder).glob("*.png"))
    results = _pool_map(hash_one, paths, workers, label=f"hash {name}")
    rows = [(r[0], name, r[1], r[2]) for r in results if not r[3]]
    errors = [(r[0], r[3]) for r in results if r[3]]
    return rows, errors


def main(input_root: str = "/kaggle/input", working: str = "/kaggle/working", temp: str = "/kaggle/temp",
         workers: int = 4, folders: list | None = None) -> dict:
    """Run every step; returns the run summary (also written to run_summary.json)."""
    input_root, working, temp = Path(input_root), Path(working), Path(temp)
    working.mkdir(parents=True, exist_ok=True)
    try:
        temp.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    if folders is None and FOLDERS_OVERRIDE is not None:
        folders = list(FOLDERS_OVERRIDE)
    summary: dict = {"kernel": KERNEL_NAME, "started_utc": _utc(), "finished_utc": None, "workers": int(workers),
                     "versions": _versions(), "inputs": {}, "steps": {}, "folders": {}, "n_hashed_total": 0}
    out_summary = working / "run_summary.json"
    t_start = time.time()

    def step(name: str, fn) -> None:
        t0 = time.time()
        try:
            info = fn() or {}
            summary["steps"][name] = {"status": "done", "seconds": round(time.time() - t0, 1), **info}
        except Exception as exc:
            summary["steps"][name] = {"status": "failed", "seconds": round(time.time() - t0, 1),
                                      "error": f"{type(exc).__name__}: {exc}",
                                      "traceback": traceback.format_exc()[-2000:]}
            print(f"[{KERNEL_NAME}] step {name} failed: {exc}", flush=True)
        _write_json(out_summary, summary)

    state: dict = {}

    def discover() -> dict:
        train_csv = find_train_csv(input_root)
        meta_csv = find_nih_meta_csv(input_root)
        nih_folders = find_nih_image_folders(input_root)
        state.update(train_csv=train_csv, meta_csv=meta_csv, nih_folders=nih_folders)
        jpegs = sorted((train_csv.parent / "train").glob("*.jpg")) if train_csv else []
        state["jpegs"] = jpegs
        summary["inputs"] = {"train_csv": str(train_csv) if train_csv else None,
                             "nih_meta_csv": str(meta_csv) if meta_csv else None,
                             "n_ranzcr_jpegs": len(jpegs),
                             "nih_folders": [n for n, _ in nih_folders]}
        return {}

    def do_yield() -> dict:
        if state.get("train_csv") is None:
            raise FileNotFoundError("RANZCR train.csv not found under the input root")
        yt = yield_table(presence(read_train_csv(state["train_csv"])))
        yt.to_csv(working / "yield_table.csv", index=False, lineterminator="\n")
        return {"n_metrics": int(len(yt))}

    def do_dimkey() -> dict:
        if state.get("meta_csv") is None:
            raise FileNotFoundError("NIH Data_Entry_2017*.csv not found under the input root")
        if not state.get("jpegs"):
            raise FileNotFoundError("no RANZCR JPEGs found in the train/ folder next to train.csv")
        meta = load_nih_meta(state["meta_csv"])
        report = dimkey_test(state["jpegs"], meta)
        _write_json(working / "dimkey_test.json", report)
        return {"share_found": report["share_found"], "n_sampled": report["n_sampled"]}

    step("discover", discover)
    step("yield_table", do_yield)
    step("dimkey_test", do_dimkey)

    for name, folder in state.get("nih_folders", []):
        if folders is not None and name not in folders:
            summary["folders"][name] = {"status": "not_selected"}
            continue
        out = working / f"nih_hashes_{name}.csv"
        if out.exists():
            summary["folders"][name] = {"status": "skipped_existing"}
            _write_json(out_summary, summary)
            continue
        t0 = time.time()
        try:
            rows, errors = hash_folder(name, folder, workers)
            _write_csv(out, ["nih_filename", "folder", "ph", "dh"], rows)
            summary["folders"][name] = {"status": "done", "n_hashed": len(rows), "n_errors": len(errors),
                                        "errors_head": errors[:10], "seconds": round(time.time() - t0, 1)}
            summary["n_hashed_total"] += len(rows)
            print(f"[{KERNEL_NAME}] {name}: {len(rows)} hashed, {len(errors)} errors, "
                  f"{time.time() - t0:.0f} s", flush=True)
        except Exception as exc:
            summary["folders"][name] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                                        "traceback": traceback.format_exc()[-2000:],
                                        "seconds": round(time.time() - t0, 1)}
            print(f"[{KERNEL_NAME}] {name} failed: {exc}", flush=True)
        _write_json(out_summary, summary)
    if folders is not None:
        for name in folders:
            if name not in summary["folders"]:
                summary["folders"][name] = {"status": "not_found"}
    summary["finished_utc"] = _utc()
    summary["seconds_total"] = round(time.time() - t_start, 1)
    _write_json(out_summary, summary)
    return summary


def main_cli(argv: list | None = None) -> int:
    """Command-line entry point; always returns 0 (the Kaggle commit must save partial outputs)."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--input-root", default="/kaggle/input")
    ap.add_argument("--working", default="/kaggle/working")
    ap.add_argument("--temp", default="/kaggle/temp")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--folders", default=None, help="comma-separated NIH folder names to hash (default: all)")
    args, _unknown = ap.parse_known_args(argv)
    folders = [f.strip() for f in args.folders.split(",") if f.strip()] if args.folders else None
    try:
        summary = main(args.input_root, args.working, args.temp, args.workers, folders)
        done = sum(1 for v in summary["folders"].values() if v.get("status") == "done")
        print(f"[{KERNEL_NAME}] finished: {done} folders hashed, {summary['n_hashed_total']} images", flush=True)
    except Exception as exc:  # never fail the commit
        print(f"[{KERNEL_NAME}] fatal error: {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        try:
            Path(args.working).mkdir(parents=True, exist_ok=True)
            _write_json(Path(args.working) / "run_summary.json",
                        {"kernel": KERNEL_NAME, "fatal_error": f"{type(exc).__name__}: {exc}",
                         "traceback": traceback.format_exc()[-4000:], "finished_utc": _utc()})
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main_cli())
