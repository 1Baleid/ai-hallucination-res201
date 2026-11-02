"""res201-match: RANZCR-to-NIH image matching (private Kaggle CPU kernel).

The only linkage is between two public de-identified releases of the same radiographs, matched by
image content; no attempt is made to identify any person (PROTOCOL sections 1 and 9).

Inputs, found by globbing under ``--input-root`` (default ``/kaggle/input``):

* RANZCR CLiP ``train.csv`` (StudyInstanceUID, PatientID) and the ``train/*.jpg`` JPEGs next to it;
* NIH ChestX-ray14 ``Data_Entry_2017*.csv`` and the PNGs (for the NCC confirmation and the spot check);
* the res201-hash output: ``nih_hashes_*.csv`` (and ``dimkey_test.json`` when present).

RANZCR perceptual hashes and thumbnails are computed in this session and kept in memory only; they
are never written to ``--working`` (the RANZCR hash shard never leaves Kaggle).

Outputs, in ``--working`` (default ``/kaggle/working``):

* ``match_table.csv``: schema S3, one row per RANZCR image (the mapping working copy: it is
  downloaded only to the git-ignored ``data/`` folder on the student's machine);
* ``match_stats.json``: calibration report (histograms, counts per status and route, match rate,
  ambiguous count and within-patient share, cut used), configuration and run information;
* ``calibration.png``: histograms of d1 and of the margin d2 - d1 with the cut and the margin rule;
* ``spotcheck_01.png`` .. ``spotcheck_06.png``: 60 random matched pairs (seed
  ``derive_seed("spotcheck")``), 10 per grid, RANZCR JPEG left and NIH PNG right at 256 px.

Run locally against a fake tree with
``python kernel_main.py --input-root <dir> --working <dir> --temp <dir> --workers 1``.
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import platform
import sys
import time
from collections import OrderedDict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

# BUNDLE-STRIP-START
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
# BUNDLE-STRIP-END
from cxrdev.constants import RANZCR_PATIENT, RANZCR_UID, derive_seed
from cxrdev.imghash import bits_to_hex, load_gray, ncc, phash_bits
from cxrdev.matching import MATCH_COLUMNS, MatchConfig, match_all
from cxrdev.nih import find_inputs, find_nih_meta_csv, load_nih_meta, locate_all_images
from cxrdev.ranzcr import is_train_csv

KERNEL_NAME = "res201-match"
THUMB_SIDE = 128
SPOT_SIDE = 256
SPOT_PAIRS = 60
SPOT_PER_GRID = 10
# Below this share of sampled JPEG sizes found among the NIH original sizes the dimension key is judged
# unreliable and every image is matched globally. Above it, a JPEG whose size has no NIH bucket is still
# matched globally (per-row fallback in match_all).
DIMKEY_MIN_SHARE = 0.5
NIH_THUMB_CACHE = 20000
CONFIG_OVERRIDES = {}  # build_kernels.py --match-config replaces this line with a dict of MatchConfig fields


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
    tmp.write_text(json.dumps(obj, indent=2, default=str) + "\n", encoding="utf-8")
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


def find_train_csv(input_root: Path) -> Path | None:
    """The RANZCR train.csv under ``input_root`` (header with StudyInstanceUID and PatientID)."""
    for p in find_inputs(input_root, "train.csv"):
        if is_train_csv(p):
            return p
    return None


# --------------------------------------------------------------------------- RANZCR side (memory only)
def ranzcr_one(path_str: str) -> tuple:
    """(uid, jpg_w, jpg_h, pHash hex, 128-px thumbnail bytes, error) for one RANZCR JPEG.

    The JPEG header gives the size; the pixels are decoded once at reduced scale (JPEG draft mode,
    never below 128 px) and both the hash and the thumbnail come from that decode.
    """
    uid = Path(path_str).stem
    try:
        with Image.open(path_str) as im:
            w, h = int(im.size[0]), int(im.size[1])
            if im.format == "JPEG":
                im.draft("L", (THUMB_SIDE, THUMB_SIDE))
            im.load()
            gray = np.array(im.convert("L"), dtype=np.uint8)
        ph = bits_to_hex(phash_bits(gray))
        th = load_gray(gray, size=THUMB_SIDE)
        return uid, w, h, ph, th.tobytes(), ""
    except Exception as exc:
        return uid, -1, -1, "", b"", f"{type(exc).__name__}: {exc}"


class NccScorer:
    """``ncc_fn(uid, nih_filename)`` for ``match_all``: 128-px NCC of a RANZCR JPEG and an NIH PNG."""

    def __init__(self, ranzcr_thumbs: dict, nih_paths: dict, cache_size: int = NIH_THUMB_CACHE) -> None:
        self.ranzcr_thumbs = ranzcr_thumbs
        self.nih_paths = nih_paths
        self.cache: OrderedDict = OrderedDict()
        self.cache_size = int(cache_size)
        self.n_nih_loaded = 0
        self.n_nih_failed = 0

    def nih_thumb(self, name: str):
        if name in self.cache:
            self.cache.move_to_end(name)
            return self.cache[name]
        path = self.nih_paths.get(name)
        th = None
        if path is not None:
            try:
                th = load_gray(path, size=THUMB_SIDE)
                self.n_nih_loaded += 1
            except Exception:
                self.n_nih_failed += 1
        self.cache[name] = th
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return th

    def __call__(self, uid: str, nih_filename: str) -> float:
        a = self.ranzcr_thumbs.get(uid)
        b = self.nih_thumb(nih_filename)
        if a is None or b is None:
            return float("nan")
        return ncc(a, b)


# --------------------------------------------------------------------------- figures
def plot_calibration(stats: dict, margin_bits: int, out_path: Path) -> None:
    """Histograms of d1 (with the cut) and of d2 - d1 (with the margin rule)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    d1 = np.asarray(stats.get("d1_hist", []), dtype=float)
    ax = axes[0]
    if d1.size:
        ax.bar(np.arange(d1.size), d1, width=1.0, color="#4c72b0")
        ax.set_yscale("symlog", linthresh=1)
    if stats.get("cut") is not None:
        ax.axvline(stats["cut"] + 0.5, color="#c44e52", linestyle="--", label=f"cut = {stats['cut']}")
        ax.legend()
    ax.set_xlabel("d1: Hamming distance to the closest NIH candidate (bits of %d)" % stats.get("n_bits", 256))
    ax.set_ylabel("RANZCR images")
    ax.set_title("Best-candidate distance")
    ax = axes[1]
    mh = stats.get("margin_hist", {})
    counts = np.asarray(mh.get("counts", []), dtype=float)
    if counts.size:
        ax.bar(mh.get("lo", 0) + np.arange(counts.size), counts, width=1.0, color="#55a868")
        ax.set_yscale("symlog", linthresh=1)
    ax.axvline(margin_bits - 0.5, color="#c44e52", linestyle="--", label=f"margin = {margin_bits}")
    ax.legend()
    ax.set_xlabel("d2 - d1 (bits)")
    ax.set_title("Margin to the second candidate")
    sc = stats.get("status_counts", {})
    fig.suptitle("res201-match calibration: " + ", ".join(f"{k} {v}" for k, v in sc.items()), fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def render_spotcheck(pairs: list, out_path: Path, side: int = SPOT_SIDE) -> None:
    """Grid of (RANZCR JPEG, NIH PNG) pairs at ``side`` px, JPEG left and PNG right, one pair per row."""
    label_h, gap = 16, 8
    width = 2 * side + gap
    height = len(pairs) * (side + label_h)
    canvas = Image.new("L", (width, max(height, 1)), color=255)
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for i, p in enumerate(pairs):
        y = i * (side + label_h)
        draw.text((4, y + 2), p["label"], fill=0, font=font)
        for x, src in ((0, p["jpeg"]), (side + gap, p["png"])):
            try:
                tile = Image.fromarray(load_gray(src, size=side, draft=True))
            except Exception:
                tile = Image.new("L", (side, side), color=128)
            canvas.paste(tile, (x, y + label_h))
    canvas.save(out_path, format="PNG")


def _match_config(dimkey: dict | None) -> tuple[MatchConfig, str]:
    cfg = MatchConfig()
    reason = "default (no dimkey_test.json)"
    if dimkey is not None and dimkey.get("share_found") is not None:
        share = float(dimkey["share_found"])
        cfg.use_dimension_key = share >= DIMKEY_MIN_SHARE
        reason = f"dimkey share_found = {share:.3f} ({'>=' if cfg.use_dimension_key else '<'} {DIMKEY_MIN_SHARE})"
    for key, value in dict(CONFIG_OVERRIDES).items():
        if not hasattr(cfg, key):
            raise ValueError(f"unknown MatchConfig field in CONFIG_OVERRIDES: {key}")
        setattr(cfg, key, value)
        reason += f"; override {key} = {value!r}"
    return cfg, reason


# --------------------------------------------------------------------------- main
def main(input_root: str = "/kaggle/input", working: str = "/kaggle/working", temp: str = "/kaggle/temp",
         workers: int = 4) -> dict:
    """Hash RANZCR JPEGs in memory, match them to NIH, write the outputs; returns the stats dict."""
    t_start = time.time()
    input_root, working, temp = Path(input_root), Path(working), Path(temp)
    working.mkdir(parents=True, exist_ok=True)
    try:
        temp.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    train_csv = find_train_csv(input_root)
    if train_csv is None:
        raise FileNotFoundError("RANZCR train.csv not found under the input root")
    meta_csv = find_nih_meta_csv(input_root)
    if meta_csv is None:
        raise FileNotFoundError("NIH Data_Entry_2017*.csv not found under the input root")
    shards = [p for p in find_inputs(input_root, "nih_hashes_*.csv")]
    if not shards:
        raise FileNotFoundError("no nih_hashes_*.csv (res201-hash output) under the input root")
    dimkey_files = find_inputs(input_root, "dimkey_test.json")
    dimkey = json.loads(dimkey_files[0].read_text(encoding="utf-8")) if dimkey_files else None

    train = pd.read_csv(train_csv, usecols=[RANZCR_UID, RANZCR_PATIENT], dtype=str, keep_default_na=False)
    patient_of = dict(zip(train[RANZCR_UID].str.strip(), train[RANZCR_PATIENT].str.strip()))
    jpeg_dir = train_csv.parent / "train"
    jpegs = {p.stem: p for p in sorted(jpeg_dir.glob("*.jpg"))}
    uids = sorted(u for u in jpegs if u in patient_of)
    print(f"[{KERNEL_NAME}] {len(train)} train.csv rows, {len(jpegs)} JPEGs, {len(uids)} with labels", flush=True)

    meta = load_nih_meta(meta_csv)
    nih_hashes = pd.concat([pd.read_csv(p, dtype=str, keep_default_na=False) for p in shards], ignore_index=True)
    nih_hashes = nih_hashes.drop_duplicates("nih_filename", keep="first")
    nih_paths = locate_all_images(input_root)
    print(f"[{KERNEL_NAME}] {len(nih_hashes)} NIH hashes from {len(shards)} shards, {len(nih_paths)} NIH PNGs",
          flush=True)

    t0 = time.time()
    results = _pool_map(ranzcr_one, [str(jpegs[u]) for u in uids], workers, label="ranzcr hash")
    ok = [r for r in results if not r[5]]
    errors = [(r[0], r[5]) for r in results if r[5]]
    thumbs = {r[0]: np.frombuffer(r[4], dtype=np.uint8).reshape(THUMB_SIDE, THUMB_SIDE) for r in ok}
    ranzcr_hashes = pd.DataFrame({
        "uid": [r[0] for r in ok], "ranzcr_patient_id": [patient_of[r[0]] for r in ok],
        "jpg_w": [r[1] for r in ok], "jpg_h": [r[2] for r in ok], "ph": [r[3] for r in ok],
    })
    del results
    t_hash = time.time() - t0

    cfg, cfg_reason = _match_config(dimkey)
    scorer = NccScorer(thumbs, nih_paths)
    t0 = time.time()
    table, stats = match_all(ranzcr_hashes, nih_hashes, meta, cfg, ncc_fn=scorer)
    t_match = time.time() - t0
    table = table.sort_values("uid", kind="stable").reset_index(drop=True)
    table.loc[:, list(MATCH_COLUMNS)].to_csv(working / "match_table.csv", index=False, lineterminator="\n")

    plot_calibration(stats, int(cfg.margin_bits), working / "calibration.png")

    matched = table[table["status"] == "matched"].reset_index(drop=True)
    rng = np.random.default_rng(derive_seed("spotcheck"))
    n_spot = min(SPOT_PAIRS, len(matched))
    pick = rng.choice(len(matched), size=n_spot, replace=False).tolist() if n_spot else []
    pairs = []
    for k, i in enumerate(pick, 1):
        row = matched.iloc[i]
        ncc1 = row["ncc1"]
        pairs.append({
            "jpeg": jpegs[row["uid"]], "png": nih_paths.get(row["nih_filename"], ""),
            "label": f"pair {k}: d1={int(row['d1'])} ncc={'nan' if not math.isfinite(ncc1) else format(ncc1, '.3f')} "
                     f"{row['route']}  NIH {row['nih_filename']}",
        })
    n_grids = 0
    for g in range(0, len(pairs), SPOT_PER_GRID):
        n_grids += 1
        render_spotcheck(pairs[g:g + SPOT_PER_GRID], working / f"spotcheck_{n_grids:02d}.png")

    stats["run"] = {
        "kernel": KERNEL_NAME, "finished_utc": _utc(), "versions": _versions(), "workers": int(workers),
        "config_reason": cfg_reason, "config_used": asdict(cfg),
        "n_train_rows": int(len(train)), "n_jpegs": len(jpegs), "n_jpegs_with_labels": len(uids),
        "n_labels_without_jpeg": int(len(set(patient_of) - set(jpegs))),
        "n_ranzcr_hashed": len(ok), "n_ranzcr_errors": len(errors), "ranzcr_errors_head": errors[:10],
        "n_nih_hash_shards": len(shards), "nih_hash_shards": sorted(p.name for p in shards),
        "n_nih_pngs_found": len(nih_paths), "n_nih_thumbs_loaded": scorer.n_nih_loaded,
        "n_nih_thumbs_failed": scorer.n_nih_failed, "dimkey_share_found": (dimkey or {}).get("share_found"),
        "n_spotcheck_pairs": len(pairs), "n_spotcheck_grids": n_grids, "spotcheck_seed": derive_seed("spotcheck"),
        "seconds_ranzcr_hashing": round(t_hash, 1), "seconds_matching": round(t_match, 1),
        "seconds_total": round(time.time() - t_start, 1),
    }
    _write_json(working / "match_stats.json", stats)
    print(f"[{KERNEL_NAME}] status counts: {stats['status_counts']}; cut {stats['cut']}; "
          f"match rate {stats['match_rate']}", flush=True)
    return stats


def main_cli(argv: list | None = None) -> int:
    """Command-line entry point (exit code 1 on a fatal error, so the Kaggle status shows it)."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--input-root", default="/kaggle/input")
    ap.add_argument("--working", default="/kaggle/working")
    ap.add_argument("--temp", default="/kaggle/temp")
    ap.add_argument("--workers", type=int, default=4)
    args, _unknown = ap.parse_known_args(argv)
    main(args.input_root, args.working, args.temp, args.workers)
    return 0


if __name__ == "__main__":
    sys.exit(main_cli())
