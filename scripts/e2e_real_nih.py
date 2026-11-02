"""End-to-end run of the Stage 1 pipeline on REAL public NIH ChestX-ray14 images (simulated RANZCR side).

What it does (every step uses the project's own code; nothing is mocked except the models and the
RANZCR CLiP competition data, which never exist on this machine):

1. Fake Kaggle input tree under ``<root>/kaggle_input`` (git-ignored ``runs/``):
   * NIH part: real PNGs from ``data/nih_box/extract/images`` (hard links, else copies) for about 400
     patients with several follow-ups each, split over ``images_001/images`` and ``images_002/images``,
     plus their real metadata rows written in the Kaggle split-header form
     (``OriginalImage[Width,Height]`` unquoted, trailing comma);
   * simulated RANZCR CLiP competition part: about 600 of those images converted the way RANZCR did
     (bicubic resize to the image's real original size from the metadata, JPEG quality 95) as
     ``train/<uid>.jpg``, plus 30 JPEGs made from real NIH images that are NOT in the NIH part (20 from
     patients outside it, 10 held-out follow-ups of patients inside it); a synthetic ``train.csv`` with
     the real RANZCR column names, new UIDs and PatientIDs (one per NIH patient) and plausible labels:
     ETT on about a third of the images in patient-level runs, central lines on about 97% of images (as
     in the real yield table, so very few no-line patients and the H1 substitute is exercised), NGT on
     about a quarter, a few Swan-Ganz catheters and a few patients with no line at all.
   The tree is cached: it is rebuilt only when its plan changes (or with ``--fresh``).
2. The BUNDLED Kaggle kernels (``kaggle_kernels/build_kernels.py``) run as isolated subprocesses
   (hash, then match with the hash output mounted as a kernel source, then export of the selected PNGs).
   Every JPEG whose source is in the NIH part must match exactly its source, the 30 absent-source JPEGs
   must stay unmatched, and no match may be wrong. Distances, the cut, NCC values and run times are
   reported and extrapolated to the Kaggle scale (112,120 PNGs, 30,083 JPEGs, 4 workers).
3. ``scripts/select_items.py`` on the kernel's ``match_table.csv``, the synthetic ``train.csv`` and the
   real v2020 NIH metadata, with scaled-down counts and ``--exclude-filenames`` (a probe list of 5 matched
   images plus one name outside the pool, as the registered run excludes the images seen in software
   tests); the S5/S6/S10 invariants and the exclusion are checked.
4. Model outputs WITHOUT a model: ``pod/smoke_test.py``, ``pod/run_vlm.py`` (items and BLIND rows) and
   ``pod/run_medsiglip.py`` run unchanged with deterministic fake backends (``FakeVLM*`` below) that
   see only pixels; their answers depend on the image content through a lookup built from the item
   list (a simulation device; a real model never sees labels).
5. ``scripts/parse_runs.py``, a simulated blind reading with ``tools/reading_core.ReadingSession``
   (tracheostomy flags on A images, a retest 49 h later), ``scripts/verify_join.py``,
   ``scripts/analyze.py --fast`` and ``scripts/build_report.py``; then leak checks (no RANZCR uid,
   PatientID, read id or item id in the results or the built report).

The summary goes to ``<root>/e2e_summary.json``. Usage, from the repository root::

    python scripts/e2e_real_nih.py [--root runs/e2e_real] [--fresh]

Data: NIH ChestX-ray14 (NIH Clinical Center; Wang et al., CVPR 2017; the NIH Box release), public,
attribution only. The RANZCR side is entirely synthetic: no RANZCR CLiP file is used or needed.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse  # noqa: E402
import hashlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import zipfile  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from dataclasses import asdict, dataclass, field  # noqa: E402
from datetime import datetime, timedelta, timezone  # noqa: E402
from types import SimpleNamespace  # noqa: E402
from typing import Any, Callable  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from cxrdev import constants as C  # noqa: E402
from cxrdev import fileio  # noqa: E402
from cxrdev.imghash import hamming  # noqa: E402
from cxrdev.prompts import PROMPT_IDS, prompt_text  # noqa: E402

KAGGLE_PNGS = 112_120
KAGGLE_JPEGS = 30_083
KAGGLE_WORKERS = 4
USERNAME = "e2euser"
RUN_ID = "e2e-core"
TRAIN_COLUMNS = [C.RANZCR_UID, *C.ETT_COLS, *C.NGT_COLS, *C.CVC_COLS, C.SWAN_COL, C.RANZCR_PATIENT]
KAGGLE_META_HEADER = ("Image Index,Finding Labels,Follow-up #,Patient ID,Patient Age,Patient Gender,View Position,"
                      "OriginalImage[Width,Height],OriginalImagePixelSpacing[x,y],")
PLAN_VERSION = "e2e-real-nih-v1"


@dataclass
class E2EConfig:
    """Sizes and paths of one end-to-end run (defaults: about 1,500 NIH PNGs and 630 JPEGs)."""

    root: Path = ROOT / "runs" / "e2e_real"
    nih_images: Path = ROOT / "data" / "nih_box" / "extract" / "images"
    nih_meta: Path = ROOT / "data" / "nih" / "Data_Entry_2017_v2020.csv"
    seed: int = C.SEED * 100 + 99  # harness seed (not a registered purpose)
    n_nih_patients: int = 400
    max_per_patient: int = 4
    n_ranzcr_patients: int = 150
    n_absent_outside: int = 20
    n_absent_heldout: int = 10
    n_noline_patients: int = 4
    jpeg_quality: int = 95
    jpeg_threads: int = 6
    # scaled-down selection (the registered defaults stay in cxrdev/constants.py)
    n_pairs: int = 40
    n_u_per_stratum: int = 20
    n_dev_pos: int = 15
    n_dev_neg: int = 15
    n_retest: int = 30
    n_smoke: int = 10
    fresh: bool = False
    log: Callable[[str], None] = field(default=print, repr=False)

    def plan_key(self) -> dict[str, Any]:
        keep = ("seed", "n_nih_patients", "max_per_patient", "n_ranzcr_patients", "n_absent_outside",
                "n_absent_heldout", "n_noline_patients", "jpeg_quality")
        return {"version": PLAN_VERSION, **{k: getattr(self, k) for k in keep}}


def data_available(cfg: E2EConfig | None = None) -> bool:
    """True when the local NIH PNGs and the v2020 metadata are present."""
    cfg = cfg or E2EConfig()
    return cfg.nih_meta.is_file() and cfg.nih_images.is_dir() and any(cfg.nih_images.glob("*.png"))


def _rel(path: Path) -> str:
    """``path`` relative to the repository root when possible (forward slashes), else absolute."""
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return Path(path).resolve().as_posix()


# ============================================================================= 1. input tree
def read_real_meta(path: Path) -> pd.DataFrame:
    """The real NIH metadata rows (raw text columns) with typed helpers."""
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    raw = raw.loc[:, [c for c in raw.columns if not c.startswith("Unnamed")]]
    raw["_pid"] = raw["Patient ID"].astype(int)
    raw["_fu"] = raw["Follow-up #"].astype(int)
    raw["_w"] = raw["OriginalImage[Width"].astype(int)
    raw["_h"] = raw["Height]"].astype(int)
    return raw


def plan_inputs(cfg: E2EConfig) -> dict[str, Any]:
    """Which real images go where, and the synthetic RANZCR labels (deterministic for ``cfg.seed``)."""
    rng = np.random.default_rng(cfg.seed)
    local = sorted(p.name for p in cfg.nih_images.glob("*.png"))
    meta = read_real_meta(cfg.nih_meta)
    meta = meta.loc[meta["Image Index"].isin(set(local))].sort_values(["_pid", "_fu"]).reset_index(drop=True)
    by_patient = {pid: g["Image Index"].tolist() for pid, g in meta.groupby("_pid")}
    multi = sorted(p for p, names in by_patient.items() if len(names) >= 3)
    nih_patients = sorted(rng.choice(multi, size=min(cfg.n_nih_patients, len(multi)), replace=False).tolist())
    nih_part: dict[int, list[str]] = {}
    heldout_pool: dict[int, list[str]] = {}
    for p in nih_patients:
        names = by_patient[p]
        keep = sorted(rng.choice(len(names), size=min(cfg.max_per_patient, len(names)), replace=False).tolist())
        nih_part[p] = [names[i] for i in keep]
        heldout_pool[p] = [n for i, n in enumerate(names) if i not in keep]
    # RANZCR part: patients of the NIH part, all of their NIH-part images
    ranzcr_patients = sorted(rng.choice(nih_patients, size=min(cfg.n_ranzcr_patients, len(nih_patients)),
                                        replace=False).tolist())
    with_heldout = [p for p in ranzcr_patients if heldout_pool[p]]
    heldout_patients = sorted(rng.choice(with_heldout, size=min(cfg.n_absent_heldout, len(with_heldout)),
                                         replace=False).tolist())
    outside = sorted(set(by_patient) - set(nih_patients))
    outside_patients = sorted(rng.choice(outside, size=min(cfg.n_absent_outside, len(outside)), replace=False).tolist())
    noline = set(rng.choice(ranzcr_patients, size=min(cfg.n_noline_patients, len(ranzcr_patients)),
                            replace=False).tolist())

    jpegs: list[dict[str, Any]] = []  # source, nih patient, kind
    for p in ranzcr_patients:
        for n in nih_part[p]:
            jpegs.append({"source": n, "nih_patient": p, "kind": "in_nih_part"})
    for p in heldout_patients:
        jpegs.append({"source": str(rng.choice(heldout_pool[p])), "nih_patient": p, "kind": "absent_heldout_followup"})
    for p in outside_patients:
        jpegs.append({"source": str(rng.choice(by_patient[p])), "nih_patient": p, "kind": "absent_other_patient"})

    # identifiers: RANZCR-like UIDs and 9-character hex PatientIDs (some digit-only), one per NIH patient
    patient_ids: dict[int, str] = {}
    for p in sorted({j["nih_patient"] for j in jpegs}):
        while True:
            pid = "".join(rng.choice(list("0123456789abcdef"), size=9).tolist())
            if len(patient_ids) % 7 == 3:
                pid = "".join(rng.choice(list("0123456789"), size=9).tolist())
            if pid not in patient_ids.values():
                break
        patient_ids[p] = pid
    uids: set[str] = set()
    for j in jpegs:
        while True:
            uid = "1.2.826.0.1.3680043.8.498." + "".join(rng.choice(list("0123456789"), size=38).tolist()).lstrip("0")
            if uid not in uids:
                break
        uids.add(uid)
        j["uid"] = uid
        j["ranzcr_patient_id"] = patient_ids[j["nih_patient"]]

    # labels: ETT in patient-level runs, lines per image
    fu_of = dict(zip(meta["Image Index"], meta["_fu"]))
    by_rp: dict[int, list[dict]] = {}
    for j in jpegs:
        by_rp.setdefault(j["nih_patient"], []).append(j)
    for p, js in by_rp.items():
        js.sort(key=lambda j: fu_of[j["source"]])
        k = len(js)
        u = rng.random()
        ett = np.zeros(k, dtype=int)
        if p in noline:
            pass
        elif u < 0.12:
            ett[:] = 1
        elif u < 0.80 and k >= 2:
            length = int(rng.integers(1, k))
            start = int(rng.integers(0, k - length + 1))
            ett[start:start + length] = 1
        for j, e in zip(js, ett):
            labels = {c: 0 for c in TRAIN_COLUMNS if c not in (C.RANZCR_UID, C.RANZCR_PATIENT)}
            if p not in noline:
                if e:
                    labels[str(rng.choice(C.ETT_COLS, p=[0.08, 0.12, 0.80]))] = 1
                if rng.random() < 0.97:
                    labels[str(rng.choice(C.CVC_COLS, p=[0.10, 0.15, 0.75]))] = 1
                if rng.random() < (0.55 if e else 0.20):
                    labels[str(rng.choice(C.NGT_COLS, p=[0.08, 0.10, 0.10, 0.72]))] = 1
                if rng.random() < 0.04:
                    labels[C.SWAN_COL] = 1
            j["labels"] = labels
    wh = dict(zip(meta["Image Index"], zip(meta["_w"], meta["_h"])))
    for j in jpegs:
        j["size"] = [int(v) for v in wh[j["source"]]]
    nih_files = sorted(n for names in nih_part.values() for n in names)
    return {"plan": cfg.plan_key(), "nih_files": nih_files, "jpegs": jpegs, "meta": meta,
            "noline_patients": sorted(int(p) for p in noline)}


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _make_jpeg(src: Path, size: tuple[int, int], dst: Path, quality: int) -> None:
    """The RANZCR conversion: bicubic resize of the NIH PNG to its original size, JPEG quality ``quality``."""
    with Image.open(src) as im:
        gray = im.convert("L")
    gray.resize(tuple(size), Image.Resampling.BICUBIC).save(dst, format="JPEG", quality=int(quality))


def build_inputs(cfg: E2EConfig) -> dict[str, Any]:
    """Write the fake Kaggle input tree (cached by plan) and return the plan with its paths."""
    t0 = time.perf_counter()
    plan = plan_inputs(cfg)
    inp = cfg.root / "kaggle_input"
    nih_root = inp / "datasets" / "nih-chest-xrays" / "data"
    comp = inp / "competitions" / C.KAGGLE_COMPETITION
    truth_dir = cfg.root / "truth"
    manifest = cfg.root / "input_manifest.json"
    key = json.dumps(plan["plan"], sort_keys=True)
    cached = (not cfg.fresh and manifest.is_file()
              and json.loads(manifest.read_text(encoding="utf-8")).get("plan_key") == key)
    paths = {"input": inp, "nih_root": nih_root, "comp": comp, "train": comp / "train.csv",
             "truth": truth_dir / "jpeg_sources.csv"}
    if cached:
        cfg.log(f"[inputs] reusing the cached input tree under {_rel(inp)}")
        return {**plan, "paths": paths, "seconds_build": 0.0, "cached": True}
    if inp.exists():
        shutil.rmtree(inp)
    nih_files = plan["nih_files"]
    half = len(nih_files) // 2
    for i, name in enumerate(nih_files):
        folder = "images_001" if i < half else "images_002"
        _link_or_copy(cfg.nih_images / name, nih_root / folder / "images" / name)
    meta = plan["meta"].set_index("Image Index")
    cols = ["Finding Labels", "Follow-up #", "Patient ID", "Patient Age", "Patient Gender", "View Position",
            "OriginalImage[Width", "Height]", "OriginalImagePixelSpacing[x", "y]"]
    lines = [KAGGLE_META_HEADER]
    for name in nih_files:  # the real rows, Kaggle split-header layout with the trailing comma
        lines.append(",".join([name, *[str(meta.at[name, c]) for c in cols]]) + ",")
    (nih_root / "Data_Entry_2017.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    train_dir = comp / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(cfg.nih_images / j["source"], tuple(j["size"]), train_dir / f"{j['uid']}.jpg") for j in plan["jpegs"]]
    with ThreadPoolExecutor(max_workers=max(1, cfg.jpeg_threads)) as pool:
        list(pool.map(lambda a: _make_jpeg(a[0], a[1], a[2], cfg.jpeg_quality), jobs))
    rows = []
    for j in plan["jpegs"]:
        rows.append({C.RANZCR_UID: j["uid"], **j["labels"], C.RANZCR_PATIENT: j["ranzcr_patient_id"]})
    pd.DataFrame(rows, columns=TRAIN_COLUMNS).to_csv(comp / "train.csv", index=False, lineterminator="\n")
    pd.DataFrame({"StudyInstanceUID": ["x"], "ETT - Abnormal": [0]}).to_csv(comp / "sample_submission.csv", index=False)
    truth_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"uid": j["uid"], "source": j["source"], "kind": j["kind"], "nih_patient_id": j["nih_patient"],
                   "ranzcr_patient_id": j["ranzcr_patient_id"]} for j in plan["jpegs"]]).to_csv(
        paths["truth"], index=False, lineterminator="\n")
    seconds = time.perf_counter() - t0
    manifest.write_text(json.dumps({"plan_key": key, "n_nih": len(nih_files), "n_jpegs": len(plan["jpegs"]),
                                    "seconds_build": round(seconds, 1)}, indent=2) + "\n", encoding="utf-8")
    cfg.log(f"[inputs] {len(nih_files)} NIH PNGs, {len(plan['jpegs'])} simulated RANZCR JPEGs in {seconds:.0f} s")
    return {**plan, "paths": paths, "seconds_build": round(seconds, 1), "cached": False}


# ============================================================================= 2. kernels
def _run_bundle(build: Path, slug: str, input_root: Path, working: Path, temp: Path) -> tuple[float, str]:
    if working.exists():
        shutil.rmtree(working)
    working.mkdir(parents=True)
    cmd = [sys.executable, "-I", str(build / slug / "kernel.py"), "--input-root", str(input_root),
           "--working", str(working), "--temp", str(temp), "--workers", "1"]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, cwd=str(working.parent))
    seconds = time.perf_counter() - t0
    (working.parent / f"{slug}.log").write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"{slug} failed with exit code {proc.returncode}: {proc.stderr[-2000:]}")
    return seconds, proc.stdout


def _margin_summary(rows: pd.DataFrame) -> dict[str, Any] | None:
    """d2 - d1 over rows with a second candidate (d2 = -1 means a one-image dimension bucket)."""
    two = rows.loc[rows["d2"].astype(int) >= 0]
    if two.empty:
        return None
    m = two["d2"].astype(int) - two["d1"].astype(int)
    return {"n": int(len(m)), "min": int(m.min()), "median": float(m.median()), "n_single_candidate": int(len(rows) - len(two))}


def _json_safe(obj: Any) -> Any:
    """JSON-ready copy: NaN and infinities become None, paths become strings."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def hamming_benchmark(bucket: int = 41_500, n_queries: int = 200, seed: int = 1) -> float:
    """Seconds per query of the 256-bit Hamming scan against ``bucket`` NIH hashes (Kaggle-size bucket)."""
    rng = np.random.default_rng(seed)
    db = rng.integers(0, 256, size=(bucket, 32), dtype=np.uint8)
    q = rng.integers(0, 256, size=(n_queries, 32), dtype=np.uint8)
    t0 = time.perf_counter()
    hamming(q, db)
    return (time.perf_counter() - t0) / n_queries


def run_kernels(cfg: E2EConfig, inputs: dict[str, Any]) -> dict[str, Any]:
    """Build the bundles, run hash then match, and score the matches against the simulation truth."""
    from kaggle_kernels import build_kernels as bk

    kroot = cfg.root / "kaggle"
    build = kroot / "build"
    bk.build_all(USERNAME, build)
    inp = inputs["paths"]["input"]
    mount = inp / "notebooks" / USERNAME / C.KERNEL_SLUGS["hash"]
    if mount.exists():
        shutil.rmtree(mount)
    hash_wd = kroot / "hash_working"
    t_hash, _ = _run_bundle(build, C.KERNEL_SLUGS["hash"], inp, hash_wd, kroot / "hash_temp")
    mount.mkdir(parents=True)
    for p in hash_wd.iterdir():
        shutil.copy2(p, mount / p.name)
    match_wd = kroot / "match_working"
    t_match, _ = _run_bundle(build, C.KERNEL_SLUGS["match"], inp, match_wd, kroot / "match_temp")

    hs = json.loads((hash_wd / "run_summary.json").read_text(encoding="utf-8"))
    ms = json.loads((match_wd / "match_stats.json").read_text(encoding="utf-8"))
    table = pd.read_csv(match_wd / "match_table.csv", dtype={"uid": str, "ranzcr_patient_id": str, "nih_filename": str},
                        keep_default_na=False, na_values={"ncc1": [""], "ncc2": [""]})
    truth = pd.read_csv(inputs["paths"]["truth"], dtype=str, keep_default_na=False)
    j = truth.merge(table, on="uid", how="left", validate="one_to_one")
    present = j["kind"] == "in_nih_part"
    absent = ~present
    correct = present & (j["status"] == "matched") & (j["nih_filename"] == j["source"])
    wrong = (j["status"] == "matched") & (j["nih_filename"] != j["source"])
    absent_unmatched = absent & (j["status"] != "matched")
    d1_true = j.loc[correct, "d1"].astype(int)
    d1_absent = j.loc[absent, "d1"].astype(int)
    ncc_true = pd.to_numeric(j.loc[correct, "ncc1"], errors="coerce")
    n_hashed = int(hs.get("n_hashed_total", 0))
    s_hash_folders = sum(float(v.get("seconds", 0.0)) for v in hs["folders"].values() if v.get("status") == "done")
    run = ms.get("run", {})
    n_jpegs = int(run.get("n_ranzcr_hashed", len(table)))
    s_png = s_hash_folders / max(n_hashed, 1)
    s_jpeg_hash = float(run.get("seconds_ranzcr_hashing", 0.0)) / max(n_jpegs, 1)
    s_jpeg_match = float(run.get("seconds_matching", 0.0)) / max(n_jpegs, 1)
    s_ham = hamming_benchmark()
    extrap_hash_h = s_png * KAGGLE_PNGS / KAGGLE_WORKERS / 3600
    extrap_match_h = (s_jpeg_hash * KAGGLE_JPEGS / KAGGLE_WORKERS + (s_jpeg_match + s_ham) * KAGGLE_JPEGS) / 3600
    res = {
        "n_nih_pngs": len(inputs["nih_files"]), "n_jpegs": int(len(truth)), "n_jpegs_source_in_nih": int(present.sum()),
        "n_jpegs_source_absent": int(absent.sum()),
        "n_absent_by_kind": {k: int((truth["kind"] == k).sum()) for k in ("absent_heldout_followup", "absent_other_patient")},
        "n_correct": int(correct.sum()), "n_wrong": int(wrong.sum()), "n_absent_unmatched": int(absent_unmatched.sum()),
        "match_accuracy": float(correct.sum() / max(int(present.sum()), 1)),
        "status_counts": {str(k): int(v) for k, v in table["status"].value_counts().sort_index().items()},
        "status_of_present": {str(k): int(v) for k, v in j.loc[present, "status"].value_counts().sort_index().items()},
        "status_of_absent": {str(k): int(v) for k, v in j.loc[absent, "status"].value_counts().sort_index().items()},
        "route_counts": ms.get("route_counts"), "key_used_counts": ms.get("key_used_counts"),
        "cut": ms.get("cut"), "cut_rule": ms.get("cut_rule"),
        "d1_true": {"min": int(d1_true.min()), "median": float(d1_true.median()), "p95": float(d1_true.quantile(0.95)),
                    "max": int(d1_true.max())} if len(d1_true) else None,
        "d1_absent": {"min": int(d1_absent.min()), "median": float(d1_absent.median()), "max": int(d1_absent.max())}
        if len(d1_absent) else None,
        "margin_true": _margin_summary(j.loc[correct]),
        "ncc_true": {"min": float(ncc_true.min()), "median": float(ncc_true.median()), "max": float(ncc_true.max())}
        if ncc_true.notna().any() else None,
        "ncc_absent": {"max": float(pd.to_numeric(j.loc[absent, "ncc1"], errors="coerce").max()),
                       "note": "NCC is computed only for candidates within the cut, so absent sources have none"},
        "n_spotcheck_pairs": int(run.get("n_spotcheck_pairs", 0)),
        "spotcheck_pairs_correct": int(run.get("n_spotcheck_pairs", 0)) if int(wrong.sum()) == 0 else None,
        "dimkey_share_found": run.get("dimkey_share_found"), "config_reason": run.get("config_reason"),
        "seconds": {"hash_kernel": round(t_hash, 1), "match_kernel": round(t_match, 1),
                    "hash_per_png": round(s_png, 4), "match_hash_per_jpeg": round(s_jpeg_hash, 4),
                    "match_per_jpeg": round(s_jpeg_match, 4), "hamming_per_query_41k_bucket": round(s_ham, 5),
                    "ranzcr_hashing_total": run.get("seconds_ranzcr_hashing"),
                    "matching_total": run.get("seconds_matching")},
        "extrapolated_kaggle_hours": {
            "hash_kernel": round(extrap_hash_h, 2), "match_kernel": round(extrap_match_h, 2),
            "assumptions": (f"{KAGGLE_PNGS:,} PNGs and {KAGGLE_JPEGS:,} JPEGs; PNG and JPEG hashing split over "
                            f"{KAGGLE_WORKERS} workers; matching (NCC confirmation, Hamming scan of a "
                            "41,500-image dimension bucket) serial; this machine's per-core speed"),
        },
        "paths": {"match_table": match_wd / "match_table.csv", "match_stats": match_wd / "match_stats.json",
                  "yield": hash_wd / "yield_table.csv", "build": build, "kroot": kroot},
    }
    cfg.log(f"[kernels] hash {t_hash:.0f} s, match {t_match:.0f} s; correct {res['n_correct']}/{res['n_jpegs_source_in_nih']}, "
            f"wrong {res['n_wrong']}, absent unmatched {res['n_absent_unmatched']}/{res['n_jpegs_source_absent']}, "
            f"cut {res['cut']}")
    return res


def run_export(cfg: E2EConfig, inputs: dict[str, Any], kern: dict[str, Any], filelist: Path) -> dict[str, Any]:
    """Export kernel on the selection's nih_filelist.csv; unzip into the pod image folder and verify."""
    inp = inputs["paths"]["input"]
    ds = inp / "datasets" / USERNAME / C.FILELIST_DATASET_SLUG
    if ds.exists():
        shutil.rmtree(ds)
    ds.mkdir(parents=True)
    shutil.copy2(filelist, ds / "nih_filelist.csv")
    kroot = kern["paths"]["kroot"]
    wd = kroot / "export_working"
    t, _ = _run_bundle(kern["paths"]["build"], C.KERNEL_SLUGS["export"], inp, wd, kroot / "export_temp")
    images = cfg.root / "pod" / "images"
    if images.exists():
        shutil.rmtree(images)
    images.mkdir(parents=True)
    with zipfile.ZipFile(wd / "selected_nih_png.zip") as zf:
        zf.extractall(images)
    manifest = pd.read_csv(wd / "manifest.csv", dtype=str)
    wanted = pd.read_csv(filelist, dtype=str)["nih_filename"].tolist()
    bad = [r.nih_filename for r in manifest.itertuples() if fileio.sha256_file(images / r.nih_filename) != r.sha256]
    cfg.log(f"[export] {len(manifest)} PNGs exported in {t:.0f} s; sha256 mismatches {len(bad)}")
    return {"n_exported": int(len(manifest)), "n_listed": len(wanted), "sha_mismatch": len(bad),
            "missing": sorted(set(wanted) - set(manifest["nih_filename"])), "seconds": round(t, 1), "images": images}


# ============================================================================= 3. selection
def run_selection(cfg: E2EConfig, inputs: dict[str, Any], kern: dict[str, Any]) -> dict[str, Any]:
    from scripts import select_items

    items_dir = cfg.root / "items"
    if items_dir.exists():
        shutil.rmtree(items_dir)
    images = cfg.root / "pod" / "images"
    # a probe list like results/probes/probe_seen_filenames.txt: 5 matched images and one outside the pool
    table = pd.read_csv(kern["paths"]["match_table"], dtype=str, keep_default_na=False)
    matched = sorted(table.loc[table["status"] == "matched", "nih_filename"])
    probes = [matched[i] for i in np.linspace(0, len(matched) - 1, 5).astype(int)] + ["99999999_000.png"]
    probe_file = cfg.root / "probe_seen_filenames.txt"
    probe_file.write_bytes(("# images seen by software tests before the protocol freeze (simulated)\n\n"
                            + "\n".join(probes) + "\n").encode("utf-8"))
    argv = ["--match", str(kern["paths"]["match_table"]), "--train", str(inputs["paths"]["train"]),
            "--nih", str(cfg.nih_meta), "--images-dir", _rel(images), "--out", str(items_dir),
            "--exclude-filenames", str(probe_file),
            "--n-pairs", str(cfg.n_pairs), "--n-u-per-stratum", str(cfg.n_u_per_stratum),
            "--n-dev-pos", str(cfg.n_dev_pos), "--n-dev-neg", str(cfg.n_dev_neg), "--n-retest", str(cfg.n_retest),
            "--n-smoke", str(cfg.n_smoke)]
    buf = io.StringIO()
    stdout = sys.stdout
    sys.stdout = buf
    try:
        rc = select_items.main(argv)
    finally:
        sys.stdout = stdout
    if rc != 0:
        raise RuntimeError(f"select_items.py exited with {rc}")
    report = json.loads((items_dir / "selection_report.json").read_text(encoding="utf-8"))
    cfg.log("[select] " + buf.getvalue().strip().replace("\n", "\n[select] "))
    return {"dir": items_dir, "report": report, "stdout": buf.getvalue(), "probes": probes}


def check_selection(items_dir: Path, train_csv: Path, probes: list[str] | None = None) -> dict[str, Any]:
    """The S5/S6/S10 invariants of a selection; returns the facts checked (raises AssertionError)."""
    from cxrdev.items import ITEM_COLS, POD_ITEM_COLS, READING_LIST_COLS, read_items

    items = read_items(items_dir / "items.csv")
    assert list(items.columns) == list(ITEM_COLS)
    pod = pd.read_csv(items_dir / "pod_items.csv", dtype=str, keep_default_na=False)
    assert list(pod.columns) == list(POD_ITEM_COLS), "pod_items.csv has a column outside S6"
    train = pd.read_csv(train_csv, dtype=str, keep_default_na=False)
    pod_text = (items_dir / "pod_items.csv").read_text(encoding="utf-8")
    assert not any(u in pod_text for u in train[C.RANZCR_UID]), "a RANZCR uid reached pod_items.csv"
    assert not any(p in pod_text for p in train[C.RANZCR_PATIENT]), "a RANZCR PatientID reached pod_items.csv"
    arm = items["arm"]
    sets = {"pairs": set(items.loc[arm.isin(["A", "B"]), "nih_patient_id"]),
            "U": set(items.loc[arm == "U", "nih_patient_id"]), "DEV": set(items.loc[arm == "DEV", "nih_patient_id"])}
    assert not (sets["pairs"] & sets["U"]) and not (sets["pairs"] & sets["DEV"]) and not (sets["U"] & sets["DEV"])
    assert items.loc[arm == "U", "nih_patient_id"].is_unique and items.loc[arm == "DEV", "nih_patient_id"].is_unique
    a = items.loc[arm == "A"].set_index("pair_id")
    b = items.loc[arm == "B"].set_index("pair_id").loc[a.index]
    assert (a["nih_patient_id"].to_numpy() == b["nih_patient_id"].to_numpy()).all()
    assert (a["ett"] == 1).all() and (b["ett"] == 0).all()
    assert items["item_id"].str.fullmatch(r"[0-9a-f]{10}").all() and items["item_id"].is_unique
    assert set(pod["item_id"]) == set(items["item_id"]) and pod["item_id"].is_monotonic_increasing
    reading = pd.read_csv(items_dir / "reading_list.csv", dtype=str, keep_default_na=False)
    key = pd.read_csv(items_dir / "reading_key.csv", dtype=str, keep_default_na=False)
    assert list(reading.columns) == list(READING_LIST_COLS)
    abu = items.loc[arm.isin(["A", "B", "U"])]
    assert len(reading) == len(abu) and set(reading["nih_filename"]) == set(abu["nih_filename"]), \
        "the reading list must be every A, B and U image"
    assert reading["read_id"].str.fullmatch(r"[0-9A-F]{8}").all() and reading["read_id"].is_unique
    assert (key["read_id"] == reading["read_id"]).all()
    joined = key.merge(items, on="item_id")
    assert not (joined["item_id"].str[:8].str.upper() == joined["read_id"]).any()
    practice = pd.read_csv(items_dir / "practice_list.csv", dtype=str, keep_default_na=False)
    arm_of = dict(zip(items["nih_filename"], items["arm"]))
    assert len(practice) == 10 and {arm_of[f] for f in practice["nih_filename"]} == {"DEV"}
    assert not set(practice["read_id"]) & set(reading["read_id"])
    retest = pd.read_csv(items_dir / "retest_list.csv", dtype=str, keep_default_na=False)
    assert set(retest["read_id"]) <= set(reading["read_id"])
    for name, purpose in (("reading_list", "reading"), ("retest_list", "retest")):
        assert json.loads((items_dir / f"{name}.meta.json").read_text(encoding="utf-8")) == {"seed": C.derive_seed(purpose)}
    report = json.loads((items_dir / "selection_report.json").read_text(encoding="utf-8"))
    if probes is not None:
        assert not set(probes) & set(items["nih_filename"]), "an excluded probe image was selected"
        ex = report["excluded_filenames"]
        assert ex["n_listed"] == len(probes) and ex["n_present_in_matched_pool"] == len(probes) - 1
    return {
        "excluded_filenames": report.get("excluded_filenames"),
        "n_items": int(len(items)), "by_arm": {k: int(v) for k, v in arm.value_counts().sort_index().items()},
        "u_strata": {k: int(v) for k, v in items.loc[arm == "U", "stratum"].value_counts().sort_index().items()},
        "h1_substitute": bool(report["h1_substitute"]), "h1_contrast": report["u"]["h1_contrast"],
        "rung": report["rung"], "n_reading_list": int(len(reading)), "n_retest": int(len(retest)),
        "n_practice": int(len(practice)), "n_smoke": int(items["smoke"].sum()),
        "n_noline_patients_available": report["u"]["n_noline_patients_available"],
    }


# ============================================================================= 4. fake model backends
FV = 64
F_PAD, F_BOS, F_EOS, F_EOT, F_IMG, F_GEN, F_UNK = 0, 1, 2, 3, 5, 6, 63
F_PIECES = {"Yes": 10, "yes": 11, " Yes": 12, " yes": 13, "YES": 14, " YES": 15,
            "No": 20, "no": 21, " No": 22, " no": 23, "NO": 24, " NO": 25,
            "I": 30, " I": 31, "Cannot": 32, "cannot": 33, " cannot": 34, "Unknown": 35, "unknown": 36, " unknown": 37,
            " don't": 40, " know": 41, ".": 42, "Maybe": 43}
F_TEXT = {v: k for k, v in F_PIECES.items()}
KIND_CODES = {"answer": 0, "abstain": 1, "unparsable": 2, "truncated": 3}
#: per model: (label weight, other-lines weight, offset, noise sd, P1 abstain band, P2hi abstain band)
MODEL_BEHAVIOUR = {"qwen3vl2b": (2.2, 0.5, -0.2, 1.2, 0.5, 1.3), "medgemma15": (1.2, 0.9, 0.3, 1.5, 0.3, 0.9)}


def pixel_key(image: Image.Image) -> str:
    """Content key of an image (SHA-256 of its 8-bit grey pixels)."""
    return hashlib.sha256(np.asarray(image.convert("L"), dtype=np.uint8).tobytes()).hexdigest()


class Oracle:
    """What the fake models "see": the RANZCR label and line count of an image, found by its pixels.

    A simulation device only: it is built by this harness from the local item list so that the fake
    answers carry signal; the real pod receives no label.
    """

    def __init__(self, items: pd.DataFrame, images: Path) -> None:
        from pod import common
        from pod.run_medsiglip import card_resize

        self.truth: dict[str, tuple[int, int, str]] = {}
        for r in items.itertuples(index=False):
            img, _sha = common.load_image_file(images / r.nih_filename)
            rec = (int(r.ett), int(r.n_other_lines), str(r.nih_filename))
            self.truth[pixel_key(img)] = rec
            self.truth[pixel_key(card_resize(img))] = rec

    @staticmethod
    def _unit(*parts: str) -> float:
        h = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
        return int.from_bytes(h[:8], "big") / 2.0 ** 64

    def _normal(self, *parts: str) -> float:
        u1, u2 = max(self._unit(*parts, "a"), 1e-12), self._unit(*parts, "b")
        return float(np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2))

    def plan(self, model_key: str, image: Image.Image | None, prompt_id: str) -> tuple[float, str]:
        """(z, kind) for one generation."""
        w_y, w_l, off, sd, band1, band2 = MODEL_BEHAVIOUR.get(model_key, MODEL_BEHAVIOUR["qwen3vl2b"])
        if image is None:
            key, z = "BLIND-text", 0.6 + off
        else:
            key = pixel_key(image)
            rec = self.truth.get(key)
            if rec is None:  # the grey BLIND image (or anything unknown)
                z = -0.7 + off
            else:
                y, lines, _name = rec
                z = w_y * (2 * y - 1) + w_l * (lines - 1.5) + off + sd * self._normal(model_key, key)
        z += {"P0": 0.0, "P1": -0.1, "P2hi": -0.3, "P2lo": -0.1}.get(prompt_id, 0.0)
        u = self._unit(model_key, key, prompt_id, "kind")
        if image is not None and u < 0.015:
            return z, ("truncated" if u < 0.004 else "unparsable")
        band = {"P1": band1, "P2hi": band2, "P2lo": band1}.get(prompt_id)
        if band is not None and abs(z) < band:
            return z, "abstain"
        return z, "answer"

    def detector_logit(self, image: Image.Image) -> float:
        rec = self.truth.get(pixel_key(image))
        if rec is None:
            return 0.0
        y, _lines, name = rec
        return 1.8 * (2 * y - 1) + 1.2 * self._normal("medsiglip", name)


class FakeTokenizer:
    """Single-token yes/no/abstain surface forms; other text one id per character."""

    eos_token_id = F_EOS
    pad_token_id = F_PAD
    unk_token_id = F_UNK

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if text in F_PIECES:
            return [F_PIECES[text]]
        return [44 + (ord(c) % 16) for c in text]

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        return "".join(F_TEXT.get(int(i), "") if int(i) in F_TEXT else ("" if int(i) < 10 else "x") for i in ids)

    def convert_tokens_to_ids(self, token: str) -> int:
        return {"<|im_end|>": F_EOT, "<end_of_turn>": F_EOT}.get(token, F_UNK)


class FakeVLMProcessor:
    """Chat template and inputs in the shape the runner expects (Qwen: image_grid_thw; Gemma: 256 tokens)."""

    def __init__(self, model_key: str, oracle: Oracle) -> None:
        import torch  # noqa: F401  (the runner needs torch tensors)

        self.model_key = model_key
        self.family = C.MODELS[model_key]["family"]
        self.qwen = self.family in ("qwen3vl", "qwen25vl")
        self.tokenizer = FakeTokenizer()
        self.oracle = oracle
        size = {"shortest_edge": 56 * 56, "longest_edge": 28 * 28 * 1280} if self.qwen else {"height": 896, "width": 896}
        self.image_processor = SimpleNamespace(size=size, merge_size=2, patch_size=16, do_pan_and_scan=False)
        self.prompts = {prompt_text(pid, "ett"): pid for pid in PROMPT_IDS}

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=False, return_dict=False,
                            return_tensors=None, **kwargs):
        import torch

        content = messages[0]["content"]
        text = [c["text"] for c in content if c["type"] == "text"][0]
        has_image = any(c["type"] == "image" for c in content)
        if not tokenize:
            return f"<user>{'<image>' if has_image else ''}{text}<end><assistant>"
        image = next((c.get("image") for c in content if c["type"] == "image"), None)
        z, kind = self.oracle.plan(self.model_key, image, self.prompts.get(text, "P0"))
        n_img = (1024 if self.qwen else 256) if image is not None else 0
        ids = [F_BOS] + [F_IMG] * n_img + self.tokenizer.encode(text[:12]) + [F_GEN]
        out = {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
               "e2e_plan": torch.tensor([[int(round(z * 1000)), KIND_CODES[kind]]], dtype=torch.long)}
        type_ids = torch.tensor([[1 if i == F_IMG else 0 for i in ids]])
        if self.qwen:
            out["mm_token_type_ids"] = type_ids
            if image is not None:
                out["image_grid_thw"] = torch.tensor([[1, 64, 64]])
                out["pixel_values"] = torch.zeros((4, 6))
        else:
            out["token_type_ids"] = type_ids
            if image is not None:
                out["pixel_values"] = torch.zeros((1, 3, 4, 4))
        return out


class FakeVLMModel:
    """Greedy generation from the planned first-step logits; returns raw logits like transformers."""

    def __init__(self, family: str) -> None:
        import transformers

        cfg = {"image_token_id": F_IMG}
        if family == "gemma3":
            cfg["mm_tokens_per_image"] = 256
        self.config = SimpleNamespace(**cfg)
        self.generation_config = transformers.GenerationConfig(eos_token_id=[F_EOS], pad_token_id=F_PAD)

    @staticmethod
    def _onehot(tok: int):
        import torch

        row = torch.full((1, FV), -9.0)
        row[0, tok] = 9.0
        return row

    def generate(self, input_ids=None, generation_config=None, e2e_plan=None, **kwargs):
        import torch

        z = float(e2e_plan[0, 0]) / 1000.0
        kind = int(e2e_plan[0, 1])
        first = torch.full((1, FV), -9.0)
        first[0, F_PIECES["Yes"]] = 4.0 + z / 2.0
        first[0, F_PIECES["No"]] = 4.0 - z / 2.0
        top = float(max(first[0, F_PIECES["Yes"]], first[0, F_PIECES["No"]]))
        first[0, F_PIECES["I"]] = top + 1.0 if kind == KIND_CODES["abstain"] else -6.0
        if kind in (KIND_CODES["unparsable"], KIND_CODES["truncated"]):
            first[0, F_PIECES["Maybe"]] = top + 1.0
        tok = int(torch.argmax(first[0]))
        follow = {F_PIECES["Yes"]: [F_PIECES["."], F_EOT], F_PIECES["No"]: [F_PIECES["."], F_EOT],
                  F_PIECES["I"]: [F_PIECES[" don't"], F_PIECES[" know"], F_PIECES["."], F_EOT],
                  F_PIECES["Maybe"]: [F_PIECES["."], F_EOT]}[tok]
        if kind == KIND_CODES["truncated"]:
            follow = [F_PIECES["."]] * 40
        steps = [tok] + follow
        steps = steps[: int(generation_config.max_new_tokens)]
        logits = [first] + [self._onehot(t) for t in steps[1:]]
        seq = input_ids[0].tolist() + steps
        return SimpleNamespace(sequences=torch.tensor([seq]), logits=tuple(logits))

    def __call__(self, *args, **kwargs):  # pragma: no cover - only the full-sequence fallback calls it
        raise NotImplementedError("the e2e fake model supports the first-token signal only")


def fake_vlm_factory(oracle: Oracle) -> Callable[..., Any]:
    """A ``backend_factory`` for pod/run_vlm.py and pod/smoke_test.py (no weights, no download)."""
    def factory(model_key, device=None, dtype=None, revision=None, local_files_only=False,
                max_new_tokens=C.MAX_NEW_TOKENS, **kwargs):
        import torch
        import transformers
        from pod.run_vlm import VLMBackend

        spec = C.MODELS[model_key]
        proc = FakeVLMProcessor(model_key, oracle)
        rev = hashlib.sha1(f"e2e-fake-{model_key}".encode()).hexdigest()
        return VLMBackend(FakeVLMModel(str(spec["family"])), proc, model_key, revision=rev,
                          dtype_name=dtype or str(spec["dtype"]), device="cpu",
                          transformers_version=transformers.__version__, torch_version=torch.__version__,
                          max_new_tokens=max_new_tokens)
    return factory


class FakeSiglipProcessor:
    def __init__(self, oracle: Oracle) -> None:
        self.oracle = oracle

    def __call__(self, text=None, images=None, padding=None, return_tensors=None):
        import torch

        d = [self.oracle.detector_logit(im) for im in images]
        return {"input_ids": torch.zeros((len(text), 64), dtype=torch.long),
                "pixel_values": torch.zeros((len(images), 3, 2, 2)),
                "e2e_d": torch.tensor([int(round(v * 1000)) for v in d], dtype=torch.long)}


class FakeSiglipModel:
    def __call__(self, input_ids=None, pixel_values=None, e2e_d=None, **kwargs):
        import torch

        d = e2e_d.to(torch.float32) / 1000.0
        return SimpleNamespace(logits_per_image=torch.stack([3.0 + d / 2.0, 3.0 - d / 2.0], dim=1))


def run_pod(cfg: E2EConfig, items_dir: Path, images: Path) -> dict[str, Any]:
    """The pod queue (smoke, items, BLIND rows, MedSigLIP) with fake backends; returns the run directory."""
    from cxrdev.items import read_items
    from pod import run_medsiglip, run_vlm, smoke_test

    pod_root = cfg.root / "pod"
    run_dir = pod_root / "runs" / RUN_ID
    if run_dir.exists():
        shutil.rmtree(run_dir)
    (pod_root / "items").mkdir(parents=True, exist_ok=True)
    pod_items = pod_root / "items" / "pod_items.csv"
    shutil.copy2(items_dir / "pod_items.csv", pod_items)  # the only item file that goes to the pod
    t0 = time.perf_counter()
    oracle = Oracle(read_items(items_dir / "items.csv"), images)
    factory = fake_vlm_factory(oracle)
    quiet = io.StringIO()
    stdout = sys.stdout
    sys.stdout = quiet
    try:
        rc = smoke_test.main(["--items", str(pod_items), "--images-dir", str(images), "--run-dir", str(run_dir),
                              "--run-id", RUN_ID], backend_factory=factory)
        if rc != 0:
            raise RuntimeError(f"smoke_test.py exited with {rc}")
        report = run_dir / "smoke_report.json"
        for mk in C.CORE_MODELS:
            out = run_dir / f"{mk}.jsonl"
            for extra in ([], ["--blind"]):
                argv = ["--model-key", mk, "--out", str(out), "--run-id", RUN_ID, "--signal", "auto",
                        "--smoke-report", str(report), *extra]
                if not extra:
                    argv += ["--items", str(pod_items), "--images-dir", str(images)]
                rc = run_vlm.main(argv, backend_factory=factory)
                if rc != 0:
                    raise RuntimeError(f"run_vlm.py {mk} {extra} exited with {rc}")
        rc = run_medsiglip.main(["--items", str(pod_items), "--images-dir", str(images),
                                 "--out", str(run_dir / "detector_scores.csv")],
                                scorer_factory=lambda **kw: run_medsiglip.MedSiglipScorer(
                                    FakeSiglipModel(), FakeSiglipProcessor(oracle), model_id=str(C.EXPERT["hf_id"]),
                                    revision=hashlib.sha1(b"e2e-fake-medsiglip").hexdigest(), device="cpu",
                                    dtype_name=kw.get("dtype", "float32"), resize=kw.get("resize", "card")))
        if rc != 0:
            raise RuntimeError(f"run_medsiglip.py exited with {rc}")
    finally:
        sys.stdout = stdout
    smoke = json.loads((run_dir / "smoke_report.json").read_text(encoding="utf-8"))
    n_lines = {mk: sum(1 for _ in open(run_dir / f"{mk}.jsonl", encoding="utf-8")) for mk in C.CORE_MODELS}
    seconds = time.perf_counter() - t0
    cfg.log(f"[pod] fake queue in {seconds:.0f} s; records {n_lines}; smoke decisions "
            f"{ {k: v.get('decision') for k, v in smoke['models'].items()} }")
    return {"run_dir": run_dir, "pod_items": pod_items, "n_records": n_lines, "seconds": round(seconds, 1),
            "smoke": {k: {"decision": v.get("decision"), "median_coverage": v.get("median_coverage"),
                          "visual_tokens": v.get("visual_tokens")} for k, v in smoke["models"].items()}}


# ============================================================================= 5. reading, verification, analysis
class _Clock:
    """Injected clocks for the reading session: monotonic seconds and an aware UTC time."""

    def __init__(self, start: datetime, step: float = 12.0) -> None:
        self.start, self.step, self.t = start, step, 0.0

    def mono(self) -> float:
        self.t += self.step
        return self.t

    def now(self) -> datetime:
        return self.start + timedelta(seconds=self.t)


def simulate_reading(cfg: E2EConfig, items_dir: Path) -> dict[str, Any]:
    """A blind read of reading_list.csv and a retest 49 h later, through tools/reading_core.

    The simulated reader answers by the RANZCR CLiP label except for planted events: three A
    images flagged "tracheostomy" (two read "absent", the tube being a tracheostomy tube, one
    "present"), one B flagged ambiguous, one U flagged out of field, one B and one U read contrary to
    the label (genuine disagreements) and one B skipped. The practice list is also run once.
    """
    from cxrdev.items import read_items
    from tools import reading_core as rc

    items = read_items(items_dir / "items.csv")
    key = pd.read_csv(items_dir / "reading_key.csv", dtype=str, keep_default_na=False)
    info = key.merge(items, on="item_id").set_index("read_id")
    order = pd.read_csv(items_dir / "reading_list.csv", dtype=str, keep_default_na=False)["read_id"].tolist()
    rng = np.random.default_rng(cfg.seed + 1)

    def pick(arm: str, n: int, used: set[str]) -> list[str]:
        cands = [r for r in order if info.at[r, "arm"] == arm and r not in used]
        chosen = [cands[i] for i in sorted(rng.choice(len(cands), size=n, replace=False).tolist())]
        used.update(chosen)
        return chosen

    used: set[str] = set()
    events: dict[str, tuple[str | None, tuple[str, ...], str]] = {}
    for r in pick("A", 2, used):
        events[r] = ("absent", ("tracheostomy",), "flag_tracheostomy")
    for r in pick("A", 1, used):
        events[r] = ("present", ("tracheostomy",), "flag_tracheostomy")
    for r in pick("B", 1, used):
        events[r] = ("present", ("ambiguous",), "flag_ambiguous")
    for r in pick("U", 1, used):
        events[r] = ("absent", ("out_of_field",), "flag_out_of_field")
    for r in pick("B", 1, used) + pick("U", 1, used):
        events[r] = ("present", (), "label_disagreement")
    for r in pick("B", 1, used):
        events[r] = ("skip", (), "skip")

    reading_dir = cfg.root / "reading"
    if reading_dir.exists():
        shutil.rmtree(reading_dir)
    reading_dir.mkdir(parents=True)
    # practice run on the dev-pool practice list (accepted by the tool, written to a temporary folder)
    plan = rc.plan_run(items_dir / "practice_list.csv", None, practice=True, practice_base=reading_dir)
    clock = _Clock(datetime(2026, 11, 10, 8, 30, tzinfo=timezone.utc))
    ps = rc.ReadingSession(plan.list_path, plan.out_path, limit=plan.limit, clock=clock.mono, now=clock.now)
    while not ps.complete:
        ps.record("absent")
    ps.finalise()

    clock = _Clock(datetime(2026, 11, 10, 9, 0, tzinfo=timezone.utc))
    out = reading_dir / "verified_items.csv"
    s = rc.ReadingSession(items_dir / "reading_list.csv", out, "read", clock=clock.mono, now=clock.now)
    while not s.complete:
        rid = s.current_item.read_id
        if rid in events:
            response, flags, _reason = events[rid]
        else:
            response, flags = ("present" if int(info.at[rid, "ett"]) == 1 else "absent"), ()
        s.record(response, flags)
    first = s.finalise()
    saved = rc.parse_utc(first.saved_utc)
    retest_start = saved + timedelta(hours=49)
    rplan = rc.plan_run(items_dir / "retest_list.csv", reading_dir / "retest.csv", "retest", first_read=out,
                        now=retest_start)
    clock = _Clock(retest_start)
    rs = rc.ReadingSession(rplan.list_path, rplan.out_path, "retest", clock=clock.mono, now=clock.now)
    first_answers = pd.read_csv(out, dtype=str, keep_default_na=False).set_index("read_id")
    flips = 0
    while not rs.complete:
        rid = rs.current_item.read_id
        prev = first_answers.at[rid, "response"]
        response = prev
        if prev in ("present", "absent") and flips < 2 and rid not in events:
            response, flips = ("absent" if prev == "present" else "present"), flips + 1
        rs.record(response)
    rs.finalise()
    cfg.log(f"[reading] {s.n} images read (saved {first.saved_utc}); retest of {rs.n} from "
            f"{retest_start:%Y-%m-%dT%H:%M:%SZ}; practice {ps.n} dev-pool images")
    item_of = dict(zip(key["read_id"], key["item_id"]))
    return {"dir": reading_dir, "verified": out, "retest": reading_dir / "retest.csv",
            "planted": {item_of[r]: ev[2] for r, ev in events.items()},
            "trach_a_items": sorted(item_of[r] for r, ev in events.items() if ev[2] == "flag_tracheostomy"),
            "saved_utc": first.saved_utc, "retest_start_utc": f"{retest_start:%Y-%m-%dT%H:%M:%SZ}", "n_read": s.n,
            "n_retest": rs.n, "n_practice": ps.n}


def secrets_of(cfg: E2EConfig, inputs: dict[str, Any], items_dir: Path) -> dict[str, set[str]]:
    """Identifiers that must never appear in results or the report."""
    train = pd.read_csv(inputs["paths"]["train"], dtype=str, keep_default_na=False)
    read_ids: set[str] = set()
    for name in ("reading_list.csv", "retest_list.csv", "practice_list.csv"):
        read_ids |= set(pd.read_csv(items_dir / name, dtype=str, keep_default_na=False)["read_id"])
    items = pd.read_csv(items_dir / "items.csv", dtype=str, keep_default_na=False)
    return {"uid": set(train[C.RANZCR_UID]), "ranzcr_patient_id": set(train[C.RANZCR_PATIENT]),
            "read_id": read_ids, "item_id": set(items["item_id"]), "nih_filename": set(items["nih_filename"])}


def leak_scan(folder: Path, secrets: dict[str, set[str]], kinds: tuple[str, ...]) -> dict[str, list[str]]:
    """{file name: [kind of every identifier found]} for the text files under ``folder``."""
    found: dict[str, list[str]] = {}
    patterns = {k: re.compile("|".join(re.escape(s) for s in sorted(secrets[k], key=len, reverse=True)))
                for k in kinds if secrets[k]}
    for p in sorted(folder.rglob("*")):
        if not p.is_file() or p.suffix.lower() in (".png", ".pdf", ".docx"):
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        hits = [k for k, rx in patterns.items() if rx.search(text)]
        if hits:
            found[p.relative_to(folder).as_posix()] = hits
    return found


def run_e2e(cfg: E2EConfig | None = None) -> dict[str, Any]:
    """Run every step and return the summary (also written to ``<root>/e2e_summary.json``)."""
    from scripts import analyze, build_report, parse_runs, verify_join

    cfg = cfg or E2EConfig()
    if not data_available(cfg):
        raise FileNotFoundError(f"NIH images or metadata missing ({cfg.nih_images}, {cfg.nih_meta})")
    t_start = time.perf_counter()
    cfg.root.mkdir(parents=True, exist_ok=True)
    inputs = build_inputs(cfg)
    kern = run_kernels(cfg, inputs)
    sel = run_selection(cfg, inputs, kern)
    checks = check_selection(sel["dir"], inputs["paths"]["train"], sel["probes"])
    export = run_export(cfg, inputs, kern, sel["dir"] / "nih_filelist.csv")
    pod = run_pod(cfg, sel["dir"], export["images"])

    parsed = cfg.root / "local_runs" / "parsed_results.csv"
    psum = parse_runs.parse_runs([pod["run_dir"]], parsed, cfg.root / "local_runs" / "parse_summary.json")
    shutil.copy2(pod["run_dir"] / "detector_scores.csv", cfg.root / "local_runs" / "detector_scores.csv")

    reading = simulate_reading(cfg, sel["dir"])
    results = cfg.root / "results"
    if results.exists():
        shutil.rmtree(results)
    excl_path = sel["dir"] / "exclusions.csv"
    vsum = verify_join.run(sel["dir"] / "items.csv", sel["dir"] / "reading_key.csv", reading["verified"], excl_path,
                           results / "verification_summary.json", retest_path=reading["retest"])
    exclusions = pd.read_csv(excl_path, dtype=str, keep_default_na=False)
    disagreements = pd.read_csv(sel["dir"] / "verification_disagreements.csv", dtype=str, keep_default_na=False)

    quiet = io.StringIO()
    stdout = sys.stdout
    sys.stdout = quiet
    try:
        rc = analyze.main(["--items", str(sel["dir"] / "items.csv"), "--parsed", str(parsed),
                           "--detector", str(cfg.root / "local_runs" / "detector_scores.csv"),
                           "--exclusions", str(excl_path), "--match-stats", str(kern["paths"]["match_stats"]),
                           "--yield", str(kern["paths"]["yield"]), "--selection", str(sel["dir"] / "selection_report.json"),
                           "--verification", str(results / "verification_summary.json"),
                           "--run-dir", str(pod["run_dir"]), "--out", str(results), "--fast",
                           # the student's 60-pair visual check, simulated from the known truth
                           *(["--spotcheck-correct", str(kern["spotcheck_pairs_correct"])]
                             if kern.get("spotcheck_pairs_correct") is not None else [])])
        rb = build_report.main(["--numbers", str(results / "numbers.json"), "--out", str(cfg.root / "report")])
    finally:
        sys.stdout = stdout
    if rc != 0 or rb != 0:
        raise RuntimeError(f"analyze.py exited {rc}, build_report.py exited {rb}")
    numbers = json.loads((results / "numbers.json").read_text(encoding="utf-8"))
    hyp = json.loads((results / "hypotheses.json").read_text(encoding="utf-8"))
    build_log = json.loads((cfg.root / "report" / "build_log.json").read_text(encoding="utf-8"))
    secrets = secrets_of(cfg, inputs, sel["dir"])
    leaks_results = leak_scan(results, secrets, ("uid", "ranzcr_patient_id", "read_id", "item_id", "nih_filename"))
    leaks_report = leak_scan(cfg.root / "report", secrets, ("uid", "ranzcr_patient_id", "read_id", "item_id"))

    excl_reason = dict(zip(exclusions["item_id"], exclusions["reason"]))
    summary: dict[str, Any] = {
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(cfg).items() if k != "log"},
        "inputs": {"n_nih_pngs": len(inputs["nih_files"]), "n_nih_patients": len({n[:8] for n in inputs["nih_files"]}),
                   "n_jpegs": len(inputs["jpegs"]), "seconds_build": inputs["seconds_build"],
                   "cached": inputs["cached"], "noline_patients": len(inputs["noline_patients"])},
        "matching": {k: v for k, v in kern.items() if k != "paths"},
        "selection": checks,
        "export": {k: v for k, v in export.items() if k != "images"},
        "pod": {k: v for k, v in pod.items() if k not in ("run_dir", "pod_items")},
        "parse": {"n_rows": psum["n_rows"], "n_duplicates_dropped": psum["n_duplicates_dropped"],
                  "skipped_files": [Path(f).name for f in psum["skipped_files"]], "files": [Path(f).name for f in psum["files"]]},
        "reading": {"n_read": reading["n_read"], "n_retest": reading["n_retest"], "n_practice": reading["n_practice"],
                    "saved_utc": reading["saved_utc"], "retest_start_utc": reading["retest_start_utc"]},
        "verification": {
            "exclusions_by_reason": vsum["exclusions"]["by_reason"], "n_disagreements": vsum["n_disagreements"],
            "trach_a_items_reason": sorted({excl_reason.get(i, "") for i in reading["trach_a_items"]}),
            "trach_a_in_disagreements": int(disagreements["item_id"].isin(reading["trach_a_items"]).sum()),
            "planted_matches": all(excl_reason.get(i) == r for i, r in reading["planted"].items()),
            "retest_gap_hours": vsum["retest"]["gap_hours_after_read"], "retest_gap_ok": vsum["retest"]["gap_ok"],
            "warnings": vsum["warnings"], "kappa_retest": vsum["retest"]["kappa"],
        },
        "analysis": {
            "files": sorted(p.name for p in results.iterdir()),
            "h0": {mk: v.get("class") for mk, v in hyp.get("H0", {}).items()},
            "h1": [{k: r.get(k) for k in ("model_key", "contrast", "substitute", "estimate", "n_exposed", "n_reference",
                                           "n_descriptive", "tested")} for r in hyp["H1"]["primary"]],
            "h3": [{k: r.get(k) for k in ("model_key", "estimate", "p_value", "n", "p_method")} for r in hyp["H3"]["primary"]],
            "h5": [{k: r.get(k) for k in ("model_key", "estimate", "p_value", "n")} for r in hyp["H5"]["primary"]],
            "numbers": {k: numbers.get(k) for k in ("match.rate", "match.cut_bits", "match.dimkey_share_found",
                                                     "match.dimkey_used", "u.h1_substitute", "verify.n_read",
                                                     "verify.n_disagreements", "excl.flag_tracheostomy.n",
                                                     "run.medsiglip.dtype", "run.gpu_name", "verify.retest_gap_hours")},
            "n_null_numbers": sum(1 for v in numbers.values() if v is None),
        },
        "report": {"n_sections": len(build_log["sections"]), "n_unresolved": len(build_log["unresolved_placeholders"]),
                   "unresolved_head": build_log["unresolved_placeholders"][:20]},
        "leaks": {"results": leaks_results, "report": leaks_report},
        "seconds_total": round(time.perf_counter() - t_start, 1),
    }
    summary = _json_safe(summary)
    (cfg.root / "e2e_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
                                               encoding="utf-8")
    cfg.log(f"[done] {summary['seconds_total']:.0f} s; summary in {_rel(cfg.root / 'e2e_summary.json')}")
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n", 1)[0])
    ap.add_argument("--root", default=str(E2EConfig.root), help="working folder (git-ignored), default runs/e2e_real")
    ap.add_argument("--fresh", action="store_true", help="rebuild the input tree even when the cached one matches")
    ap.add_argument("--jpeg-threads", type=int, default=6)
    a = ap.parse_args(argv)
    cfg = E2EConfig(root=Path(a.root), fresh=a.fresh, jpeg_threads=a.jpeg_threads)
    if not data_available(cfg):
        print(f"NIH data not found under {cfg.nih_images} / {cfg.nih_meta}; nothing to do", file=sys.stderr)
        return 2
    s = run_e2e(cfg)
    m = s["matching"]
    print(json.dumps({"match_accuracy": m["match_accuracy"], "n_wrong": m["n_wrong"],
                      "absent_unmatched": f"{m['n_absent_unmatched']}/{m['n_jpegs_source_absent']}", "cut": m["cut"],
                      "d1_true": m["d1_true"], "d1_absent": m["d1_absent"], "ncc_true": m["ncc_true"],
                      "seconds": m["seconds"], "extrapolated_kaggle_hours": m["extrapolated_kaggle_hours"],
                      "leaks": s["leaks"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
