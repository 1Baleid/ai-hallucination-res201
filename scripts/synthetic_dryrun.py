"""Synthetic end-to-end dry run of the Stage 1 analysis, with known (planted) effects.

Everything here is synthetic: no RANZCR or NIH record is read or written. The run builds

* ``items/items.csv`` (S5): 120 A/B pairs (one per patient, or two for some patients when
  ``n_two_pair_patients`` > 0), U = 60 "line" + 60 "noline" (or "ge2"/"eq1" with
  ``h1_substitute``) with 30 AP and 30 PA images per stratum, DEV = 150 pos + 150 neg with 20 smoke
  items, plus ``reading_list.csv``, ``reading_key.csv``, ``retest_list.csv`` and
  ``selection_report.json``;
* ``runs/parsed_results.csv`` (S8) for two models ("qwen3vl2b" planted GO, "medgemma15" planted
  CONDITIONAL): P0, P1 and P2hi on A, B and U, P0 on DEV, and BLIND-text / BLIND-grey rows, with
  exact planted counts per model x prompt x group (a co-occurrence prior: FAR_U(line) well above
  FAR_U(noline); more abstention under P2hi than P1; a few unparsable and truncated outputs), z
  consistent with the answers except a few flipped items, wrong answers with small |z|;
* ``runs/detector_scores.csv`` (S9, informative: AUROC about 0.85) and
  ``runs/detector_scores_noise.csv`` (pure noise);
* ``items/verified_items.csv`` and ``items/retest.csv`` (S11, with meta files): agreement with
  the labels except planted flags, skips and disagreements;
* ``data/match_stats.json`` and ``data/yield_table.csv`` (synthetic aggregates);

then runs ``scripts/verify_join.py`` (exclusions, verification summary) and the analysis
(``scripts/analyze.py``) into ``results/``, writes ``truth.json`` with the planted values, and runs
Arm 1 once more with the pure-noise detector as a control (``noise_control.json``: its gain should
sit inside the shuffled-expert null, with a large p).
The whole run is parse-free (S8 is written directly) and deterministic: every random draw comes
from a generator seeded with ``constants.derive_seed``.

Usage:
  python scripts/synthetic_dryrun.py --out runs/synthetic/ [--n-boot 199] [--n-perm 199]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from cxrdev import arm1 as A1  # noqa: E402
from cxrdev import constants as C  # noqa: E402
from cxrdev import fileio, parse, prompts  # noqa: E402
from cxrdev import metrics as M  # noqa: E402
from cxrdev.items import ITEM_COLS, read_items, selection_report  # noqa: E402
from cxrdev.metrics import S8_COLUMNS, coerce_parsed, h0_gate  # noqa: E402
from scripts import analyze, verify_join  # noqa: E402

MODELS: tuple[str, ...] = ("qwen3vl2b", "medgemma15")
TEST_PROMPTS: tuple[str, ...] = ("P0", "P1", "P2hi")
FAST = 199

#: Planted (P(yes), P(abstain)) over valid outputs, per model x prompt x group.
PLANTED: dict[str, dict[str, dict[str, tuple[float, float]]]] = {
    "qwen3vl2b": {
        "P0": {"A": (0.85, 0.0), "B": (0.20, 0.0), "U_1": (0.45, 0.0), "U_0": (0.10, 0.0), "DEV_pos": (0.85, 0.0),
               "DEV_neg": (0.20, 0.0)},
        "P1": {"A": (0.80, 0.04), "B": (0.18, 0.05), "U_1": (0.40, 0.05), "U_0": (0.08, 0.05)},
        "P2hi": {"A": (0.55, 0.25), "B": (0.08, 0.35), "U_1": (0.20, 0.35), "U_0": (0.04, 0.35)},
    },
    "medgemma15": {
        "P0": {"A": (0.60, 0.0), "B": (0.15, 0.0), "U_1": (0.55, 0.0), "U_0": (0.15, 0.0), "DEV_pos": (0.60, 0.0),
               "DEV_neg": (0.15, 0.0)},
        "P1": {"A": (0.55, 0.05), "B": (0.12, 0.05), "U_1": (0.50, 0.05), "U_0": (0.12, 0.05)},
        "P2hi": {"A": (0.40, 0.30), "B": (0.06, 0.40), "U_1": (0.30, 0.40), "U_0": (0.06, 0.40)},
    },
}
#: (unparsable, truncated) outputs per prompt x group (same for both models).
BAD_OUTPUTS: dict[str, dict[str, tuple[int, int]]] = {
    "P0": {"A": (2, 1), "B": (1, 1), "U_1": (1, 0), "U_0": (0, 1), "DEV_pos": (1, 0), "DEV_neg": (1, 0)},
    "P1": {"A": (1, 1), "B": (2, 0), "U_1": (1, 1), "U_0": (1, 0)},
    "P2hi": {"A": (2, 0), "B": (1, 1), "U_1": (2, 0), "U_0": (1, 1)},
}
N_FLIPPED_Z = 4  # P0 test items per model whose z sign disagrees with the greedy text
#: Blind rows: (z, raw output) per model x prompt x condition.
BLIND: dict[str, dict[tuple[str, str], tuple[float, str]]] = {
    "qwen3vl2b": {("P0", "BLIND-text"): (1.2, "Yes."), ("P0", "BLIND-grey"): (-0.5, "No."),
                  ("P1", "BLIND-text"): (0.4, "Yes"), ("P1", "BLIND-grey"): (-0.2, "I don't know."),
                  ("P2hi", "BLIND-text"): (-0.3, "I don't know."), ("P2hi", "BLIND-grey"): (-0.9, "No")},
    "medgemma15": {("P0", "BLIND-text"): (0.3, "Yes"), ("P0", "BLIND-grey"): (-1.4, "No"),
                   ("P1", "BLIND-text"): (-0.6, "No."), ("P1", "BLIND-grey"): (-1.1, "No."),
                   ("P2hi", "BLIND-text"): (-0.8, "I do not know."), ("P2hi", "BLIND-grey"): (-1.5, "No.")},
}
#: Planted reading outcomes on distinct read items: (arm, kind) -> count. "trach_absent" is an A image
#: whose only airway tube is a tracheostomy (RANZCR CLiP labels it "ETT normal"): read "absent" with the
#: tracheostomy flag, it must be excluded as flag_tracheostomy and never counted as a disagreement.
READING_PLAN: dict[tuple[str, str], int] = {
    ("B", "disagree"): 2, ("U", "disagree"): 1, ("A", "disagree"): 1,
    ("B", "flag_ambiguous"): 2, ("A", "flag_tracheostomy"): 1, ("A", "trach_absent"): 2,
    ("A", "flag_out_of_field"): 1, ("B", "skip"): 1, ("U", "skip"): 1,
}
DETECTOR_SHIFT = 0.9  # informative detector logit mean +/- shift, sd DETECTOR_SD
DETECTOR_SD = 1.2


def _rng(purpose: str, k: int = 0) -> np.random.Generator:
    """A generator seeded from ``constants.derive_seed(purpose)`` and a sub-stream index."""
    return np.random.default_rng([C.derive_seed(purpose), int(k)])


def _hex_ids(rng: np.random.Generator, n: int, length: int, upper: bool) -> list[str]:
    """Unique random hex ids that always contain a letter (never numeric-looking)."""
    out: list[str] = []
    seen: set[str] = set()
    while len(out) < n:
        v = "".join("0123456789abcdef"[int(d)] for d in rng.integers(0, 16, size=length))
        if v in seen or v.isdigit():
            continue
        seen.add(v)
        out.append(v.upper() if upper else v)
    return out


# ----------------------------------------------------------------------------- items (S5)
def make_items(h1_substitute: bool = False, n_two_pair_patients: int = 0, n_pairs: int = C.N_PAIRS) -> pd.DataFrame:
    """Synthetic S5 items: pairs, U strata (view balanced), DEV pool with 20 smoke items."""
    rng = _rng("pairs")
    rows: list[dict[str, Any]] = []
    rung = "two_per_patient" if n_two_pair_patients else "standard"
    followups: dict[int, int] = {}

    def base(pid: int, arm: str, ett: int, other: int, view: str) -> dict[str, Any]:
        fu = followups.get(pid, 0)
        followups[pid] = fu + int(rng.integers(1, 4))
        lines = [0, 0, 0]
        for j in rng.permutation(3)[:min(other, 3)]:
            lines[int(j)] = 1
        return {"uid": f"synthetic.uid.{len(rows) + 1:05d}", "nih_filename": f"{pid:08d}_{fu:03d}.png",
                "nih_patient_id": pid, "arm": arm, "pair_id": "", "stratum": "", "ett": ett, "ngt": lines[0],
                "ngt_incomplete": 0, "cvc": lines[1], "swan": lines[2], "n_other_lines": int(sum(lines)), "view": view,
                "followup": fu, "age": int(rng.integers(18, 90)), "sex": "M" if rng.random() < 0.55 else "F",
                "direction": "", "gap": -1, "rung": rung, "smoke": 0}

    for k in range(n_pairs):
        pid = 10_000 + (k // 2 if k < 2 * n_two_pair_patients else k - n_two_pair_patients)
        view = "AP" if rng.random() < 0.7 else "PA"
        first_is_a = bool(rng.random() < 0.5)
        recs = []
        for arm in (("A", "B") if first_is_a else ("B", "A")):
            recs.append(base(pid, arm, 1 if arm == "A" else 0, int(rng.integers(0, 3)), view))
        a, b = (recs[0], recs[1]) if first_is_a else (recs[1], recs[0])
        for r in (a, b):
            r["pair_id"] = f"p{k + 1:03d}"
            r["direction"] = "B_after_A" if b["followup"] > a["followup"] else "B_before_A"
            r["gap"] = abs(b["followup"] - a["followup"])
        rows += [a, b]
    strata = ("ge2", "eq1") if h1_substitute else ("line", "noline")
    pid = 20_000
    for s in strata:
        for i in range(C.N_U_PER_STRATUM):
            pid += 1
            other = {"line": int(rng.integers(1, 4)), "noline": 0, "ge2": int(rng.integers(2, 4)), "eq1": 1}[s]
            r = base(pid, "U", 0, other, "AP" if i < C.N_U_PER_STRATUM // 2 else "PA")
            r["stratum"] = s
            rows.append(r)
    pid = 30_000
    for s, n in (("pos", C.N_DEV_POS), ("neg", C.N_DEV_NEG)):
        for _ in range(n):
            pid += 1
            r = base(pid, "DEV", 1 if s == "pos" else 0, int(rng.integers(0, 3)), "AP" if rng.random() < 0.6 else "PA")
            r["stratum"] = s
            rows.append(r)
    items = pd.DataFrame(rows)
    ids = _hex_ids(_rng("ids"), len(items), 10, upper=False)
    items.insert(0, "item_id", ids)
    srng = _rng("smoke")
    for s in ("pos", "neg"):
        idx = items.index[(items["arm"] == "DEV") & (items["stratum"] == s)].to_numpy()
        items.loc[srng.permutation(idx)[: C.N_SMOKE // 2], "smoke"] = 1
    return items.loc[:, list(ITEM_COLS)].reset_index(drop=True)


def _group(arm: str, stratum: str) -> str:
    """Planting group of an item: "A", "B", "U_1" (line or ge2), "U_0" (noline or eq1), "DEV_pos", "DEV_neg"."""
    if arm in ("A", "B"):
        return arm
    if arm == "U":
        return "U_1" if stratum in ("line", "ge2") else "U_0"
    return f"DEV_{stratum}"


# ----------------------------------------------------------------------------- model outputs (S8)
def _log_sigmoid(z: float) -> float:
    return -math.log1p(math.exp(-z)) if z >= 0 else z - math.log1p(math.exp(z))


def _record(model: str, item: dict[str, Any] | None, condition: str, pid: str, raw: str, z: float, cov: float,
            n_tokens: int, eos: bool, rng: np.random.Generator) -> dict[str, Any]:
    spec = C.MODELS[model]
    text = prompts.prompt_text(pid)
    visual = 0 if condition == "BLIND-text" else int(spec["expected_visual_tokens"])
    if item is None:
        item_id = {"BLIND-text": "BLIND-TEXT", "BLIND-grey": "BLIND-GREY"}[condition]
        fname = ""
        img_sha = "" if condition == "BLIND-text" else fileio.sha256_bytes(b"synthetic grey 1024x1024")
    else:
        item_id, fname = item["item_id"], item["nih_filename"]
        img_sha = fileio.sha256_bytes(f"synthetic pixels {fname}".encode())
    rec = {
        "run_id": f"synthetic-{model}", "model_key": model, "model_id": spec["hf_id"], "revision": "synthetic0000",
        "transformers_version": "5.22.1", "torch_version": "2.15.0", "dtype": "bfloat16", "device": "SYNTHETIC",
        "item_id": item_id, "nih_filename": fname, "image_sha256": img_sha, "condition": condition, "prompt_id": pid,
        "prompt_sha256": prompts.prompt_sha256(text), "prompt_text": text, "visual_tokens": visual,
        "seed": C.derive_seed("model"), "max_new_tokens": C.MAX_NEW_TOKENS, "raw_output": raw,
        "n_generated_tokens": n_tokens, "ended_with_eos": eos, "parsed": parse.parse_answer(raw),
        "parser_version": parse.PARSER_VERSION, "logp_yes": math.log(cov) + _log_sigmoid(z),
        "logp_no": math.log(cov) + _log_sigmoid(-z), "logp_abstain": float("nan"), "z": z, "coverage": cov,
        "signal": "first_token", "wall_s": round(float(rng.uniform(0.25, 0.45)), 3),
        "timestamp_utc": "2026-11-11T12:00:00Z",
    }
    rec["truncated"] = parse.is_truncated(n_tokens, C.MAX_NEW_TOKENS, eos)
    return rec


def make_parsed(items: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Synthetic S8 rows with exact planted counts; returns (frame, truth counts)."""
    truth: dict[str, Any] = {}
    recs: list[dict[str, Any]] = []
    item_recs = items.to_dict("records")
    groups = [_group(r["arm"], r["stratum"]) for r in item_recs]
    group_of = {r["item_id"]: g for r, g in zip(item_recs, groups)}
    for mi, model in enumerate(MODELS):
        rng = _rng("model", mi)
        truth[model] = {}
        for pid in TEST_PROMPTS:
            truth[model][pid] = {}
            for g, (p_yes, p_abs) in PLANTED[model][pid].items():
                idx = [i for i, gg in enumerate(groups) if gg == g]
                n_unp, n_tr = BAD_OUTPUTS[pid][g]
                n_valid = len(idx) - n_unp - n_tr
                n_abs = int(round(p_abs * n_valid))
                n_yes = int(round(p_yes * n_valid))
                n_no = n_valid - n_yes - n_abs
                if n_no < 0:
                    raise ValueError(f"planted counts do not fit: {model} {pid} {g}")
                order = [idx[j] for j in rng.permutation(len(idx))]
                kinds = ["unparsable"] * n_unp + ["truncated"] * n_tr + ["abstain"] * n_abs + ["yes"] * n_yes + ["no"] * n_no
                truth[model][pid][g] = {"p_yes": p_yes, "p_abstain": p_abs, "n_total": len(idx), "n_unparsable": n_unp,
                                        "n_truncated": n_tr, "n_valid": n_valid, "n_yes": n_yes, "n_no": n_no,
                                        "n_abstain": n_abs}
                for i, kind in zip(order, kinds):
                    item = item_recs[i]
                    y = int(item["ett"])
                    eos, ntok = True, 2
                    if kind == "yes" or kind == "no":
                        raw = str(rng.choice(["Yes", "Yes.", "yes"] if kind == "yes" else ["No", "No.", "no"]))
                        correct = (kind == "yes") == (y == 1)
                        mag = float(rng.uniform(0.6, 3.0) if correct else rng.uniform(0.05, 1.2))
                        z = mag if kind == "yes" else -mag
                    elif kind == "abstain":
                        raw = str(rng.choice(["I don't know.", "I don’t know", "I'm not sure."]))
                        z, ntok = float(rng.normal(0.0, 0.6)), 5
                    elif kind == "unparsable":
                        raw = str(rng.choice(["The image shows a chest radiograph.", "Possibly.", "There is a line."]))
                        z, ntok = float(rng.normal(0.0, 0.6)), 8
                    else:  # truncated: parsable text, no end-of-sequence within the budget
                        first = "Yes" if rng.random() < 0.5 else "No"
                        raw = f"{first}, the tube is visible in the trachea with its tip located above the carina here and"
                        z = float(rng.uniform(0.05, 1.0)) * (1 if first == "Yes" else -1)
                        eos, ntok = False, C.MAX_NEW_TOKENS
                    recs.append(_record(model, item, "image", pid, raw, z, float(rng.uniform(0.92, 0.995)), ntok, eos, rng))
            if pid == "P0":
                cand = [k for k, r in enumerate(recs) if r["model_key"] == model and r["prompt_id"] == "P0"
                        and r["condition"] == "image" and r["parsed"] in ("yes", "no") and not r["truncated"]
                        and group_of[r["item_id"]] in ("A", "B", "U_1", "U_0")]
                for k in rng.permutation(cand)[:N_FLIPPED_Z]:
                    recs[int(k)]["z"] = -recs[int(k)]["z"]
                    lp, ln = recs[int(k)]["logp_yes"], recs[int(k)]["logp_no"]
                    recs[int(k)]["logp_yes"], recs[int(k)]["logp_no"] = ln, lp
        for (pid, cond), (z, raw) in BLIND[model].items():
            recs.append(_record(model, None, cond, pid, raw, z, 0.97, 3, True, rng))
    df = pd.DataFrame(recs, columns=list(S8_COLUMNS))
    return coerce_parsed(df), truth


# ----------------------------------------------------------------------------- detector (S9)
def make_detector(items: pd.DataFrame, informative: bool = True) -> pd.DataFrame:
    """Synthetic MedSigLIP scores for every item: informative (AUROC about 0.85) or pure noise."""
    rng = _rng("model", 10 if informative else 11)
    y = items["ett"].to_numpy(dtype=int)
    mean = np.where(y == 1, DETECTOR_SHIFT, -DETECTOR_SHIFT) if informative else np.zeros(len(y))
    logit = rng.normal(mean, DETECTOR_SD)
    s = 1.0 / (1.0 + np.exp(-logit))
    return pd.DataFrame({
        "item_id": items["item_id"].astype(str), "nih_filename": items["nih_filename"].astype(str), "s": s,
        "logit_with": 3.0 + logit / 2.0, "logit_without": 3.0 - logit / 2.0, "model_id": C.EXPERT["hf_id"],
        "revision": "synthetic0000", "dtype": "float32",
    })


# ----------------------------------------------------------------------------- reading (S10, S11)
def make_reading(items: pd.DataFrame, items_dir: Path) -> dict[str, Any]:
    """Reading list, key, retest list, the planted reading and retest (with meta files)."""
    rng = _rng("reading")
    a = items.loc[items["arm"] == "A"]
    chosen = pd.concat([items.loc[items["arm"].isin(["B", "U"])], a.iloc[np.sort(rng.permutation(len(a))[: C.N_READ_A])]])
    chosen = chosen.iloc[rng.permutation(len(chosen))].reset_index(drop=True)
    read_ids = _hex_ids(rng, len(chosen), 8, upper=True)
    reading = pd.DataFrame({"read_id": read_ids, "nih_filename": chosen["nih_filename"].tolist(),
                            "image_path": [f"images/{f}" for f in chosen["nih_filename"]]})
    key = pd.DataFrame({"read_id": read_ids, "item_id": chosen["item_id"].tolist()})
    response = np.where(chosen["ett"].to_numpy() == 1, "present", "absent").astype(object)
    flags = {f: np.zeros(len(chosen), dtype=int) for f in ("flag_ambiguous", "flag_tracheostomy", "flag_out_of_field")}
    used: set[int] = set()
    planted: dict[str, str] = {}
    for (arm, kind), n in READING_PLAN.items():
        pool = [i for i in rng.permutation(np.flatnonzero(chosen["arm"].to_numpy() == arm)).tolist() if i not in used][:n]
        for i in pool:
            used.add(i)
            if kind == "disagree":
                response[i] = "absent" if response[i] == "present" else "present"
                planted[chosen["item_id"].iloc[i]] = "label_disagreement"
            elif kind == "skip":
                response[i] = "skip"
                planted[chosen["item_id"].iloc[i]] = "skip"
            elif kind == "trach_absent":
                response[i] = "absent"
                flags["flag_tracheostomy"][i] = 1
                planted[chosen["item_id"].iloc[i]] = "flag_tracheostomy"
            else:
                flags[kind][i] = 1
                planted[chosen["item_id"].iloc[i]] = kind
    t0 = pd.Timestamp("2026-11-10T09:00:00Z")
    verified = pd.DataFrame({
        "read_id": read_ids, "nih_filename": reading["nih_filename"], "response": response, **flags, "note": "",
        "seconds": np.round(rng.uniform(8, 40, len(chosen)), 3), "order_index": np.arange(len(chosen)),
        "timestamp_utc": [(t0 + pd.Timedelta(seconds=30 * i)).strftime("%Y-%m-%dT%H:%M:%SZ") for i in range(len(chosen))],
    })
    rr = _rng("retest")
    pick = rr.permutation(len(reading))[: C.N_RETEST]
    retest_list = reading.iloc[pick].reset_index(drop=True)
    rt = verified.set_index("read_id").loc[retest_list["read_id"]].reset_index()
    changeable = [i for i in range(len(rt)) if rt.loc[i, "response"] != "skip"]
    for i in rr.permutation(changeable)[:3]:
        rt.loc[int(i), "response"] = "absent" if rt.loc[int(i), "response"] == "present" else "present"
    t1 = pd.Timestamp("2026-11-12T15:00:00Z")
    rt["order_index"] = np.arange(len(rt))
    rt["timestamp_utc"] = [(t1 + pd.Timedelta(seconds=30 * i)).strftime("%Y-%m-%dT%H:%M:%SZ") for i in range(len(rt))]
    paths = {}
    for name, df in (("reading_list", reading), ("reading_key", key), ("retest_list", retest_list)):
        paths[name] = items_dir / f"{name}.csv"
        fileio.write_csv(df, paths[name])
    for name, df, mode, saved in (("verified_items", verified, "read", "2026-11-10T11:30:00Z"),
                                  ("retest", rt.loc[:, list(verified.columns)], "retest", "2026-11-12T16:00:00Z")):
        p = items_dir / f"{name}.csv"
        sha = fileio.write_csv(df, p)
        meta = {"sha256": sha, "saved_utc": saved, "seed": C.derive_seed("reading" if mode == "read" else "retest"),
                "n": int(len(df)), "mode": mode, "tool_version": "synthetic"}
        p.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8", newline="\n")
        paths[name] = p
    return {"paths": paths, "planted_exclusions": planted, "n_read": int(len(chosen))}


# ----------------------------------------------------------------------------- aggregates
def make_aggregates(items: pd.DataFrame, items_dir: Path, data_dir: Path, h1_substitute: bool) -> dict[str, Path]:
    """Synthetic match statistics, yield table and selection report."""
    match_stats = {
        "synthetic": True, "n_ranzcr": 30083, "n_ranzcr_patients": 3255,
        "status_counts": {"matched": 27950, "ambiguous": 812, "unmatched": 1121, "duplicate_conflict": 120,
                          "patient_conflict": 80},
        "route_counts": {"margin": 27010, "ncc_tiebreak": 940, "none": 2133}, "match_rate": 27950 / 30083,
        "ambiguous_count": 812, "ambiguous_share_within_patient": 0.71, "cut": 38, "n_bits": 256,
        "matched_patients": 3101, "run": {"n_spotcheck_pairs": 60, "dimkey_share_found": 0.985},
        "config": {"margin_bits": 16, "cut": None, "ncc_min": 0.95, "ncc_margin": 0.02, "topk_ncc": 5,
                   "use_dimension_key": True},
    }
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "match_stats.json").write_text(json.dumps(match_stats, indent=2) + "\n", encoding="utf-8", newline="\n")
    yield_rows = [("n_images", 30083), ("n_patients", 3255), ("ett_present_images", 7650), ("ett_absent_images", 22433),
                  ("patients_with_both_ett_states", 412), ("ett_absent_line_images", 21800),
                  ("ett_absent_line_patients", 2990), ("ett_absent_noline_images", 633),
                  ("ett_absent_noline_patients", 25 if h1_substitute else 214), ("ett_absent_ge2_patients", 1130),
                  ("ett_absent_eq1_patients", 2410), ("ngt_incomplete_images", 274)]
    fileio.write_csv(pd.DataFrame(yield_rows, columns=["metric", "value"]), data_dir / "yield_table.csv")
    rung = str(items["rung"].iloc[0])
    core = {"seed": C.SEED, "rung": rung, "h1_substitute": bool(h1_substitute), "n_eligible_patients": 150,
            "pairs": {"n_selected": int((items["arm"] == "A").sum()), "rung": rung, "synthetic": True},
            "u": {"h1_substitute": bool(h1_substitute),
                  "h1_contrast": ({"exposed": "ge2", "reference": "eq1"} if h1_substitute
                                  else {"exposed": "line", "reference": "noline"})},
            "dev": {"selected": {"pos": C.N_DEV_POS, "neg": C.N_DEV_NEG}}}
    rep = selection_report(items, core)
    (items_dir / "selection_report.json").write_text(json.dumps(rep, indent=2, sort_keys=True) + "\n", encoding="utf-8",
                                                     newline="\n")
    return {"match_stats": data_dir / "match_stats.json", "yield": data_dir / "yield_table.csv",
            "selection": items_dir / "selection_report.json"}


def make_synthetic(out_dir: str | Path, *, h1_substitute: bool = False, n_two_pair_patients: int = 0) -> tuple[dict[str, Path], dict[str, Any]]:
    """Write every synthetic input under ``out_dir``; returns (paths, truth)."""
    out = Path(out_dir)
    items_dir, runs_dir, data_dir = out / "items", out / "runs", out / "data"
    for d in (items_dir, runs_dir, data_dir):
        d.mkdir(parents=True, exist_ok=True)
    items = make_items(h1_substitute=h1_substitute, n_two_pair_patients=n_two_pair_patients)
    paths: dict[str, Path] = {"items": items_dir / "items.csv", "parsed": runs_dir / "parsed_results.csv",
                              "detector": runs_dir / "detector_scores.csv", "detector_noise": runs_dir / "detector_scores_noise.csv"}
    fileio.write_csv(items, paths["items"])
    parsed, counts = make_parsed(items)
    fileio.write_csv(parsed, paths["parsed"])
    fileio.write_csv(make_detector(items, True), paths["detector"])
    fileio.write_csv(make_detector(items, False), paths["detector_noise"])
    # run settings sidecar as pod/run_vlm.py writes it (settings only): the source of run.gpu_name
    (runs_dir / "qwen3vl2b.meta.json").write_text(
        json.dumps({"sessions": [{"mode": "items", "backend": {"gpu_name": "NVIDIA RTX A5000 (synthetic)"}}]}, indent=2)
        + "\n", encoding="utf-8", newline="\n")
    paths["run_dir"] = runs_dir
    reading = make_reading(items, items_dir)
    paths.update(reading["paths"])
    paths.update(make_aggregates(items, items_dir, data_dir, h1_substitute))
    arm_of = dict(zip(items["item_id"], items["arm"]))
    planted = reading["planted_exclusions"]
    truth: dict[str, Any] = {
        "synthetic": True, "models": list(MODELS), "planted": PLANTED, "bad_outputs": BAD_OUTPUTS, "counts": counts,
        "n_flipped_z": N_FLIPPED_Z, "h1_substitute": bool(h1_substitute), "rung": str(items["rung"].iloc[0]),
        "planted_exclusions": planted,
        "exclusions_by_arm": {a: sum(1 for i in planted if arm_of[i] == a) for a in ("A", "B", "U")},
        "detector": {"shift": DETECTOR_SHIFT, "sd": DETECTOR_SD,
                     "expected_auroc": float(0.5 * math.erfc(-(2 * DETECTOR_SHIFT / DETECTOR_SD) / 2.0))},
        "h0_expected": {},
    }
    for model in MODELS:
        p0 = PLANTED[model]["P0"]
        hit, far_b = p0["A"][0], p0["B"][0]
        n_a, n_b, n_u = (int((items["arm"] == a).sum()) for a in ("A", "B", "U"))
        n_u1 = int(((items["arm"] == "U") & items["stratum"].isin(["line", "ge2"])).sum())
        yes_rate = (hit * n_a + far_b * n_b + p0["U_1"][0] * n_u1 + p0["U_0"][0] * (n_u - n_u1)) / (n_a + n_b + n_u)
        truth["h0_expected"][model] = h0_gate((hit + 1 - far_b) / 2, hit, yes_rate)
    (out / "truth.json").write_text(json.dumps(truth, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    paths["truth"] = out / "truth.json"
    return paths, truth


def noise_control(paths: dict[str, Path], n_perm: int, out_path: Path | None = None) -> dict[str, Any]:
    """Arm 1 per model with the pure-noise detector (S9 noise file), after the reading's exclusions.

    Returns, per model, the observed balanced-accuracy gain (fused minus beta = 0), its one-sided
    shuffled-expert p, the null's 2.5% and 97.5% quantiles and the detector's dev-pool AUROC (about
    0.5). Written to ``out_path`` when given. No bootstrap (n_boot = 0): only the permutation test.
    """
    items = read_items(paths["items"])
    frame = M.scoring_frame(M.read_parsed(paths["parsed"]), items)
    excl = M.read_exclusions(paths["exclusions"]) if paths.get("exclusions") else None
    jf = A1.join_detector(frame, A1.read_detector(paths["detector_noise"]))
    out: dict[str, Any] = {}
    for mk in MODELS:
        r = A1.run_arm1(jf, jf, mk, n_perm=n_perm, n_boot=0, exclusions=excl, with_oracle=False)
        p = r["shuffled_expert"]
        out[mk] = {"gain": p["observed_gain"], "p_value": p["p_value"], "n_perm": p["n_perm"],
                   "null_mean": p["null_mean"], "null_q025": p["null_q025"], "null_q975": p["null_q975"],
                   "dev_auroc_s": (r.get("dev") or {}).get("auroc_s", float("nan")), "n": p["n"]}
    if out_path is not None:
        out_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return out


def run_dryrun(
    out_dir: str | Path,
    n_boot: int = FAST,
    n_perm: int = FAST,
    *,
    h1_substitute: bool = False,
    n_two_pair_patients: int = 0,
    log: Callable[[str], None] | None = print,
) -> dict[str, Any]:
    """Generate the synthetic inputs, run verify_join, the analysis and the noise-detector control.

    Returns ``paths``, ``truth``, ``verification`` (summary), ``analysis`` (``analyze.run_analysis``
    result), ``noise_control`` and ``results_dir``.
    """
    out = Path(out_dir)
    paths, truth = make_synthetic(out, h1_substitute=h1_substitute, n_two_pair_patients=n_two_pair_patients)
    results_dir = out / "results"
    paths["exclusions"] = out / "items" / "exclusions.csv"
    paths["verification"] = results_dir / "verification_summary.json"
    vsum = verify_join.run(paths["items"], paths["reading_key"], paths["verified_items"], paths["exclusions"],
                           paths["verification"], retest_path=paths["retest"])
    inputs = analyze.AnalysisInputs(items=paths["items"], parsed=paths["parsed"], detector=paths["detector"],
                                    exclusions=paths["exclusions"], match_stats=paths["match_stats"],
                                    yield_table=paths["yield"], selection=paths["selection"],
                                    verification=paths["verification"], run_dir=paths["run_dir"],
                                    spotcheck_correct=59)
    res = analyze.run_analysis(inputs, results_dir, n_boot, n_perm, log=log)
    paths["noise_control"] = out / "noise_control.json"
    noise = noise_control(paths, n_perm, paths["noise_control"])
    return {"paths": paths, "truth": truth, "verification": vsum, "analysis": res, "noise_control": noise,
            "results_dir": results_dir}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="runs/synthetic/")
    ap.add_argument("--n-boot", type=int, default=FAST)
    ap.add_argument("--n-perm", type=int, default=FAST)
    ap.add_argument("--h1-substitute", action="store_true", help="use the ge2 / eq1 U strata")
    ap.add_argument("--two-pair-patients", type=int, default=0, help="patients with two pairs (two_per_patient rung)")
    a = ap.parse_args(argv)
    r = run_dryrun(a.out, a.n_boot, a.n_perm, h1_substitute=a.h1_substitute, n_two_pair_patients=a.two_pair_patients)
    h = r["analysis"]["hypotheses"]
    for fam in ("H1", "H3", "H5"):
        for t in (h.get(fam) or {}).get("primary", []):
            print(f"{fam} {t['model_key']}: estimate {t['estimate']:.3f} p {t['p_value']:.3g} p_holm {t['p_holm']:.3g}")
    print(f"H0: { {k: v['class'] for k, v in h.get('H0', {}).items()} } (planted {r['truth']['h0_expected']})")
    for mk, nc in r["noise_control"].items():
        print(f"noise-detector control {mk}: gain {nc['gain']:.3f} p {nc['p_value']:.3g} "
              f"(null 95% range {nc['null_q025']:.3f} to {nc['null_q975']:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
