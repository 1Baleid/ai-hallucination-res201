"""Run the Stage 1 analysis: Tables 1-3, Figure 1 (and PLUS Figure 2), hypotheses.json, numbers.json.

Inputs (each optional; whatever depends on a missing input is skipped and the log says so):
  --items         items/items.csv (S5)                      needed for every model metric
  --parsed        runs/parsed_results.csv (S8)              model metrics, H0, H1, H3, blind rows
  --detector      runs/detector_scores.csv (S9)             Arm 1, H5, the verifier rows of Table 1
  --exclusions    items/exclusions.csv (S12)                without it every item is kept (reading pending)
  --match-stats   data/kaggle/res201-match/match_stats.json Table 1 matching rows
  --yield         data/kaggle/res201-hash/yield_table.csv   Table 1 yield rows
  --selection     items/selection_report.json               Table 1 selection rows
  --verification  results/verification_summary.json        Table 1 reading rows, kappa
  --run-dir       runs/<run_id>                             run.gpu_name (from the pod's run settings files)
Outputs in --out (default results/): table1/2/3 .csv/.md/.tex, figure1 .pdf/.png (figure2 for the
PLUS risk-coverage plot), hypotheses.json (structured tests), numbers.json (a flat dict of every
number the report may quote, keyed by stable names such as "qwen3vl2b.P0.hit_A", ".lo", ".hi",
".n"), metrics.json (every summary) and analysis_log.txt. All outputs are aggregates.

Resamples: --n-boot and --n-perm default to constants.N_BOOT and N_PERM (9,999); --fast sets both
to 199 for development. Seeds: constants.derive_seed("bootstrap") and ("permutation").

Numbers that no input file carries (for example the 100-output parser hand check) are added with
--number KEY=VALUE (repeatable); a key the analysis computes itself is never overwritten. An output
file of an earlier run that this run does not produce (for example table3 when the detector scores
are missing) is removed from --out and the log says so, so results/ never mixes two runs.

Usage:
  python scripts/analyze.py --items items/items.csv --parsed runs/parsed_results.csv
      --detector runs/detector_scores.csv --exclusions items/exclusions.csv
      --match-stats data/kaggle/res201-match/match_stats.json --yield data/kaggle/res201-hash/yield_table.csv
      --selection items/selection_report.json --verification results/verification_summary.json --out results/
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from cxrdev import arm1 as A1  # noqa: E402
from cxrdev import constants as C  # noqa: E402
from cxrdev import figures as F  # noqa: E402
from cxrdev import hypotheses as H  # noqa: E402
from cxrdev import metrics as M  # noqa: E402
from cxrdev import tables as T  # noqa: E402
from cxrdev.fileio import sha256_file  # noqa: E402

FAST_N = 199
PLUS_H2 = ("chexone", "qwen25vl3b")  # CheXOne minus Qwen2.5-VL-3B
OUTPUT_FILES: tuple[str, ...] = (
    "table1.csv", "table1.md", "table1.tex", "table2.csv", "table2.md", "table2.tex", "table3.csv", "table3.md",
    "table3.tex", "figure1.pdf", "figure1.png", "hypotheses.json", "numbers.json", "metrics.json", "analysis_log.txt",
)
#: Every file this script may write into --out (PLUS Figure 2 included); stale ones are removed.
MANAGED_FILES: tuple[str, ...] = OUTPUT_FILES + ("figure2.pdf", "figure2.png")
#: numbers.json keys must be usable as ``[[NUM:key]]`` placeholders (scripts/build_report.py).
NUMBER_KEY_RE = re.compile(r"[A-Za-z0-9_.\-]+")


@dataclass
class AnalysisInputs:
    """Paths of the analysis inputs (None or a missing file = not available).

    ``spotcheck_correct`` (pairs judged correct in the 60-pair spot check) goes to Table 1 and
    numbers.json; ``extra`` holds further numbers for numbers.json (key -> value), added only
    where the analysis does not compute the key itself.
    """

    items: Path | None = None
    parsed: Path | None = None
    detector: Path | None = None
    exclusions: Path | None = None
    match_stats: Path | None = None
    yield_table: Path | None = None
    selection: Path | None = None
    verification: Path | None = None
    run_dir: Path | None = None
    example_pair: Path | Mapping[str, Any] | None = None
    spotcheck_correct: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class _Log:
    """Collects log lines (and echoes them through ``sink``)."""

    def __init__(self, sink: Callable[[str], None] | None) -> None:
        self.lines: list[str] = []
        self.sink = sink

    def __call__(self, msg: str) -> None:
        self.lines.append(msg)
        if self.sink is not None:
            self.sink(msg)


# ----------------------------------------------------------------------------- JSON helpers
def _jsonable(obj: Any) -> Any:
    """Plain JSON types: numpy scalars to Python, tuples to lists, DataFrames to records, paths to str.

    NaN and infinite floats become None, so every JSON output is valid JSON (null, never NaN).
    """
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, pd.DataFrame):
        return [_jsonable(r) for r in obj.to_dict("records")]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj) if math.isfinite(float(obj)) else None
    if isinstance(obj, Path):
        return obj.name
    return obj


def _dump(obj: Any, path: Path) -> None:
    """Write sorted-key JSON; undefined numbers are null (``allow_nan=False`` guards against NaN)."""
    path.write_text(json.dumps(_jsonable(obj), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8",
                    newline="\n")


# ----------------------------------------------------------------------------- numbers.json helpers
def put(numbers: dict[str, Any], key: str, value: Any) -> None:
    """Add one number (numpy types converted; None skipped; NaN kept as None, written as null)."""
    if value is None:
        return
    numbers[key] = _jsonable(value)


def put_null(numbers: dict[str, Any], key: str, source: Mapping[str, Any], field: str) -> None:
    """Add ``source[field]`` under ``key``, keeping a present-but-undefined value (None or NaN) as null."""
    if field in source:
        numbers[key] = _jsonable(source[field])


def put_stat(numbers: dict[str, Any], prefix: str, d: Mapping[str, Any] | None) -> None:
    """Add estimate (as ``prefix``), ``.lo``, ``.hi``, ``.n``, ``.k`` and ``.method`` from a result dict."""
    if not d:
        return
    put(numbers, prefix, d.get("estimate"))
    for k in ("lo", "hi", "n", "k"):
        if k in d:
            put(numbers, f"{prefix}.{k}", d[k])


def summary_numbers(numbers: dict[str, Any], s: Mapping[str, Any]) -> None:
    """Flatten one ``metrics.summarise`` result into ``numbers``."""
    mk, pid = s["model_key"], s["prompt_id"]
    if s.get("condition", "image") == "BLIND-demo":
        prefix = f"demo.{mk}.{pid}"
    else:
        prefix = f"{mk}.{pid}.sens" if s.get("unparsable_as_wrong") else f"{mk}.{pid}"
    for k, v in s.items():
        if isinstance(v, Mapping) and "estimate" in v:
            put_stat(numbers, f"{prefix}.{k}", v)
    de = s.get("design_effect") or {}
    put(numbers, f"{prefix}.deff", de.get("deff"))
    put(numbers, f"{prefix}.icc", de.get("icc"))
    for k in ("n_rows", "n_rows_excluded", "flag_unparsable", "coverage_median", "signal"):
        if k in s:
            put(numbers, f"{prefix}.{k}", s[k])


def hypothesis_numbers(numbers: dict[str, Any], prefix: str, r: Mapping[str, Any]) -> None:
    """Flatten a hypothesis result (``h1``/``h3``/``h5``/``h2``) into ``numbers`` under ``prefix``."""
    for k in ("estimate", "lo", "hi", "p_value", "p_holm", "n", "n_clusters", "tested", "b", "c", "n_dropped",
              "n_candidates", "n_exposed", "n_reference", "n_perm", "null_mean", "null_q025", "null_q975", "substitute",
              "mcnemar_p", "p_value_descriptive", "family_m", "method", "p_flag"):
        if k in r:
            put(numbers, f"{prefix}.{k}", r[k])
    for k in ("rate_exposed", "rate_reference", "abstention_ref", "abstention_pen", "hfr_a", "hfr_b", "ba_diff"):
        if isinstance(r.get(k), Mapping):
            put_stat(numbers, f"{prefix}.{k}", r[k])
    chk = r.get("mcnemar_check")
    if isinstance(chk, Mapping):
        put(numbers, f"{prefix}.mcnemar_check.p_value", chk.get("p_value"))
        put(numbers, f"{prefix}.mcnemar_check.n", chk.get("n"))


def arm1_numbers(numbers: dict[str, Any], res: Mapping[str, Any]) -> None:
    """Flatten an ``arm1.run_arm1`` result into ``numbers`` (prefix ``arm1.<model>``)."""
    mk = res["model_key"]
    p = f"arm1.{mk}"
    for k in ("n", "n_A", "n_B", "n_U", "n_dropped_no_z", "n_dropped_no_score", "tau", "beta_dev"):
        put(numbers, f"{p}.{k}", res.get(k))
    if res.get("skipped_reason"):
        put(numbers, f"{p}.skipped", res["skipped_reason"])
        return
    put_stat(numbers, f"{p}.agreement_greedy_beta0", res.get("agreement_greedy_beta0"))
    put(numbers, f"{p}.b_matched", (res.get("matched_bias") or {}).get("b"))
    put(numbers, f"{p}.matched_yes_rate", (res.get("matched_bias") or {}).get("yes_rate"))
    orc = res.get("oracle") or {}
    put(numbers, f"{p}.share_abs_z_gt_gamma", orc.get("share_abs_z_gt_gamma"))
    put(numbers, f"{p}.clipped_oracle_ba", orc.get("clipped_oracle_ba"))
    dev = res.get("dev") or {}
    put(numbers, f"{p}.dev_auroc_s", dev.get("auroc_s"))
    put(numbers, f"{p}.dev_auroc_z", dev.get("auroc_z"))
    for r in res.get("rows", []):
        rp = f"{p}.{r['row']}"
        if r.get("ba"):
            put_stat(numbers, f"{rp}.ba", r["ba"])
        if r.get("delta_ba"):
            put_stat(numbers, f"{rp}.delta", r["delta_ba"])
        for k in ("hit_A", "FAR_B", "FAR_U", "yes_rate", "coverage", "risk"):
            if isinstance(r.get(k), Mapping):
                put_stat(numbers, f"{rp}.{k}", r[k])
        if r.get("p_value") is not None:
            put(numbers, f"{rp}.p_value", r["p_value"])
        if r.get("param") is not None:
            put(numbers, f"{rp}.param", r["param"])
        if r.get("fused_minus_matched"):
            put_stat(numbers, f"{rp}.fused_minus_matched", r["fused_minus_matched"])


def input_numbers(
    numbers: dict[str, Any],
    match_stats: Mapping[str, Any] | None,
    spotcheck_correct: int | None,
    yield_df: pd.DataFrame | None,
    items: pd.DataFrame | None,
    selection: Mapping[str, Any] | None,
    verification: Mapping[str, Any] | None,
    exclusions: pd.DataFrame | None,
) -> None:
    """Numbers from the non-model inputs: matching, yield, selection, verification, exclusions."""
    if match_stats:
        sc = match_stats.get("status_counts", {}) or {}
        put(numbers, "match.rate", match_stats.get("match_rate"))
        put(numbers, "match.n_ranzcr", match_stats.get("n_ranzcr"))
        put(numbers, "match.n_matched", sc.get("matched"))
        put(numbers, "match.n_ambiguous", sc.get("ambiguous", match_stats.get("ambiguous_count")))
        put(numbers, "match.n_unmatched", sc.get("unmatched"))
        put(numbers, "match.n_duplicate_conflict", sc.get("duplicate_conflict"))
        put(numbers, "match.n_patient_conflict", sc.get("patient_conflict"))
        put(numbers, "match.cut_bits", match_stats.get("cut"))
        # undefined (no ambiguous image): written as null so the report shows "n/a"
        put_null(numbers, "match.ambiguous_within_patient_share", match_stats, "ambiguous_share_within_patient")
        put(numbers, "match.matched_patients", match_stats.get("matched_patients"))
        put(numbers, "match.spotcheck_correct", match_stats.get("spotcheck_correct"))
        run = match_stats.get("run") or {}
        put(numbers, "match.dimkey_share_found", run.get("dimkey_share_found", match_stats.get("dimkey_share_found")))
        cfg = match_stats.get("config") or run.get("config_used") or {}
        if "use_dimension_key" in cfg:
            put(numbers, "match.dimkey_used", bool(cfg["use_dimension_key"]))
    if spotcheck_correct is not None:
        put(numbers, "match.spotcheck_correct", int(spotcheck_correct))
    if yield_df is not None:
        for r in yield_df.itertuples(index=False):
            put(numbers, f"yield.{r.metric}", int(r.value) if pd.notna(r.value) else None)
    if items is not None:
        arm = items["arm"].astype(str)
        a = items.loc[arm == "A"]
        put(numbers, "pairs.n", int(len(a)))
        put(numbers, "pairs.n_b_after_a", int((a["direction"] == "B_after_A").sum()))
        put(numbers, "pairs.n_b_before_a", int((a["direction"] == "B_before_A").sum()))
        if len(a):
            g = a["gap"].to_numpy(dtype=float)
            put(numbers, "pairs.gap_median", float(np.median(g)))
            put(numbers, "pairs.gap_q1", float(np.percentile(g, 25)))
            put(numbers, "pairs.gap_q3", float(np.percentile(g, 75)))
        rungs = sorted(set(items["rung"].astype(str)))
        put(numbers, "pairs.rung", rungs[0] if len(rungs) == 1 else "+".join(rungs))
        u = items.loc[arm == "U"]
        for s, n in u["stratum"].astype(str).value_counts().sort_index().items():
            put(numbers, f"u.{s}.n", int(n))
        # Counts by number of other labelled lines for every stratum name, whichever design is active,
        # so report placeholders resolve under both the standard and the substitute H1 contrast.
        if "n_other_lines" in u.columns:
            lines = pd.to_numeric(u["n_other_lines"], errors="coerce")
            by_lines = {"line": lines >= 1, "noline": lines == 0, "ge2": lines >= 2, "eq1": lines == 1}
            for s, mask in by_lines.items():
                if f"u.{s}.n" not in numbers:
                    put(numbers, f"u.{s}.n", int(mask.sum()))
        put(numbers, "u.h1_substitute", bool({"ge2", "eq1"} <= set(u["stratum"].astype(str))))
        d = items.loc[arm == "DEV"]
        for s in ("pos", "neg"):
            put(numbers, f"dev.{s}.n", int((d["stratum"] == s).sum()))
        put(numbers, "items.n_test", int(arm.isin(["A", "B", "U"]).sum()))
        put(numbers, "items.n_patients", int(items["nih_patient_id"].nunique()))
    if selection:
        put(numbers, "pairs.eligible_patients", selection.get("n_eligible_patients"))
        if selection.get("rung") is not None:
            put(numbers, "pairs.rung", selection["rung"])
        if selection.get("h1_substitute") is not None:
            put(numbers, "u.h1_substitute", bool(selection["h1_substitute"]))
        ex = selection.get("excluded_filenames") or {}
        put(numbers, "select.excluded_filenames.n_listed", ex.get("n_listed"))
        put(numbers, "select.excluded_filenames.n_present", ex.get("n_present_in_matched_pool"))
    if verification:
        put_stat(numbers, "verify.agreement", verification.get("agreement"))
        put(numbers, "verify.n_read", verification.get("n_read"))
        put(numbers, "verify.n_disagreements", verification.get("n_disagreements"))
        put(numbers, "verify.kappa_vs_ranzcr", verification.get("kappa_vs_ranzcr"))
        for a_ in ("A", "B", "U"):
            put_stat(numbers, f"verify.agreement.{a_}", (verification.get("by_arm", {}).get(a_) or {}).get("agreement"))
        for f_, n in (verification.get("flags") or {}).items():
            put(numbers, f"verify.{f_}", n)
        rt = verification.get("retest") or {}
        put(numbers, "verify.kappa", rt.get("kappa"))
        put(numbers, "verify.kappa_n", rt.get("n_compared"))
        put(numbers, "verify.retest_gap_hours", rt.get("gap_hours_after_read"))
        ex = verification.get("exclusions") or {}
        put(numbers, "verify.n_excluded", ex.get("n_items"))
    if exclusions is not None:
        put(numbers, "excl.n_items", int(exclusions["item_id"].nunique()))
        for reason in M.EXCLUSION_REASONS:
            put(numbers, f"excl.{reason}.n", int((exclusions["reason"] == reason).sum()))
        if items is not None:
            arm_of = dict(zip(items["item_id"].astype(str), items["arm"].astype(str)))
            arms = exclusions["item_id"].astype(str).map(arm_of)
            for a_ in ("A", "B", "U", "DEV"):
                put(numbers, f"excl.{a_}.n", int((arms == a_).sum()))


def run_numbers(numbers: dict[str, Any], parsed: pd.DataFrame, detector: pd.DataFrame | None) -> None:
    """Run metadata (revisions, versions, device, seconds per item) from S8 and S9."""
    img = parsed.loc[parsed["condition"] == "image"]
    for key in ("transformers_version", "torch_version", "device"):
        vals = sorted(set(parsed[key].astype(str)) - {""})
        if vals:
            put(numbers, f"run.{key}", vals[0] if len(vals) == 1 else "; ".join(vals))
    for mk in M.ordered_models(parsed["model_key"]):
        sub = parsed.loc[parsed["model_key"] == mk]
        si = img.loc[img["model_key"] == mk]
        for key in ("revision", "model_id", "dtype", "device", "signal"):
            vals = sorted(set(sub[key].astype(str)) - {""})
            if vals:
                put(numbers, f"run.{mk}.{key}", vals[0] if len(vals) == 1 else "; ".join(vals))
        put(numbers, f"run.{mk}.n_rows", int(len(sub)))
        if len(si):
            w = si["wall_s"].to_numpy(dtype=float)
            w = w[np.isfinite(w)]
            if w.size:
                put(numbers, f"run.{mk}.s_per_item", round(float(np.median(w)), 2))
            put(numbers, f"run.{mk}.visual_tokens", int(np.median(si["visual_tokens"].to_numpy())))
    if detector is not None:
        for key in ("revision", "model_id", "dtype"):  # dtype: optional S9 column written by pod/run_medsiglip.py
            if key not in detector.columns:
                continue
            vals = sorted(set(detector[key].astype(str)) - {""})
            if vals:
                put(numbers, f"run.medsiglip.{key}", vals[0] if len(vals) == 1 else "; ".join(vals))


def gpu_names(run_dir: str | Path) -> list[str]:
    """GPU names recorded by the pod under ``run_dir`` (sorted, unique; empty when none is recorded).

    Sources: the ``backend.gpu_name`` of every session in the run settings sidecars
    (``<model>.jsonl`` -> ``<model>.meta.json``, written by ``pod/run_vlm.py``) and the per-model
    ``gpu_name`` of ``smoke_report.json``. Settings files only; no model output is read.
    """
    root = Path(run_dir)
    names: set[str] = set()
    if not root.is_dir():
        return []
    for p in sorted(root.rglob("*.meta.json")):
        try:
            meta = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        sessions = meta.get("sessions") if isinstance(meta, dict) else None
        for s in sessions if isinstance(sessions, list) else []:
            name = ((s or {}).get("backend") or {}).get("gpu_name") if isinstance(s, dict) else None
            if name:
                names.add(str(name))
        if isinstance(meta, dict) and meta.get("gpu_name"):  # pod/run_medsiglip.py sidecar
            names.add(str(meta["gpu_name"]))
    for p in sorted(root.rglob("smoke_report.json")):
        try:
            rep = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for sec in ((rep or {}).get("models") or {}).values() if isinstance(rep, dict) else []:
            if isinstance(sec, dict) and sec.get("gpu_name"):
                names.add(str(sec["gpu_name"]))
    return sorted(names)


# ----------------------------------------------------------------------------- the analysis
def _load(path: Path | None, what: str, reader: Callable[[Path], Any], log: _Log, missing_note: str) -> Any:
    if path is None:
        log(f"[skip] {what}: no path given; {missing_note}")
        return None
    p = Path(path)
    if not p.exists():
        log(f"[skip] {what}: {p.name} not found; {missing_note}")
        return None
    log(f"[input] {what}: {p.name} (sha256 {sha256_file(p)[:12]})")
    return reader(p)


def _read_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def run_analysis(
    inputs: AnalysisInputs,
    out_dir: str | Path,
    n_boot: int = C.N_BOOT,
    n_perm: int = C.N_PERM,
    *,
    log: Callable[[str], None] | None = print,
    plus_figure: bool = True,
) -> dict[str, Any]:
    """Run every analysis step whose inputs exist and write the outputs into ``out_dir``.

    Returns {"numbers", "hypotheses", "summaries", "arm1", "files", "log"}.
    """
    L = _Log(log)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    seed_b, seed_p = C.derive_seed("bootstrap"), C.derive_seed("permutation")
    L(f"Stage 1 analysis: n_boot={n_boot}, n_perm={n_perm}, bootstrap seed {seed_b}, permutation seed {seed_p}")

    from cxrdev.items import read_items

    items = _load(inputs.items, "items (S5)", read_items, L, "no model metric can be computed")
    parsed = _load(inputs.parsed, "parsed results (S8)", M.read_parsed, L,
                   "metrics, H0, H1, H3, blind rows, Tables 2-3 and Figure 1 are skipped")
    detector = _load(inputs.detector, "detector scores (S9)", A1.read_detector, L,
                     "Arm 1, H5, Table 3 and the verifier rows of Table 1 are skipped")
    exclusions = _load(inputs.exclusions, "exclusions (S12)", M.read_exclusions, L,
                       "no item is excluded (the blind reading is pending); results are provisional")
    match_stats = _load(inputs.match_stats, "match statistics", _read_json, L, "Table 1 matching rows say not available")
    yield_df = _load(inputs.yield_table, "yield table", lambda p: pd.read_csv(p), L, "Table 1 yield rows say not available")
    selection = _load(inputs.selection, "selection report", _read_json, L, "Table 1 selection rows come from items only")
    verification = _load(inputs.verification, "verification summary", _read_json, L,
                         "Table 1 reading rows and kappa say not available")
    example_pair = inputs.example_pair
    if isinstance(example_pair, (str, Path)):
        example_pair = _load(Path(example_pair), "example pair", _read_json, L, "Figure 1 panel a shows placeholder text")

    numbers: dict[str, Any] = {}
    input_numbers(numbers, match_stats, inputs.spotcheck_correct, yield_df, items, selection, verification, exclusions)
    hyp: dict[str, Any] = {"meta": {
        "n_boot": int(n_boot), "n_perm": int(n_perm), "seed_bootstrap": seed_b, "seed_permutation": seed_p,
        "alpha": C.ALPHA, "min_cell_n": C.MIN_CELL_N, "gate": dict(C.GATE),
        "inputs": {name: {"file": Path(p).name, "sha256": sha256_file(p)}
                   for name, p in (("items", inputs.items), ("parsed", inputs.parsed), ("detector", inputs.detector),
                                   ("exclusions", inputs.exclusions), ("match_stats", inputs.match_stats),
                                   ("yield", inputs.yield_table), ("selection", inputs.selection),
                                   ("verification", inputs.verification)) if p is not None and Path(p).exists()},
        "exclusions_applied": exclusions is not None,
    }}
    summaries: list[dict[str, Any]] = []
    demo_summaries: list[dict[str, Any]] = []
    blind_frames: list[pd.DataFrame] = []
    arm1_results: dict[str, dict[str, Any]] = {}
    detector_info: dict[str, Any] | None = None
    files: list[Path] = []

    if items is not None and detector is not None:
        detector_info = A1.detector_summary(items, detector, exclusions, n_boot, seed_b)
        if detector_info.get("dev_auroc"):
            put_stat(numbers, "detector.dev_auroc", detector_info["dev_auroc"])
            put(numbers, "detector.tau", detector_info["tau"])
            for a_, r in (detector_info.get("agreement") or {}).items():
                put_stat(numbers, f"detector.agree.{a_}", r)
        put(numbers, "detector.n_missing", detector_info.get("n_missing"))
        hyp["detector"] = detector_info

    if items is not None and parsed is not None:
        frame = M.scoring_frame(parsed, items, "image")
        rungs = sorted(set(items["rung"].astype(str)))
        rung = (selection or {}).get("rung") or (rungs[0] if len(rungs) == 1 else "standard")
        models = M.ordered_models(parsed["model_key"].astype(str))
        L(f"models: {', '.join(models)}; rung {rung}; exclusions {'applied' if exclusions is not None else 'not available'}")
        h0: dict[str, Any] = {}
        h1_res, h3_res, h1_sec, h3_sec = [], [], {}, {}
        by_model_prompt: dict[str, dict[str, dict[str, Any]]] = {}
        for mk in models:
            prompts = M.ordered_prompts(frame.loc[frame["model_key"] == mk, "prompt_id"])
            for pid in prompts:
                s = M.summarise(frame, mk, pid, exclusions, False, n_boot=n_boot, seed=seed_b)
                s_sens = M.summarise(frame, mk, pid, exclusions, True, n_boot=n_boot, seed=seed_b, auroc_prompts=())
                summaries += [s, s_sens]
                by_model_prompt.setdefault(mk, {})[pid] = s
                summary_numbers(numbers, s)
                summary_numbers(numbers, s_sens)
                if s.get("flag_unparsable"):
                    L(f"[flag] {mk} {pid}: unparsable rate {s['unparsable_rate']['estimate']:.3f} > {C.UNPARSABLE_FLAG}")
            b = M.blind_rows(parsed, mk)
            if len(b):
                blind_frames.append(b)
                for r in b.itertuples(index=False):
                    cond = "text" if r.condition == "BLIND-text" else "grey"
                    put(numbers, f"blind.{mk}.{r.prompt_id}.{cond}.p_yes", float(r.p_yes_norm))
                    put(numbers, f"blind.{mk}.{r.prompt_id}.{cond}.greedy", str(r.greedy))
                    put(numbers, f"blind.{mk}.{r.prompt_id}.{cond}.coverage", float(r.coverage))
            else:
                L(f"[note] {mk}: no BLIND-text / BLIND-grey rows")
            p0 = by_model_prompt.get(mk, {}).get("P0")
            if p0 is not None:
                h0[mk] = {"class": p0.get("h0_class"), "ba": p0["ba"], "hit_A": p0["hit_A"], "yes_rate": p0["yes_rate"],
                          "rule": "NO-GO if BA < 0.60 or yes-rate < 0.10 or > 0.90; else GO if BA >= 0.70 and "
                                  "hit_A >= 0.70 and 0.25 <= yes-rate <= 0.75; else CONDITIONAL (P0, after exclusions)"}
                put(numbers, f"h0.{mk}.class", p0.get("h0_class"))
                h1_res.append(H.h1(frame, mk, n_boot, seed_b, exclusions=exclusions, rung=rung,
                                   contrast=((selection or {}).get("u") or {}).get("h1_contrast")))
                h1_sec[mk] = H.h1_secondary(frame, mk, n_boot, seed_b, exclusions=exclusions, summary=p0,
                                            blind=b if len(b) else None)
                put_stat(numbers, f"h1.{mk}.far_b_minus_far_u", h1_sec[mk]["far_b_minus_far_u"])
            else:
                L(f"[skip] {mk}: no P0 rows; H0 and H1 skipped")
            if {"P1", "P2hi"} <= set(by_model_prompt.get(mk, {})):
                h3_res.append(H.h3(frame, mk, rung, exclusions=exclusions, n_boot=n_boot, seed=seed_b))
                h3_sec[mk] = H.h3_secondary(by_model_prompt[mk])
            else:
                L(f"[skip] {mk}: P1 or P2hi missing; H3 skipped")
        h1_fam, h3_fam = H.holm_family(h1_res), H.holm_family(h3_res)
        for r in h1_fam:
            hypothesis_numbers(numbers, f"h1.{r['model_key']}", r)
        for r in h3_fam:
            hypothesis_numbers(numbers, f"h3.{r['model_key']}", r)
        if h1_fam and h1_fam[0].get("contrast"):
            put(numbers, "h1.contrast", h1_fam[0]["contrast"])
        hyp["H0"] = h0
        hyp["H1"] = {"primary": h1_fam, "secondary": h1_sec}
        hyp["H3"] = {"primary": h3_fam, "secondary": h3_sec}

        # ---- Arm 1 and H5
        if detector is not None:
            jf = A1.join_detector(frame, detector)
            h5_res = []
            for mk in models:
                if "P0" not in by_model_prompt.get(mk, {}):
                    continue
                res = A1.run_arm1(jf, jf, mk, n_perm=n_perm, n_boot=n_boot, seed=seed_p, boot_seed=seed_b,
                                  exclusions=exclusions)
                arm1_results[mk] = res
                arm1_numbers(numbers, res)
                for sk in res.get("skipped", []):
                    L(f"[skip] Arm 1 {mk}: {sk['row']}: {sk['reason']}")
                if res.get("skipped_reason"):
                    L(f"[skip] Arm 1 {mk}: {res['skipped_reason']}")
                h5_res.append(H.h5(res, mk))
            h5_fam = H.holm_family(h5_res)
            for r in h5_fam:
                hypothesis_numbers(numbers, f"h5.{r['model_key']}", r)
            hyp["H5"] = {"primary": h5_fam,
                         "secondary": {mk: {k: r.get(k) for k in ("rows", "matched_bias", "disagreement_abstain", "oracle",
                                                                  "agreement_greedy_beta0", "tau", "beta_dev", "dev")}
                                       for mk, r in arm1_results.items()}}
        else:
            hyp["H5"] = {"skipped": "no detector scores"}

        # ---- PLUS H2
        if all(m in by_model_prompt and "P0" in by_model_prompt[m] for m in PLUS_H2):
            r2 = H.h2(frame, PLUS_H2[0], PLUS_H2[1], n_boot, seed_b, exclusions=exclusions)
            hyp["H2"] = {"primary": H.holm_family([r2])}
            hypothesis_numbers(numbers, "h2", hyp["H2"]["primary"][0])
        else:
            hyp["H2"] = {"skipped": "PLUS models (CheXOne and Qwen2.5-VL-3B) not both present"}

        # ---- PLUS BLIND-demo rates
        if (parsed["condition"] == "BLIND-demo").any():
            demo = M.scoring_frame(parsed, items, "BLIND-demo")
            for mk in M.ordered_models(demo["model_key"]):
                for pid in M.ordered_prompts(demo.loc[demo["model_key"] == mk, "prompt_id"]):
                    s = M.summarise(demo, mk, pid, exclusions, False, condition="BLIND-demo", n_boot=0, auroc_prompts=())
                    demo_summaries.append(s)
                    summary_numbers(numbers, s)

        # ---- smoke and run metadata
        for mk in models:
            sm = frame.loc[(frame["model_key"] == mk) & (frame["arm"] == "DEV") & (frame["prompt_id"] == "P0")]
            if "smoke" in sm.columns:
                sm = sm.loc[sm["smoke"].astype(int) == 1]
                cov = sm["coverage"].to_numpy(dtype=float)
                cov = cov[np.isfinite(cov)]
                if cov.size:
                    put(numbers, f"smoke.{mk}.coverage_median", float(np.median(cov)))
                    put(numbers, f"smoke.{mk}.n", int(cov.size))
                sig = sorted(set(sm["signal"].astype(str)) - {""})
                if sig:
                    put(numbers, f"smoke.{mk}.signal", sig[0] if len(sig) == 1 else "; ".join(sig))
        run_numbers(numbers, parsed, detector)
    else:
        L("[skip] model metrics, hypotheses, Tables 2-3 and Figure 1: items or parsed results missing")

    # ---- GPU name from the pod's run settings files
    if inputs.run_dir is None:
        L("[note] run.gpu_name not written: no --run-dir given")
    else:
        gpus = gpu_names(inputs.run_dir)
        if gpus:
            put(numbers, "run.gpu_name", gpus[0] if len(gpus) == 1 else "; ".join(gpus))
            L(f"[input] run.gpu_name = {numbers['run.gpu_name']!r} (from the run settings in {Path(inputs.run_dir).name})")
        else:
            L(f"[note] run.gpu_name not written: no GPU name recorded under {Path(inputs.run_dir).name} (CPU run or settings missing)")

    # ---- extra numbers (never overwrite a computed key)
    for key, value in sorted((inputs.extra or {}).items()):
        if not NUMBER_KEY_RE.fullmatch(str(key)):
            raise ValueError(f"extra number key {key!r} is not a valid placeholder key ({NUMBER_KEY_RE.pattern})")
        if key in numbers:
            L(f"[note] extra number {key} ignored: the analysis computes it ({numbers[key]!r})")
            continue
        put(numbers, str(key), value)
        L(f"[input] extra number {key} = {value!r}")

    # ---- tables
    blind_all = pd.concat(blind_frames, ignore_index=True) if blind_frames else None
    kappa = (verification or {}).get("retest")
    if match_stats and inputs.spotcheck_correct is not None:
        match_stats = {**match_stats, "spotcheck_correct": int(inputs.spotcheck_correct)}
    t1 = T.table1(match_stats, yield_df, selection, verification, kappa, detector_info)
    files += list(T.write_table(t1, "table1", out).values())
    if summaries:
        files += list(T.write_table(T.table2(summaries, blind_all), "table2", out).values())
        yr = F.yes_rate_frame(summaries)
        files += list(F.figure1(yr, blind_all, out / "figure1", example_pair).values())
        if plus_figure:
            curves, verbal = {}, []
            for mk in M.ordered_models({s["model_key"] for s in summaries}):
                curves[mk] = M.risk_coverage(frame, mk, "P0", exclusions)
                for s in summaries:
                    if s["model_key"] == mk and not s["unparsable_as_wrong"] and s["prompt_id"] in ("P1", "P2hi", "P2lo"):
                        verbal.append({"model_key": mk, "prompt_id": s["prompt_id"], "coverage": s["answered"]["estimate"],
                                       "risk": s["risk"]["estimate"]})
            files += list(F.figure2(curves, pd.DataFrame(verbal, columns=["model_key", "prompt_id", "coverage", "risk"]),
                                    out / "figure2").values())
    if arm1_results:
        files += list(T.write_table(T.table3(arm1_results), "table3", out).values())
    elif summaries:
        L("[skip] Table 3: Arm 1 not run (no detector scores)")

    # ---- remove outputs of an earlier run that this run did not produce
    written = {p.name for p in files} | {"hypotheses.json", "numbers.json", "metrics.json", "analysis_log.txt"}
    for name in MANAGED_FILES:
        stale = out / name
        if name not in written and stale.is_file():
            stale.unlink()
            L(f"[stale] removed {name}: left by an earlier run and not produced by this one")

    # ---- JSON outputs
    hyp["meta"]["log_skips"] = [line for line in L.lines if line.startswith(("[skip]", "[stale]"))]
    _dump(hyp, out / "hypotheses.json")
    _dump(dict(sorted(numbers.items())), out / "numbers.json")
    _dump({"summaries": summaries, "blind": blind_all if blind_all is not None else [], "demo": demo_summaries,
           "arm1": arm1_results}, out / "metrics.json")
    files += [out / "hypotheses.json", out / "numbers.json", out / "metrics.json"]
    L(f"wrote {len(files) + 1} files")
    (out / "analysis_log.txt").write_text("\n".join(L.lines) + "\n", encoding="utf-8", newline="\n")
    files.append(out / "analysis_log.txt")
    return {"numbers": numbers, "hypotheses": hyp, "summaries": summaries, "arm1": arm1_results, "files": files,
            "log": L.lines, "detector": detector_info}


def parse_number(text: str) -> tuple[str, Any]:
    """``KEY=VALUE`` from the command line: VALUE becomes an int, else a float, else stays text."""
    key, sep, raw = str(text).partition("=")
    key, raw = key.strip(), raw.strip()
    if not sep or not key or not NUMBER_KEY_RE.fullmatch(key):
        raise argparse.ArgumentTypeError(f"expected KEY=VALUE with KEY matching {NUMBER_KEY_RE.pattern}, got {text!r}")
    for conv in (int, float):
        try:
            return key, conv(raw)
        except ValueError:
            pass
    return key, raw


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", default="items/items.csv")
    ap.add_argument("--parsed", default="runs/parsed_results.csv")
    ap.add_argument("--detector", default="runs/detector_scores.csv")
    ap.add_argument("--exclusions", default="items/exclusions.csv")
    ap.add_argument("--match-stats", default="data/kaggle/res201-match/match_stats.json")
    ap.add_argument("--yield", dest="yield_table", default="data/kaggle/res201-hash/yield_table.csv")
    ap.add_argument("--selection", default="items/selection_report.json")
    ap.add_argument("--verification", default="results/verification_summary.json")
    ap.add_argument("--run-dir", default=None, help="pod run directory runs/<run_id> (for run.gpu_name)")
    ap.add_argument("--example-pair", default=None, help="JSON with a_path, b_path, a_arrow, b_arrow (Figure 1 panel a)")
    ap.add_argument("--spotcheck-correct", type=int, default=None, help="pairs judged correct in the 60-pair spot check")
    ap.add_argument("--number", type=parse_number, action="append", default=[], metavar="KEY=VALUE",
                    help="extra numbers.json entry, e.g. parse.handcheck_agree=98 (repeatable)")
    ap.add_argument("--out", default="results/")
    ap.add_argument("--n-boot", type=int, default=C.N_BOOT)
    ap.add_argument("--n-perm", type=int, default=C.N_PERM)
    ap.add_argument("--fast", action="store_true", help=f"development run: {FAST_N} resamples and permutations")
    ap.add_argument("--no-plus-figure", action="store_true", help="skip the PLUS risk-coverage figure")
    a = ap.parse_args(argv)
    n_boot, n_perm = (FAST_N, FAST_N) if a.fast else (a.n_boot, a.n_perm)
    inputs = AnalysisInputs(
        items=Path(a.items), parsed=Path(a.parsed), detector=Path(a.detector), exclusions=Path(a.exclusions),
        match_stats=Path(a.match_stats), yield_table=Path(a.yield_table), selection=Path(a.selection),
        verification=Path(a.verification), run_dir=Path(a.run_dir) if a.run_dir else None,
        example_pair=Path(a.example_pair) if a.example_pair else None,
        spotcheck_correct=a.spotcheck_correct, extra=dict(a.number),
    )
    run_analysis(inputs, a.out, n_boot, n_perm, plus_figure=not a.no_plus_figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
