"""Tables 1-3 of the Stage 1 report and their CSV, Markdown and LaTeX files.

* Table 1: benchmark construction and verification (image matching, the RANZCR CLiP label
  yield, item selection, the blind reading, the automatic verifier).
* Table 2: metrics per model x prompt (primary rows and the "unparsable counted as wrong"
  sensitivity rows) with the BLIND-text and BLIND-grey rows (single generations, no interval).
* Table 3: Arm 1 rows per model (greedy text, beta = 0, fused, dev-selected beta, detector alone,
  shuffled expert, matched constant bias, disagreement-abstain, oracle rows).

Cells format a rate as ``0.62 [0.53, 0.70] (n=118)``; "n/a" marks an undefined value. Tables hold
aggregates only (no item id, filename or label of a single image). Wording follows the project
rules: RANZCR CLiP labels (43 trained labellers), within-patient contrast.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from cxrdev import metrics as M
from cxrdev.fileio import write_csv

__all__ = [
    "MODEL_LABELS", "PROMPT_LABELS", "STRATUM_LABELS", "TABLE1_COLUMNS", "TABLE2_BASE_COLUMNS", "TABLE3_COLUMNS",
    "fmt_ci", "fmt_num", "fmt_p", "fmt_rate", "model_label", "table1", "table2", "table3", "to_latex", "to_markdown",
    "write_table",
]

MODEL_LABELS: dict[str, str] = {
    "qwen3vl2b": "Qwen3-VL-2B", "medgemma15": "MedGemma 1.5 4B", "chexone": "CheXOne",
    "qwen25vl3b": "Qwen2.5-VL-3B", "nvreason3b": "NV-Reason-CXR-3B",
}
PROMPT_LABELS: dict[str, str] = {"P0": "P0", "P1": "P1", "P2hi": "P2-hi", "P2lo": "P2-lo"}
STRATUM_LABELS: dict[str, str] = {"line": "line", "noline": "no line", "ge2": ">=2 lines", "eq1": "1 line"}

TABLE1_COLUMNS: tuple[str, ...] = ("section", "quantity", "value")
TABLE2_BASE_COLUMNS: tuple[str, ...] = (
    "model", "prompt", "row", "hit_A", "FAR_B", "FAR_U", "BA", "yes_rate", "abstention", "ER_0.9", "AUROC_AB",
    "AUROC_AU", "persistence", "pair_correct", "unparsable", "truncated", "H0", "flag", "blind_p_yes", "blind_greedy",
)
TABLE3_COLUMNS: tuple[str, ...] = (
    "model", "row", "rule", "BA", "delta_BA", "hit_A", "FAR_B", "FAR_U", "yes_rate", "coverage", "risk", "p", "note",
)
NA = "n/a"


# ----------------------------------------------------------------------------- formatting
def _finite(x: Any) -> bool:
    try:
        return x is not None and not isinstance(x, bool) and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def fmt_num(x: Any, digits: int = 2) -> str:
    """A number with ``digits`` decimals ("n/a" when missing or not finite)."""
    return f"{float(x):.{digits}f}" if _finite(x) else NA


def fmt_ci(d: Mapping[str, Any] | None, digits: int = 2, with_n: bool = True) -> str:
    """``0.62 [0.53, 0.70] (n=118)`` from a dict with estimate, lo, hi and n (interval omitted when undefined)."""
    if not d or not _finite(d.get("estimate")):
        return NA if not d or "n" not in d or not with_n else f"{NA} (n={int(d['n'])})"
    text = fmt_num(d["estimate"], digits)
    if _finite(d.get("lo")) and _finite(d.get("hi")):
        text += f" [{fmt_num(d['lo'], digits)}, {fmt_num(d['hi'], digits)}]"
    if with_n and d.get("n") is not None:
        text += f" (n={int(d['n'])})"
    return text


def fmt_rate(d: Mapping[str, Any] | None, digits: int = 2) -> str:
    """A Wilson rate dict as ``0.62 [0.53, 0.70] (n=118)``."""
    return fmt_ci(d, digits, with_n=True)


def fmt_p(p: Any) -> str:
    """A p-value as "<0.001" or with three decimals ("n/a" when missing)."""
    if not _finite(p):
        return NA
    v = float(p)
    return "<0.001" if v < 0.001 else f"{v:.3f}"


def model_label(key: str) -> str:
    """Display name of a model key (the key itself when unknown)."""
    return MODEL_LABELS.get(key, key)


# ----------------------------------------------------------------------------- Table 1
def _get(d: Mapping[str, Any] | None, *path: str, default: Any = None) -> Any:
    cur: Any = d
    for p in path:
        if not isinstance(cur, Mapping) or p not in cur:
            return default
        cur = cur[p]
    return cur


def _stratum_order(strata: Iterable[str]) -> list[str]:
    """U strata in ``metrics.U_STRATUM_ORDER`` (exposed stratum first), unknown ones last in sorted order."""
    uniq = sorted(set(str(s) for s in strata))
    return [s for s in M.U_STRATUM_ORDER if s in uniq] + [s for s in uniq if s not in M.U_STRATUM_ORDER]


def _val(x: Any) -> str:
    if x is None:
        return NA
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, int):
        return f"{x:,}"
    if isinstance(x, float):
        if not math.isfinite(x):
            return NA
        return f"{x:,.0f}" if x.is_integer() and abs(x) >= 100 else f"{x:.3g}" if abs(x) < 1 else f"{x:.2f}"
    return str(x)


def table1(
    match_stats: Mapping[str, Any] | None,
    yield_df: pd.DataFrame | None,
    selection_report: Mapping[str, Any] | None,
    verification_summary: Mapping[str, Any] | None,
    kappa: float | Mapping[str, Any] | None,
    detector_agreement: Mapping[str, Any] | None,
) -> pd.DataFrame:
    """Table 1 (columns ``section``, ``quantity``, ``value``); every input may be None (rows say "not available").

    ``match_stats`` is ``match_stats.json`` (``matching.calibration_report`` plus a ``run`` block and,
    optionally, ``spotcheck_correct``); ``yield_df`` the yield table (metric, value);
    ``selection_report`` ``selection_report.json``; ``verification_summary`` the output of
    ``scripts/verify_join.py``; ``kappa`` the retest Cohen's kappa (a float or a dict with
    ``kappa`` and ``n``; defaults to the summary's); ``detector_agreement`` ``arm1.detector_summary``.
    """
    rows: list[tuple[str, str, str]] = []

    def add(section: str, quantity: str, value: Any) -> None:
        rows.append((section, quantity, value if isinstance(value, str) else _val(value)))

    sec = "Image matching (RANZCR CLiP JPEG to NIH PNG)"
    if match_stats:
        sc = match_stats.get("status_counts", {}) or {}
        add(sec, "RANZCR images", match_stats.get("n_ranzcr"))
        add(sec, "Matched images", sc.get("matched"))
        add(sec, "Match rate", fmt_num(match_stats.get("match_rate"), 3))
        add(sec, "Ambiguous (excluded)", sc.get("ambiguous", match_stats.get("ambiguous_count")))
        add(sec, "Ambiguous: share within one NIH patient", fmt_num(match_stats.get("ambiguous_share_within_patient"), 2))
        add(sec, "Unmatched", sc.get("unmatched"))
        add(sec, "Duplicate conflicts", sc.get("duplicate_conflict"))
        add(sec, "Patient conflicts", sc.get("patient_conflict"))
        add(sec, "Hamming cut (bits)", match_stats.get("cut"))
        rc = match_stats.get("route_counts", {}) or {}
        if rc:
            add(sec, "Accepted by margin / by NCC tie-break", f"{_val(rc.get('margin'))} / {_val(rc.get('ncc_tiebreak'))}")
        if match_stats.get("spotcheck_correct") is not None:
            n_spot = _get(match_stats, "run", "n_spotcheck_pairs", default=60)
            add(sec, "Visual spot check correct", f"{_val(match_stats['spotcheck_correct'])} of {_val(n_spot)}")
    else:
        add(sec, "Matching statistics", "not available")

    sec = "RANZCR CLiP label yield (43 trained labellers; before matching)"
    if yield_df is not None and len(yield_df):
        for r in yield_df.itertuples(index=False):
            add(sec, str(r.metric).replace("_", " "), int(r.value) if _finite(r.value) else r.value)
    else:
        add(sec, "Yield table", "not available")

    sec = "Item selection"
    if selection_report:
        rep = selection_report
        add(sec, "Rung", rep.get("rung"))
        add(sec, "Eligible patients (both ETT states)", rep.get("n_eligible_patients"))
        by_arm = _get(rep, "counts", "by_arm", default={}) or {}
        add(sec, "Pairs (A/B, within-patient contrast)", by_arm.get("A"))
        dc = rep.get("direction_counts", {}) or {}
        if dc:
            add(sec, "Direction: B after A / B before A", f"{_val(dc.get('B_after_A'))} / {_val(dc.get('B_before_A'))}")
        gap = rep.get("followup_gap", {}) or {}
        if gap.get("n"):
            add(sec, "Follow-up gap, median [IQR]", f"{fmt_num(gap.get('median'), 1)} [{fmt_num(gap.get('q1'), 1)}, {fmt_num(gap.get('q3'), 1)}]")
        u_counts = _get(rep, "counts", "by_stratum", "U", default={}) or {}
        substitute = bool(rep.get("h1_substitute"))
        for s in _stratum_order(u_counts):
            note = " (descriptive)" if substitute and s == "noline" else ""
            add(sec, f"U stratum {STRATUM_LABELS.get(s, s)}{note}", u_counts[s])
        add(sec, "H1 substitute contrast (>=2 vs 1 line)", substitute)
        excluded = rep.get("excluded_filenames") or {}
        if excluded:
            add(sec, "Images excluded before selection (seen in software tests)",
                f"{_val(excluded.get('n_present_in_matched_pool'))} of {_val(excluded.get('n_listed'))} listed")
        for key, label in (("A", "A"), ("B", "B"), ("U", "U")):
            share = _get(rep, "view_shares", "by_arm", key, "ap_share")
            if share is not None:
                add(sec, f"AP share, arm {label}", fmt_num(share, 2))
        vs_strata = _get(rep, "view_shares", "by_stratum", default={}) or {}
        for s in _stratum_order(k[2:] for k in vs_strata if k.startswith("U:")):
            add(sec, f"AP share, U {STRATUM_LABELS.get(s, s)}", fmt_num(vs_strata[f"U:{s}"].get("ap_share"), 2))
        dev = _get(rep, "counts", "by_stratum", "DEV", default={}) or {}
        if dev:
            add(sec, "Dev pool ETT present / absent", f"{_val(dev.get('pos'))} / {_val(dev.get('neg'))}")
        if _get(rep, "smoke", "n") is not None:
            add(sec, "Smoke-test items (dev pool)", _get(rep, "smoke", "n"))
    else:
        add(sec, "Selection report", "not available")

    sec = "Blind reading (student; blind to arm and to model output)"
    vs = verification_summary
    if vs:
        add(sec, "Images read", vs.get("n_read"))
        resp = vs.get("responses", {}) or {}
        add(sec, "Present / absent / skip", f"{_val(resp.get('present'))} / {_val(resp.get('absent'))} / {_val(resp.get('skip'))}")
        for flag, label in (("flag_ambiguous", "ambiguous"), ("flag_tracheostomy", "tracheostomy"), ("flag_out_of_field", "out of field")):
            add(sec, f"Flag: {label}", _get(vs, "flags", flag))
        add(sec, "Agreement with RANZCR CLiP labels", fmt_rate(vs.get("agreement")))
        for a in ("A", "B", "U"):
            agr = _get(vs, "by_arm", a, "agreement")
            if agr:
                add(sec, f"Agreement, arm {a}", fmt_rate(agr))
        add(sec, "Disagreements with the label", vs.get("n_disagreements"))
        ex = vs.get("exclusions", {}) or {}
        add(sec, "Items excluded", ex.get("n_items"))
        for reason, n in (ex.get("by_reason", {}) or {}).items():
            add(sec, f"Excluded: {reason.replace('_', ' ')}", n)
    else:
        add(sec, "Verification", "not available (reading not done yet)")
    k = kappa if kappa is not None else _get(vs, "retest")
    if isinstance(k, Mapping):
        if k.get("kappa") is not None:
            add(sec, "Retest Cohen's kappa (>= 48 h later)", f"{fmt_num(k.get('kappa'), 2)} (n={_val(k.get('n_compared', k.get('n')))})")
    elif _finite(k):
        add(sec, "Retest Cohen's kappa (>= 48 h later)", fmt_num(k, 2))

    sec = "Automatic verifier (MedSigLIP zero-shot; never used to exclude)"
    det = detector_agreement
    if det and det.get("dev_auroc"):
        add(sec, "Dev-pool AUROC", fmt_ci(det["dev_auroc"]))
        add(sec, "Youden threshold tau (dev pool)", fmt_num(det.get("tau"), 3))
        for a, r in (det.get("agreement") or {}).items():
            add(sec, f"Agreement with RANZCR CLiP label, arm {a}", fmt_rate(r))
    else:
        add(sec, "Detector scores", "not available")
    return pd.DataFrame(rows, columns=list(TABLE1_COLUMNS))


# ----------------------------------------------------------------------------- Table 2
def _stratum_cols(summaries: Sequence[Mapping[str, Any]]) -> list[str]:
    keys = {k for s in summaries for k in s if k.startswith("FAR_U_")}
    order = [f"FAR_U_{s}" for s in M.U_STRATUM_ORDER if f"FAR_U_{s}" in keys]
    return order + sorted(keys - set(order))


def table2(summaries: Sequence[Mapping[str, Any]], blind: pd.DataFrame | None) -> pd.DataFrame:
    """Table 2: one row per model x prompt x scoring mode, then the model's blind rows.

    ``summaries`` are ``metrics.summarise`` results (primary and ``unparsable_as_wrong`` rows);
    ``blind`` is the concatenation of ``metrics.blind_rows`` (or None). Columns are
    ``TABLE2_BASE_COLUMNS`` with one ``FAR_U_<stratum>`` column per U stratum inserted after FAR_U.
    """
    strata = _stratum_cols(summaries)
    cols = list(TABLE2_BASE_COLUMNS)
    i = cols.index("FAR_U") + 1
    cols[i:i] = strata
    models = M.ordered_models([s["model_key"] for s in summaries] + ([] if blind is None else list(blind["model_key"])))
    out: list[dict[str, str]] = []
    for mk in models:
        ms = [s for s in summaries if s["model_key"] == mk]
        for pid in M.ordered_prompts([s["prompt_id"] for s in ms]):
            for sens in (False, True):
                for s in [x for x in ms if x["prompt_id"] == pid and bool(x.get("unparsable_as_wrong")) == sens]:
                    row = {c: "" for c in cols}
                    row.update(model=model_label(mk), prompt=PROMPT_LABELS.get(pid, pid),
                               row="unparsable as wrong" if sens else "primary")
                    for c in ("hit_A", "FAR_B", "FAR_U", "yes_rate", "abstention", "persistence", "pair_correct", *strata):
                        row[c] = fmt_rate(s.get(c)) if s.get(c) else ""
                    row["BA"] = fmt_ci(s.get("ba"))
                    row["ER_0.9"] = fmt_ci(s.get("ER_0.9"))
                    row["AUROC_AB"] = fmt_ci(s.get("auroc_AB")) if s.get("auroc_AB") else ""
                    row["AUROC_AU"] = fmt_ci(s.get("auroc_AU")) if s.get("auroc_AU") else ""
                    if not sens:
                        row["unparsable"] = fmt_rate(s.get("unparsable_rate"))
                        row["truncated"] = fmt_rate(s.get("truncated_rate"))
                        row["H0"] = s.get("h0_class") or ""
                        row["flag"] = ">10% unparsable" if s.get("flag_unparsable") else ""
                    out.append(row)
        if blind is not None and len(blind):
            b = blind.loc[blind["model_key"].astype(str) == mk]
            for r in b.itertuples(index=False):
                row = {c: "" for c in cols}
                row.update(model=model_label(mk), prompt=PROMPT_LABELS.get(str(r.prompt_id), str(r.prompt_id)),
                           row=str(r.condition), blind_p_yes=fmt_num(r.p_yes_norm, 2), blind_greedy=str(r.greedy))
                out.append(row)
    return pd.DataFrame(out, columns=cols)


# ----------------------------------------------------------------------------- Table 3
def table3(arm1_results: Sequence[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    """Table 3: the Arm 1 rows of every model (``arm1.run_arm1`` results), in model order.

    ``delta_BA`` is the balanced-accuracy difference from the beta = 0 rule on the same items
    (for the matched-bias row the fused-minus-matched difference is added to ``note``); ``p`` is
    the one-sided shuffled-expert permutation p (fused row and the null row). A last row per
    model gives the greedy-text vs beta = 0 agreement share.
    """
    results = list(arm1_results.values()) if isinstance(arm1_results, Mapping) else list(arm1_results)
    by_model = {r["model_key"]: r for r in results}
    out: list[dict[str, str]] = []
    for mk in M.ordered_models(by_model):
        res = by_model[mk]
        if res.get("skipped_reason"):
            out.append({c: "" for c in TABLE3_COLUMNS} | {"model": model_label(mk), "row": "Arm 1 not run",
                                                          "note": str(res["skipped_reason"])})
            continue
        for r in res.get("rows", []):
            row = {c: "" for c in TABLE3_COLUMNS}
            row["model"] = model_label(mk)
            row["row"] = str(r["label"])
            if r.get("param_name") and _finite(r.get("param")):
                row["rule"] = f"{r['param_name']} = {fmt_num(r['param'], 3)}"
            row["BA"] = fmt_ci(r.get("ba")) if r.get("ba") else ""
            row["delta_BA"] = fmt_ci(r.get("delta_ba"), with_n=False) if r.get("delta_ba") else ""
            for c in ("hit_A", "FAR_B", "FAR_U", "yes_rate", "coverage", "risk"):
                row[c] = fmt_rate(r.get(c)) if r.get(c) else ""
            row["p"] = fmt_p(r.get("p_value")) if r.get("p_value") is not None else ""
            note = str(r.get("note", ""))
            if r.get("fused_minus_matched"):
                note += f"; fused minus matched: {fmt_ci(r['fused_minus_matched'], with_n=False)}"
            row["note"] = note
            out.append(row)
        ag = res.get("agreement_greedy_beta0")
        if ag:
            out.append({c: "" for c in TABLE3_COLUMNS} | {"model": model_label(mk), "row": "Greedy text vs beta = 0 agreement",
                                                          "yes_rate": "", "note": fmt_rate(ag)})
    return pd.DataFrame(out, columns=list(TABLE3_COLUMNS))


# ----------------------------------------------------------------------------- writers
def to_markdown(df: pd.DataFrame) -> str:
    """A GitHub pipe table (pipes in cells escaped, newlines replaced by spaces)."""

    def cell(x: Any) -> str:
        return str(x).replace("|", "\\|").replace("\n", " ")

    head = "| " + " | ".join(cell(c) for c in df.columns) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    body = ["| " + " | ".join(cell(v) for v in r) + " |" for r in df.astype(str).itertuples(index=False)]
    return "\n".join([head, sep, *body]) + "\n"


_LATEX_MAP = {
    "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
    "~": r"\textasciitilde{}", "^": r"\textasciicircum{}", "<": r"\textless{}", ">": r"\textgreater{}",
}


def _tex(x: Any) -> str:
    return "".join(_LATEX_MAP.get(ch, ch) for ch in str(x))


def to_latex(df: pd.DataFrame, caption: str | None = None, label: str | None = None) -> str:
    """A booktabs ``tabular`` (left-aligned columns, every special character escaped)."""
    lines = []
    if caption or label:
        lines.append(r"\begin{table}[ht]")
        lines.append(r"\centering")
        if caption:
            lines.append(r"\caption{" + _tex(caption) + "}")
        if label:
            lines.append(r"\label{" + label + "}")
    lines.append(r"\begin{tabular}{" + "l" * len(df.columns) + "}")
    lines.append(r"\toprule")
    lines.append(" & ".join(_tex(c) for c in df.columns) + r" \\")
    lines.append(r"\midrule")
    for r in df.astype(str).itertuples(index=False):
        lines.append(" & ".join(_tex(v) for v in r) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    if caption or label:
        lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def write_table(df: pd.DataFrame, stem: str, outdir: str | Path) -> dict[str, Path]:
    """Write ``<stem>.csv`` (``fileio.write_csv``), ``<stem>.md`` and ``<stem>.tex`` into ``outdir``.

    Returns the three paths. Files are UTF-8 with ``\\n`` line endings, so the same table always
    gives the same bytes.
    """
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {"csv": out / f"{stem}.csv", "md": out / f"{stem}.md", "tex": out / f"{stem}.tex"}
    write_csv(df, paths["csv"])
    paths["md"].write_text(to_markdown(df), encoding="utf-8", newline="\n")
    paths["tex"].write_text(to_latex(df), encoding="utf-8", newline="\n")
    return paths
