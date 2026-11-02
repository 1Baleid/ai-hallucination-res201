"""Per model x prompt x arm metrics, the H0 floor gate and the blind rows.

PROTOCOL.md sections 3 (unparsable rule), 5 (metrics) and 6 (H0 gate); docs/PIPELINE_SPEC.md
section 3, ``metrics.py``.

Inputs. ``read_parsed`` reads S8 (``parsed_results.csv``). ``scoring_frame`` keeps the rows of
one condition (default "image") and joins the S5 item columns (arm, pair id, stratum, NIH patient
id, labels) on ``item_id``. The join happens only here, at scoring time: the pod never sees an
arm or a label. The labels of record are the RANZCR CLiP labels (43 trained labellers:
radiologists, radiology trainees, hospital medical officers); ``y`` = 1 when the ETT is present.

Denominators (pre-registered rule). An output is *valid* when its parsed answer is "yes", "no"
or "abstain" and it is not truncated. Unparsable and truncated outputs are excluded from the
denominators of hit_A, FAR_B, FAR_U, balanced accuracy, ER_t and the abstain indicator and are
reported as their own rates, overall and per arm. Abstentions stay in the denominators and are
never correct. ``unparsable_as_wrong=True`` gives the sensitivity row: every output excluded by
that rule is put back and scored as the wrong answer for its arm ("no" on A, "yes" on B and U).
AUROC uses z, which does not depend on the parse, so it uses every retained item with a finite z.

Exclusions (S12, from the blind reading) remove items before anything is computed; a pair-level
metric (persistence, pair-correct) uses a pair only when both members are retained and valid.

Intervals: Wilson 95% for every rate (``stats.wilson_ci``); patient-clustered BCa bootstrap
(``stats.clustered_ci``, cluster = NIH patient id, so A and B of a pair move together) for
balanced accuracy, ER_t and AUROC. When a statistic mixes pair patients and U patients, the
resampling is stratified by that design set so each replicate keeps the design's numbers of pair
and U patients.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cxrdev import constants as C
from cxrdev import stats
from cxrdev.fileio import require_columns
from cxrdev.logits import p_yes_normalised

__all__ = [
    "BLIND_CONDITIONS", "CONDITIONS", "EXCLUSION_REASONS", "PROMPT_ORDER", "S7_FIELDS", "S8_BOOL_COLS",
    "S8_COLUMNS", "S8_FLOAT_COLS", "S8_INT_COLS", "S8_STR_COLS", "SCORING_ITEM_COLS", "TEST_ARMS",
    "U_STRATUM_ORDER", "VALID_ANSWERS", "apply_exclusions", "blind_rows", "clustered_estimate", "coerce_parsed",
    "auroc_stat", "ba_stat", "effective_answers", "er_cost", "h0_gate", "h1_contrast", "mean_stat", "ordered_models", "ordered_prompts",
    "rate", "read_exclusions", "read_parsed", "risk_coverage", "rows_for", "scoring_frame", "summarise",
]

# ----------------------------------------------------------------------------- schemas
#: S7 run-record fields, in the order of docs/PIPELINE_SPEC.md section 2.
S7_FIELDS: tuple[str, ...] = (
    "run_id", "model_key", "model_id", "revision", "transformers_version", "torch_version", "dtype",
    "device", "item_id", "nih_filename", "image_sha256", "condition", "prompt_id", "prompt_sha256",
    "prompt_text", "visual_tokens", "seed", "max_new_tokens", "raw_output", "n_generated_tokens",
    "ended_with_eos", "parsed", "parser_version", "logp_yes", "logp_no", "logp_abstain", "z",
    "coverage", "signal", "yes_ids", "no_ids", "wall_s", "timestamp_utc",
)
#: S8 = every S7 field except the id lists, plus ``truncated``.
S8_COLUMNS: tuple[str, ...] = tuple(f for f in S7_FIELDS if f not in ("yes_ids", "no_ids")) + ("truncated",)
S8_FLOAT_COLS: tuple[str, ...] = ("logp_yes", "logp_no", "logp_abstain", "z", "coverage", "wall_s")
S8_INT_COLS: tuple[str, ...] = ("visual_tokens", "seed", "max_new_tokens", "n_generated_tokens")
S8_BOOL_COLS: tuple[str, ...] = ("ended_with_eos", "truncated")
S8_STR_COLS: tuple[str, ...] = tuple(
    c for c in S8_COLUMNS if c not in S8_FLOAT_COLS + S8_INT_COLS + S8_BOOL_COLS
)

VALID_ANSWERS: tuple[str, ...] = ("yes", "no", "abstain")
TEST_ARMS: tuple[str, ...] = ("A", "B", "U")
CONDITIONS: tuple[str, ...] = ("image", "BLIND-text", "BLIND-grey", "BLIND-demo")
BLIND_CONDITIONS: tuple[str, ...] = ("BLIND-text", "BLIND-grey")
PROMPT_ORDER: tuple[str, ...] = ("P0", "P1", "P2hi", "P2lo")
#: U strata in display order: the H1 contrast first (line vs noline, or the substitute ge2 vs eq1), then the
#: descriptive no-line stratum that remains under the substitute.
U_STRATUM_ORDER: tuple[str, ...] = ("line", "ge2", "eq1", "noline")
EXCLUSION_REASONS: tuple[str, ...] = (
    "flag_ambiguous", "flag_tracheostomy", "flag_out_of_field", "label_disagreement", "skip",
)
#: S5 columns the scoring frame needs (the RANZCR uid is deliberately not carried).
SCORING_ITEM_COLS: tuple[str, ...] = (
    "item_id", "nih_filename", "nih_patient_id", "arm", "pair_id", "stratum", "ett", "n_other_lines",
)
_OPTIONAL_ITEM_COLS: tuple[str, ...] = (
    "ngt", "ngt_incomplete", "cvc", "swan", "view", "followup", "age", "sex", "direction", "gap", "rung", "smoke",
)
#: The wrong answer for an arm (used by the "unparsable counted as wrong" sensitivity row).
WRONG_ANSWER: dict[str, str] = {"A": "no", "B": "yes", "U": "yes"}
CLUSTER_COL = "nih_patient_id"
_NAN = float("nan")


# ----------------------------------------------------------------------------- small helpers
def _f(x: Any) -> float:
    """A plain Python float (NaN for None)."""
    return _NAN if x is None else float(x)


def rate(k: int, n: int, alpha: float = C.ALPHA) -> dict[str, Any]:
    """A rate with its Wilson interval: ``estimate`` (NaN when n == 0), ``k``, ``n``, ``lo``, ``hi``."""
    k, n = int(k), int(n)
    lo, hi = stats.wilson_ci(k, n, alpha)
    return {"estimate": (k / n) if n else _NAN, "k": k, "n": n, "lo": float(lo), "hi": float(hi), "method": "wilson"}


def er_cost(t: float) -> float:
    """The ER_t penalty c = t / (1 - t), rounded to 10 decimals (9 at t = 0.9, 1 at t = 0.5)."""
    t = float(t)
    if not 0.0 < t < 1.0:
        raise ValueError(f"t must be in (0, 1), got {t!r}")
    return round(t / (1.0 - t), 10)


def ordered_models(keys: Iterable[str]) -> list[str]:
    """Model keys in report order: CORE models first, then the other ``constants.MODELS`` keys, then the rest."""
    known = list(C.CORE_MODELS) + [k for k in C.MODELS if k not in C.CORE_MODELS]
    uniq = sorted(set(str(k) for k in keys))
    return [k for k in known if k in uniq] + [k for k in uniq if k not in known]


def ordered_prompts(ids: Iterable[str]) -> list[str]:
    """Prompt ids in PROTOCOL order (P0, P1, P2hi, P2lo), unknown ids last in sorted order."""
    uniq = sorted(set(str(p) for p in ids))
    return [p for p in PROMPT_ORDER if p in uniq] + [p for p in uniq if p not in PROMPT_ORDER]


def _to_bool(series: pd.Series, name: str) -> pd.Series:
    """Booleans from bool, 0/1 or "True"/"False" text (any case); anything else raises ValueError."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    mapping = {"true": True, "false": False, "1": True, "0": False, "1.0": True, "0.0": False}
    text = series.astype(str).str.strip().str.lower()
    bad = ~text.isin(list(mapping))
    if bool(bad.any()):
        raise ValueError(f"column {name!r} has non-boolean values, e.g. {series[bad].head(3).tolist()}")
    return text.map(mapping).astype(bool)


# ----------------------------------------------------------------------------- reading inputs
def coerce_parsed(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of an S8 frame with its column types (text, int64, float64, bool).

    Text columns keep empty strings (never NaN); float columns may hold NaN; boolean columns
    accept bool, 0/1 or "True"/"False". Raises ValueError for missing columns or bad values.
    """
    require_columns(df, S8_COLUMNS, "parsed results (S8)")
    out = df.copy()
    for c in S8_STR_COLS:
        out[c] = out[c].fillna("").astype(str)
    for c in S8_INT_COLS:
        vals = pd.to_numeric(out[c], errors="raise")
        if bool(vals.isna().any()):
            raise ValueError(f"column {c!r} has missing values")
        out[c] = vals.astype("int64")
    for c in S8_FLOAT_COLS:
        out[c] = pd.to_numeric(out[c], errors="raise").astype("float64")
    for c in S8_BOOL_COLS:
        out[c] = _to_bool(out[c], c)
    return out


def read_parsed(path: str | Path) -> pd.DataFrame:
    """Read ``parsed_results.csv`` (S8) with its types.

    Text fields are read as written (an output "None" or "NA" stays text, an empty field stays "");
    an empty float field is NaN. Extra columns are kept after the S8 columns.
    """
    p = Path(path)
    head = pd.read_csv(p, nrows=0)
    require_columns(head, S8_COLUMNS, str(p))
    numeric = set(S8_FLOAT_COLS) | set(S8_INT_COLS)
    dtype = {c: str for c in head.columns if c not in numeric}
    na_values = {c: ["", "NaN", "nan"] for c in S8_FLOAT_COLS}
    df = pd.read_csv(p, dtype=dtype, keep_default_na=False, na_values=na_values)
    return coerce_parsed(df)


def read_exclusions(path: str | Path) -> pd.DataFrame:
    """Read ``exclusions.csv`` (S12: ``item_id``, ``reason``); reasons must be the registered ones."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    require_columns(df, ("item_id", "reason"), str(path))
    bad = sorted(set(df["reason"]) - set(EXCLUSION_REASONS))
    if bad:
        raise ValueError(f"{path}: unknown exclusion reason(s) {bad}; allowed: {list(EXCLUSION_REASONS)}")
    return df.loc[:, ["item_id", "reason"]].reset_index(drop=True)


def apply_exclusions(df: pd.DataFrame, exclusions: pd.DataFrame | Iterable[str] | None) -> pd.DataFrame:
    """Drop the rows whose ``item_id`` is excluded (an S12 frame or an iterable of item ids)."""
    if exclusions is None:
        return df
    if isinstance(exclusions, pd.DataFrame):
        require_columns(exclusions, ("item_id",), "exclusions")
        ids = set(exclusions["item_id"].astype(str))
    else:
        ids = {str(x) for x in exclusions}
    if not ids:
        return df
    return df.loc[~df["item_id"].astype(str).isin(ids)]


def scoring_frame(parsed: pd.DataFrame, items: pd.DataFrame, condition: str = "image") -> pd.DataFrame:
    """Rows of one condition joined to their S5 item columns on ``item_id``.

    Adds ``valid`` (parsed in yes/no/abstain and not truncated), ``y`` (RANZCR ETT label, 0/1) and
    ``set`` ("pair" for A and B, "U", "DEV"). Raises ValueError when an item id is unknown, when
    the NIH filename of a row differs from the item list (a wrong item list), when an arm's label
    contradicts its definition (A present, B and U absent, DEV pos present and neg absent), or when
    a model x prompt x item appears twice.
    """
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}; expected one of {CONDITIONS}")
    require_columns(parsed, ("model_key", "item_id", "nih_filename", "condition", "prompt_id", "parsed", "truncated", "z"),
                    "parsed results")
    require_columns(items, SCORING_ITEM_COLS, "items (S5)")
    rows = parsed.loc[parsed["condition"].astype(str) == condition].copy()
    rows["item_id"] = rows["item_id"].astype(str)
    keep = list(SCORING_ITEM_COLS) + [c for c in _OPTIONAL_ITEM_COLS if c in items.columns]
    it = items.loc[:, keep].copy()
    it["item_id"] = it["item_id"].astype(str)
    if bool(it["item_id"].duplicated().any()):
        raise ValueError("items: item_id values must be unique")
    it = it.rename(columns={"nih_filename": "_item_nih_filename"})
    joined = rows.merge(it, on="item_id", how="left", validate="many_to_one", sort=False)
    unknown = joined["arm"].isna()
    if bool(unknown.any()):
        ids = sorted(set(joined.loc[unknown, "item_id"]))
        raise ValueError(f"{len(ids)} item id(s) of condition {condition!r} are not in the item list, e.g. {ids[:3]}")
    mismatch = joined["nih_filename"].astype(str) != joined["_item_nih_filename"].astype(str)
    if bool(mismatch.any()):
        raise ValueError(f"{int(mismatch.sum())} row(s) have an NIH filename that differs from the item list")
    joined = joined.drop(columns="_item_nih_filename")
    dup = joined.duplicated(subset=["model_key", "prompt_id", "item_id"])
    if bool(dup.any()):
        raise ValueError(f"{int(dup.sum())} duplicated model x prompt x item row(s); run parse_runs.py to de-duplicate")
    joined["nih_patient_id"] = pd.to_numeric(joined["nih_patient_id"], errors="raise").astype("int64")
    joined["ett"] = pd.to_numeric(joined["ett"], errors="raise").astype("int64")
    joined["n_other_lines"] = pd.to_numeric(joined["n_other_lines"], errors="raise").astype("int64")
    for c in ("arm", "pair_id", "stratum"):
        joined[c] = joined[c].fillna("").astype(str)
    arm = joined["arm"]
    ett = joined["ett"]
    stratum = joined["stratum"]
    bad = (
        ((arm == "A") & (ett != 1)) | (arm.isin(["B", "U"]) & (ett != 0))
        | ((arm == "DEV") & (stratum == "pos") & (ett != 1)) | ((arm == "DEV") & (stratum == "neg") & (ett != 0))
    )
    if bool(bad.any()):
        raise ValueError(f"{int(bad.sum())} item(s) have an ETT label that contradicts their arm")
    joined["y"] = ett.astype("int64")
    joined["set"] = np.where(arm.isin(["A", "B"]), "pair", arm)
    joined["valid"] = joined["parsed"].astype(str).isin(VALID_ANSWERS) & ~joined["truncated"].astype(bool)
    return joined.reset_index(drop=True)


def rows_for(
    df: pd.DataFrame,
    model_key: str,
    prompt_id: str,
    arms: Sequence[str] = TEST_ARMS,
    exclusions: pd.DataFrame | Iterable[str] | None = None,
    condition: str | None = None,
) -> pd.DataFrame:
    """The scoring-frame rows of one model x prompt in ``arms``, after exclusions (condition filter optional)."""
    mask = (df["model_key"].astype(str) == model_key) & (df["prompt_id"].astype(str) == prompt_id)
    mask &= df["arm"].astype(str).isin(list(arms))
    if condition is not None:
        mask &= df["condition"].astype(str) == condition
    return apply_exclusions(df.loc[mask], exclusions)


def effective_answers(sub: pd.DataFrame, unparsable_as_wrong: bool = False) -> np.ndarray:
    """Answers used for scoring: the parsed answer of valid rows, None for excluded rows.

    With ``unparsable_as_wrong`` every row excluded by the pre-registered rule (unparsable or
    truncated) gets the wrong answer for its arm instead ("no" on A and DEV pos, "yes" on B, U
    and DEV neg).
    """
    parsed = sub["parsed"].astype(str).to_numpy(dtype=object)
    valid = sub["valid"].to_numpy(dtype=bool)
    out = np.where(valid, parsed, None).astype(object)
    if unparsable_as_wrong:
        arm = sub["arm"].astype(str).to_numpy(dtype=object)
        y = sub["y"].to_numpy(dtype=np.int64)
        wrong = np.where((arm == "A") | ((arm == "DEV") & (y == 1)), "no", "yes").astype(object)
        out = np.where(valid, out, wrong).astype(object)
    return out


# ----------------------------------------------------------------------------- bootstrap helpers
def clustered_estimate(
    frame: pd.DataFrame,
    stat_fn: Callable[[pd.DataFrame], float],
    n_boot: int,
    seed: int,
    alpha: float,
    strata: bool,
) -> dict[str, Any]:
    """Point estimate and patient-clustered BCa interval of ``stat_fn`` on a small frame.

    ``frame`` has the cluster column "cl" (NIH patient id) and, when ``strata`` is True, the
    strata column "st" (design set). Wraps ``stats.clustered_ci``; ``n_boot <= 0`` gives the
    estimate only (method "none"). Returns plain floats and ints (JSON-serialisable).
    """
    base = {"estimate": _NAN, "lo": _NAN, "hi": _NAN, "method": "none", "n_clusters": 0, "n_rows": int(len(frame)),
            "n_boot": int(n_boot)}
    if len(frame) == 0:
        return base
    if int(n_boot) <= 0:
        est = float(stat_fn(frame))
        return {**base, "estimate": est, "n_clusters": int(frame["cl"].nunique())}
    res = stats.clustered_ci(frame, "cl", stat_fn, n_boot=int(n_boot), seed=int(seed), alpha=alpha,
                             strata_col="st" if strata else None)
    return {
        "estimate": _f(res["estimate"]), "lo": _f(res["lo"]), "hi": _f(res["hi"]), "method": str(res["method"]),
        "p_value": _f(res["p_value"]), "p_flag": str(res["p_flag"]), "n_clusters": int(res["n_clusters"]),
        "n_rows": int(res["n_rows"]), "n_boot": int(res["n_boot"]), "n_boot_valid": int(res["n_boot_valid"]),
    }


def ba_stat(d: pd.DataFrame) -> float:
    """Balanced accuracy (P(yes | A) + P(no | B)) / 2 from columns ``a``, ``yes``, ``no``."""
    a = d["a"].to_numpy(dtype=bool)
    na = int(a.sum())
    nb = a.size - na
    if na == 0 or nb == 0:
        return _NAN
    return 0.5 * (float(d["yes"].to_numpy(dtype=bool)[a].sum()) / na + float(d["no"].to_numpy(dtype=bool)[~a].sum()) / nb)


def mean_stat(d: pd.DataFrame) -> float:
    """Mean of column ``v`` (NaN for an empty frame); a fast ``stat_fn`` for ``clustered_estimate``."""
    v = d["v"].to_numpy(dtype=np.float64)
    return float(v.mean()) if v.size else _NAN


def auroc_stat(d: pd.DataFrame) -> float:
    """AUROC of column ``z`` with ``a`` (True) as positives; NaN when a class is empty."""
    a = d["a"].to_numpy(dtype=bool)
    z = d["z"].to_numpy(dtype=np.float64)
    if a.all() or not a.any():
        return _NAN
    return stats.auroc(z[a], z[~a])


# ----------------------------------------------------------------------------- the H0 gate
def h0_gate(ba: float, hit_a: float, yes_rate: float) -> str:
    """H0 floor gate (PROTOCOL section 6), evaluated in the registered order.

    NO-GO if balanced accuracy < 0.60 or yes-rate < 0.10 or > 0.90 (constant-answer policy);
    else GO if balanced accuracy >= 0.70 and hit_A >= 0.70 and 0.25 <= yes-rate <= 0.75;
    else CONDITIONAL. Thresholds come from ``constants.GATE``. Inputs are rounded to 12 decimals
    before comparison, so a rate equal to a threshold in exact arithmetic is not misclassified by
    floating-point error. Raises ValueError for a non-finite input.
    """
    vals = {"ba": ba, "hit_a": hit_a, "yes_rate": yes_rate}
    for name, v in vals.items():
        if v is None or not math.isfinite(float(v)):
            raise ValueError(f"h0_gate needs finite inputs; {name} = {v!r}")
    b, h, y = (round(float(v), 12) for v in (ba, hit_a, yes_rate))
    g = C.GATE
    if b < g["nogo_ba"] or y < g["const_lo"] or y > g["const_hi"]:
        return "NO-GO"
    if b >= g["go_ba"] and h >= g["go_hit"] and g["go_yes_lo"] <= y <= g["go_yes_hi"]:
        return "GO"
    return "CONDITIONAL"


def h1_contrast(strata: Iterable[str]) -> tuple[str, str, bool] | None:
    """The H1 contrast for the U strata present: (exposed, reference, substitute).

    ("ge2", "eq1", True) when both substitute strata exist (PROTOCOL section 2, registered when
    the no-line stratum has fewer than 30 patients); otherwise ("line", "noline", False) when both
    exist; otherwise None.
    """
    present = set(str(s) for s in strata)
    if {"ge2", "eq1"} <= present:
        return ("ge2", "eq1", True)
    if {"line", "noline"} <= present:
        return ("line", "noline", False)
    return None


# ----------------------------------------------------------------------------- summarise
def summarise(
    df: pd.DataFrame,
    model_key: str,
    prompt_id: str,
    exclusions: pd.DataFrame | Iterable[str] | None = None,
    unparsable_as_wrong: bool = False,
    *,
    condition: str = "image",
    n_boot: int = C.N_BOOT,
    seed: int | None = None,
    alpha: float = C.ALPHA,
    auroc_prompts: Sequence[str] = ("P0",),
) -> dict[str, Any]:
    """Every metric of one model x prompt over the test arms (A, B, U), after exclusions.

    ``df`` is a ``scoring_frame``. Returns a JSON-serialisable dict with:

    * identifiers: ``model_key``, ``prompt_id``, ``condition``, ``unparsable_as_wrong``,
      ``n_rows`` (rows after exclusions), ``n_rows_excluded`` (rows removed by S12);
    * rates as dicts {estimate, k, n, lo, hi, method="wilson"}: ``hit_A``, ``FAR_B``, ``FAR_U``,
      ``FAR_U_<stratum>`` per U stratum, ``FAR_B_line`` / ``FAR_B_noline`` (B strata by other
      labelled lines), ``yes_rate`` (A, B and U), ``abstention`` and ``abstention_A/B/U``,
      ``accuracy`` (correct / n over A, B and U), ``answered`` (coverage = 1 - abstention) and
      ``risk`` (wrong among answered), ``persistence`` = P(yes on B | yes on A) over pairs with A
      yes and B valid, ``pair_correct`` = P(yes on A and no on B) over pairs with both valid,
      ``unparsable_rate``, ``truncated_rate``, ``excluded_rate`` and their ``_A/_B/_U`` versions
      (over all rows, so they are not affected by ``unparsable_as_wrong``);
    * ``ba`` (balanced accuracy on A and B) and ``ER_0.9`` / ``ER_0.5`` (abstentions score 0,
      wrong answers -c with c = t / (1 - t)) as {estimate, lo, hi, method, n, n_clusters, ...}
      with patient-clustered BCa intervals;
    * for prompts in ``auroc_prompts`` (P0 by default): ``auroc_AB`` and ``auroc_AU`` of z with
      clustered BCa intervals;
    * under P0: ``h0_class`` (None in the sensitivity row or when a gate input is undefined);
    * ``flag_unparsable`` (P0 with more than 10% unparsable), ``design_effect`` of the yes
      indicator, ``signal`` and ``coverage_median`` of the first-token signal.

    ``n_boot = 0`` skips the bootstrap (estimates only, method "none").
    """
    seed = C.derive_seed("bootstrap") if seed is None else int(seed)
    require_columns(df, ("model_key", "prompt_id", "condition", "item_id", "arm", "parsed", "valid", "y", "set",
                         "pair_id", "stratum", "nih_patient_id", "z", "n_other_lines", "truncated"), "scoring frame")
    pick = (
        (df["model_key"].astype(str) == model_key) & (df["prompt_id"].astype(str) == prompt_id)
        & (df["condition"].astype(str) == condition) & df["arm"].astype(str).isin(TEST_ARMS)
    )
    before = df.loc[pick]
    sub = apply_exclusions(before, exclusions).reset_index(drop=True)
    arm = sub["arm"].astype(str).to_numpy(dtype=object)
    parsed = sub["parsed"].astype(str).to_numpy(dtype=object)
    truncated = sub["truncated"].to_numpy(dtype=bool)
    ans = effective_answers(sub, unparsable_as_wrong)
    has = np.array([a is not None for a in ans], dtype=bool)
    yes = has & (ans == "yes")
    no = has & (ans == "no")
    abst = has & (ans == "abstain")
    stratum = sub["stratum"].astype(str).to_numpy(dtype=object)
    other = sub["n_other_lines"].to_numpy(dtype=np.int64)
    patient = sub["nih_patient_id"].to_numpy(dtype=np.int64)
    dset = sub["set"].astype(str).to_numpy(dtype=object)
    is_a, is_b, is_u = arm == "A", arm == "B", arm == "U"

    def cnt(mask: np.ndarray) -> int:
        return int(np.count_nonzero(mask))

    out: dict[str, Any] = {
        "model_key": model_key, "prompt_id": prompt_id, "condition": condition,
        "unparsable_as_wrong": bool(unparsable_as_wrong),
        "n_rows": int(len(sub)), "n_rows_excluded": int(len(before) - len(sub)),
        "n_items": int(sub["item_id"].nunique()),
    }
    # rates of excluded outputs (over every row, whatever the scoring mode)
    unp = parsed == "unparsable"
    exc = unp | truncated
    for name, mask in (("unparsable_rate", unp), ("truncated_rate", truncated), ("excluded_rate", exc)):
        out[name] = rate(cnt(mask), len(sub), alpha)
        for a_name, a_mask in (("A", is_a), ("B", is_b), ("U", is_u)):
            out[f"{name}_{a_name}"] = rate(cnt(mask & a_mask), cnt(a_mask), alpha)
    out["flag_unparsable"] = bool(prompt_id == "P0" and out["unparsable_rate"]["n"] > 0
                                  and out["unparsable_rate"]["estimate"] > C.UNPARSABLE_FLAG)

    # item-level rates
    out["hit_A"] = rate(cnt(yes & is_a), cnt(has & is_a), alpha)
    out["FAR_B"] = rate(cnt(yes & is_b), cnt(has & is_b), alpha)
    out["FAR_U"] = rate(cnt(yes & is_u), cnt(has & is_u), alpha)
    for s in [s for s in U_STRATUM_ORDER if bool(np.any(is_u & (stratum == s)))] + sorted(
            set(stratum[is_u]) - set(U_STRATUM_ORDER)):
        m = is_u & (stratum == s)
        out[f"FAR_U_{s}"] = rate(cnt(yes & m), cnt(has & m), alpha)
    out["FAR_B_line"] = rate(cnt(yes & is_b & (other >= 1)), cnt(has & is_b & (other >= 1)), alpha)
    out["FAR_B_noline"] = rate(cnt(yes & is_b & (other == 0)), cnt(has & is_b & (other == 0)), alpha)
    test = is_a | is_b | is_u
    n_test = cnt(has & test)
    out["yes_rate"] = rate(cnt(yes & test), n_test, alpha)
    out["abstention"] = rate(cnt(abst & test), n_test, alpha)
    for a_name, a_mask in (("A", is_a), ("B", is_b), ("U", is_u)):
        out[f"abstention_{a_name}"] = rate(cnt(abst & a_mask), cnt(has & a_mask), alpha)
    correct = (yes & is_a) | (no & (is_b | is_u))
    wrong = (no & is_a) | (yes & (is_b | is_u))
    n_answered = cnt((yes | no) & test)
    out["accuracy"] = rate(cnt(correct), n_test, alpha)
    out["answered"] = rate(n_answered, n_test, alpha)
    out["risk"] = rate(cnt(wrong), n_answered, alpha)
    out["n_correct"], out["n_wrong"], out["n_abstain"] = cnt(correct), cnt(wrong), cnt(abst & test)

    # pair-level rates: pairs whose two members are retained and valid
    pid = sub["pair_id"].astype(str).to_numpy(dtype=object)
    a_ans = {p: v for p, v, m in zip(pid, ans, is_a) if m}
    b_ans = {p: v for p, v, m in zip(pid, ans, is_b) if m}
    common = sorted(set(a_ans) & set(b_ans))
    pers_n = sum(1 for p in common if a_ans[p] == "yes" and b_ans[p] is not None)
    pers_k = sum(1 for p in common if a_ans[p] == "yes" and b_ans[p] == "yes")
    pc_n = sum(1 for p in common if a_ans[p] is not None and b_ans[p] is not None)
    pc_k = sum(1 for p in common if a_ans[p] == "yes" and b_ans[p] == "no")
    out["persistence"] = rate(pers_k, pers_n, alpha)
    out["pair_correct"] = rate(pc_k, pc_n, alpha)

    # clustered intervals: balanced accuracy and ER_t
    ab = has & (is_a | is_b)
    ba_frame = pd.DataFrame({"cl": patient[ab], "st": dset[ab].astype(str), "a": is_a[ab], "yes": yes[ab], "no": no[ab]})
    ba = clustered_estimate(ba_frame, ba_stat, n_boot, seed, alpha, strata=False)
    ba.pop("p_value", None)  # a CI-inversion p against BA = 0 tests nothing; BA is an interval only
    ba.pop("p_flag", None)
    ba["n"] = int(len(ba_frame))
    out["ba"] = ba
    for t in (0.9, 0.5):
        c = er_cost(t)
        score = np.where(correct, 1.0, np.where(wrong, -c, 0.0))
        m = has & test
        er_frame = pd.DataFrame({"cl": patient[m], "st": dset[m].astype(str), "v": score[m]})
        er = clustered_estimate(er_frame, mean_stat, n_boot, seed, alpha, strata=True)
        er.pop("p_value", None)
        er.pop("p_flag", None)
        er.update(n=int(len(er_frame)), t=t, c=c)
        out[f"ER_{t}"] = er

    # AUROC of z (every retained item with a finite z; the parse does not enter)
    if prompt_id in auroc_prompts:
        z = sub["z"].to_numpy(dtype=np.float64)
        fin = np.isfinite(z)
        for key, other_mask in (("auroc_AB", is_b), ("auroc_AU", is_u)):
            m = fin & (is_a | other_mask)
            fr = pd.DataFrame({"cl": patient[m], "st": dset[m].astype(str), "a": is_a[m], "z": z[m]})
            res = clustered_estimate(fr, auroc_stat, n_boot, seed, alpha, strata=(key == "auroc_AU"))
            res.pop("p_value", None)
            res.pop("p_flag", None)
            res.update(n=int(len(fr)), n_pos=cnt(is_a & m), n_neg=cnt(other_mask & m))
            out[key] = res
    if prompt_id == "P0":
        try:
            out["h0_class"] = None if unparsable_as_wrong else h0_gate(
                out["ba"]["estimate"], out["hit_A"]["estimate"], out["yes_rate"]["estimate"])
        except ValueError:
            out["h0_class"] = None

    # design effect of the yes indicator (clusters = NIH patients)
    m = has & test
    if cnt(m):
        de = stats.design_effect(yes[m].astype(float), pd.Series(patient[m]))
        out["design_effect"] = {k: (_f(v) if k != "n_clusters" else int(v)) for k, v in de.items()}
    else:
        out["design_effect"] = {"icc": _NAN, "mean_cluster_size": _NAN, "deff": _NAN, "n_clusters": 0}
    cov = sub["coverage"].to_numpy(dtype=np.float64) if "coverage" in sub.columns else np.array([])
    cov = cov[np.isfinite(cov)]
    out["coverage_median"] = float(np.median(cov)) if cov.size else _NAN
    if "signal" in sub.columns and len(sub):
        sig = sorted(set(sub["signal"].astype(str)))
        out["signal"] = sig[0] if len(sig) == 1 else "mixed:" + "+".join(sig)
    else:
        out["signal"] = ""
    return out


# ----------------------------------------------------------------------------- blind rows
def blind_rows(df: pd.DataFrame, model_key: str) -> pd.DataFrame:
    """BLIND-text and BLIND-grey rows of one model: one deterministic generation per prompt.

    ``df`` is an S8 frame (any conditions). Columns: ``model_key``, ``prompt_id``, ``condition``,
    ``z``, ``p_yes_norm`` = sigmoid(z) (first-token P(yes) renormalised over yes and no forms),
    ``coverage``, ``greedy`` (parsed answer) and ``truncated``. No interval (PROTOCOL section 4).
    Raises ValueError when a prompt x condition appears twice.
    """
    require_columns(df, ("model_key", "condition", "prompt_id", "z", "coverage", "parsed", "truncated"), "parsed results")
    sub = df.loc[(df["model_key"].astype(str) == model_key) & df["condition"].astype(str).isin(BLIND_CONDITIONS)]
    if bool(sub.duplicated(subset=["prompt_id", "condition"]).any()):
        raise ValueError(f"{model_key}: a blind prompt x condition appears more than once")
    z = sub["z"].to_numpy(dtype=np.float64)
    out = pd.DataFrame({
        "model_key": sub["model_key"].astype(str).to_numpy(),
        "prompt_id": sub["prompt_id"].astype(str).to_numpy(),
        "condition": sub["condition"].astype(str).to_numpy(),
        "z": z,
        "p_yes_norm": np.asarray(p_yes_normalised(z), dtype=np.float64).reshape(-1) if z.size else np.array([], dtype=float),
        "coverage": sub["coverage"].to_numpy(dtype=np.float64),
        "greedy": sub["parsed"].astype(str).to_numpy(),
        "truncated": sub["truncated"].to_numpy(dtype=bool),
    })
    prank = {p: i for i, p in enumerate(PROMPT_ORDER)}
    crank = {c: i for i, c in enumerate(BLIND_CONDITIONS)}
    out["_p"] = out["prompt_id"].map(lambda p: prank.get(p, len(prank)))
    out["_c"] = out["condition"].map(crank)
    return out.sort_values(["_p", "prompt_id", "_c"], kind="stable").drop(columns=["_p", "_c"]).reset_index(drop=True)


# ----------------------------------------------------------------------------- risk-coverage (PLUS Figure 2)
def risk_coverage(
    df: pd.DataFrame,
    model_key: str,
    prompt_id: str = "P0",
    exclusions: pd.DataFrame | Iterable[str] | None = None,
) -> pd.DataFrame:
    """Risk-coverage curve of the logit rule (answer yes iff z > 0) ranked by |z| (PLUS Figure 2).

    Over the retained A, B and U items with a finite z: answering only items with |z| >= t gives
    ``coverage`` = answered / n and ``risk`` = errors among answered; one row per distinct |z|
    (descending), columns ``threshold``, ``n_answered``, ``coverage``, ``risk``.
    """
    sub = rows_for(df, model_key, prompt_id, TEST_ARMS, exclusions, condition="image")
    z = sub["z"].to_numpy(dtype=np.float64)
    y = sub["y"].to_numpy(dtype=np.int64)
    fin = np.isfinite(z)
    z, y = z[fin], y[fin]
    if z.size == 0:
        return pd.DataFrame({"threshold": [], "n_answered": [], "coverage": [], "risk": []})
    conf = np.abs(z)
    err = ((z > 0).astype(np.int64) != y).astype(np.int64)
    order = np.argsort(-conf, kind="stable")
    conf_s, err_s = conf[order], err[order]
    cum_err = np.cumsum(err_s)
    last = np.r_[np.flatnonzero(np.diff(conf_s) != 0), conf_s.size - 1]
    n_ans = last + 1
    return pd.DataFrame({
        "threshold": conf_s[last], "n_answered": n_ans.astype(np.int64),
        "coverage": n_ans / conf_s.size, "risk": cum_err[last] / n_ans,
    })
