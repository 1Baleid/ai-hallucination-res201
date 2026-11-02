"""Item selection and every derived list (PIPELINE_SPEC v1.0 section 3, ``items.py``).

PROTOCOL.md section 2 defines the science and section 9 the data terms. This module turns the
matched-image table into the Stage 1 item lists:

* A/B pairs: a within-patient contrast between an ETT-present image (A) and the same NIH
  patient's ETT-absent image nearest in follow-up number (B), one pair per patient, with the
  pre-registered rungs ("two_per_patient", then "n80") when the yield is short.
* U: ETT-absent images of other patients, one per patient, in two strata matched on NIH view
  position ("line" / "noline", or "ge2" / "eq1" under the registered H1 substitute).
* DEV: the patient-disjoint dev pool (150 ETT present, 150 ETT absent) used for the MedSigLIP
  AUROC, tau, the beta sensitivity row and the smoke test, never for a test statistic.
* opaque item ids, the pod item list, the NIH filename list, the blind reading list with its key,
  the retest list and the selection report.

Labels of record are the RANZCR CLiP labels (43 trained labellers: radiologists, radiology
trainees, hospital medical officers); the NIH metadata supplies the patient id, follow-up number,
view position, age and sex. Ground truth is never a detector output.

Data boundaries (PROTOCOL section 9): ``items.csv``, ``reading_key.csv`` and
``selection_report.json`` are local working files (the ``items/`` directory is git-ignored).
``pod_items.csv`` is the only item file that goes to the pod; it carries opaque item ids, NIH
filenames, the prompt set and the smoke flag, and nothing else. ``nih_filelist.csv`` carries NIH
filenames only. No RANZCR uid or label is ever written to either file.

Randomness: each purpose draws from its own ``numpy.random.Generator`` seeded with
``constants.derive_seed(purpose)`` ("pairs", "u", "dev", "ids", "smoke", "reading", "retest").
Candidates are put in a canonical order (NIH patient id, follow-up number, filename) before any
draw, so identical inputs and seeds give byte-identical files whatever the input row order.
Library code never touches the network.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cxrdev import constants as C

__all__ = [
    "ARMS",
    "DEV_STRATA",
    "DIRECTIONS",
    "ITEM_COLS",
    "LABEL_COLS",
    "MASTER_COLS",
    "MATCH_COLS",
    "NIH_FILELIST_COLS",
    "NIH_META_COLS",
    "META_FILES",
    "OUTPUT_FILES",
    "PRACTICE_N",
    "POD_ITEM_COLS",
    "PRESENCE_COLS",
    "PROMPT_SET_BY_ARM",
    "READING_KEY_COLS",
    "READING_LIST_COLS",
    "RUNGS",
    "SelectionOutputs",
    "U_STRATA",
    "VIEWS",
    "assign_item_ids",
    "build_master_table",
    "build_outputs",
    "mark_smoke",
    "nih_filelist",
    "pod_items",
    "practice_list",
    "practice_seed",
    "read_filename_list",
    "read_items",
    "remove_filenames",
    "reading_lists",
    "retest_list",
    "select_core",
    "selection_report",
    "write_outputs",
]

# ----------------------------------------------------------------------------- schemas (spec section 2)
LABEL_COLS: tuple[str, ...] = ("ett", "ngt", "ngt_incomplete", "cvc", "swan")
MATCH_COLS: tuple[str, ...] = ("uid", "ranzcr_patient_id", "nih_filename", "nih_patient_id", "status")
PRESENCE_COLS: tuple[str, ...] = ("uid", "ranzcr_patient_id", *LABEL_COLS, "n_other_lines")
NIH_META_COLS: tuple[str, ...] = ("nih_filename", "nih_patient_id", "followup", "age", "sex", "view")
MASTER_COLS: tuple[str, ...] = (
    "uid", "ranzcr_patient_id", "nih_filename", "nih_patient_id", "followup", "view", "age", "sex",
    *LABEL_COLS, "n_other_lines",
)
ITEM_COLS: tuple[str, ...] = (
    "item_id", "uid", "nih_filename", "nih_patient_id", "arm", "pair_id", "stratum",
    *LABEL_COLS, "n_other_lines", "view", "followup", "age", "sex", "direction", "gap", "rung", "smoke",
)
POD_ITEM_COLS: tuple[str, ...] = ("item_id", "nih_filename", "prompt_set", "smoke")
NIH_FILELIST_COLS: tuple[str, ...] = ("nih_filename",)
READING_LIST_COLS: tuple[str, ...] = ("read_id", "nih_filename", "image_path")
READING_KEY_COLS: tuple[str, ...] = ("read_id", "item_id")

ARMS: tuple[str, ...] = ("A", "B", "U", "DEV")
RUNGS: tuple[str, ...] = ("standard", "two_per_patient", "n80")
DIRECTIONS: tuple[str, ...] = ("B_after_A", "B_before_A")
VIEWS: tuple[str, ...] = ("AP", "PA")
U_STRATA: tuple[str, ...] = ("line", "noline", "ge2", "eq1")
DEV_STRATA: tuple[str, ...] = ("pos", "neg")
PROMPT_SET_BY_ARM: dict[str, str] = {"A": "full", "B": "full", "U": "full", "DEV": "p0"}
ITEM_ID_HEX_LEN = 10
READ_ID_HEX_LEN = 8
#: images in the practice list (mirrors ``tools/reading_core.PRACTICE_N``)
PRACTICE_N = 10
#: purpose index of the practice-list seed (master seed x 100 + 50). ``constants._PURPOSES`` is fixed
#: and has no practice entry; 50 is outside its 1..11 range, so the seed never equals another purpose's.
PRACTICE_SEED_INDEX = 50

OUTPUT_FILES: dict[str, str] = {
    "items": "items.csv",
    "pod_items": "pod_items.csv",
    "nih_filelist": "nih_filelist.csv",
    "reading_list": "reading_list.csv",
    "reading_key": "reading_key.csv",
    "retest_list": "retest_list.csv",
    "practice_list": "practice_list.csv",
    "selection_report": "selection_report.json",
}
#: Seed sidecars of the lists the reading tool opens (``tools/reading_core.read_list_seed`` reads
#: ``<list stem>.meta.json`` = {"seed": ...}).
META_FILES: dict[str, str] = {
    "reading_list": "reading_list.meta.json",
    "retest_list": "retest_list.meta.json",
    "practice_list": "practice_list.meta.json",
}

_MASTER_STR_COLS: tuple[str, ...] = ("uid", "ranzcr_patient_id", "nih_filename", "view", "sex")
_ITEM_STR_COLS: tuple[str, ...] = (
    "item_id", "uid", "nih_filename", "arm", "pair_id", "stratum", "view", "sex", "direction", "rung",
)
_ITEM_INT_COLS: tuple[str, ...] = tuple(c for c in ITEM_COLS if c not in _ITEM_STR_COLS)
_TABLE_COLS: dict[str, tuple[str, ...]] = {
    "items": ITEM_COLS,
    "pod_items": POD_ITEM_COLS,
    "nih_filelist": NIH_FILELIST_COLS,
    "reading_list": READING_LIST_COLS,
    "reading_key": READING_KEY_COLS,
    "retest_list": READING_LIST_COLS,
    "practice_list": READING_LIST_COLS,
}


# ----------------------------------------------------------------------------- small helpers
def _require_columns(df: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    """Raise ValueError naming every column of ``columns`` missing from ``df``."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing column(s) {missing}")


def _as_int(series: pd.Series, frame: str, col: str) -> pd.Series:
    """Return ``series`` as int64, raising ValueError on missing or non-integer values."""
    if len(series) == 0:
        return series.astype("int64")
    try:
        num = pd.to_numeric(series, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{frame}: column '{col}' must hold integers") from exc
    if num.isna().any():
        raise ValueError(f"{frame}: column '{col}' has {int(num.isna().sum())} missing value(s)")
    if pd.api.types.is_bool_dtype(num) or pd.api.types.is_integer_dtype(num):
        return num.astype("int64")
    if pd.api.types.is_float_dtype(num) and bool((num == np.floor(num)).all()):
        return num.astype("int64")
    raise ValueError(f"{frame}: column '{col}' must hold integers")


def _as_str(series: pd.Series, frame: str, col: str, *, allow_empty: bool = False) -> pd.Series:
    """Return ``series`` as strings, raising ValueError on missing (or, by default, empty) values."""
    if series.isna().any():
        raise ValueError(f"{frame}: column '{col}' has {int(series.isna().sum())} missing value(s)")
    out = series.astype(str)
    if not allow_empty and bool((out.str.len() == 0).any()):
        raise ValueError(f"{frame}: column '{col}' has {int((out.str.len() == 0).sum())} empty value(s)")
    return out


def _check_unique(df: pd.DataFrame, cols: Sequence[str], frame: str, hint: str = "") -> None:
    """Raise ValueError when the key ``cols`` is duplicated in ``df`` (counts only, no values)."""
    dup = df.duplicated(subset=list(cols), keep=False)
    if bool(dup.any()):
        key = "/".join(cols)
        raise ValueError(f"{frame}: {int(dup.sum())} row(s) share a duplicated '{key}'{hint}")


def _purpose_seed(seed: int, purpose: str) -> int:
    """Seed for ``purpose`` under the master seed ``seed``.

    Equals ``constants.derive_seed(purpose)`` when ``seed == constants.SEED`` and applies the same
    derivation (master seed x 100 + purpose index) to any other master seed.
    """
    if int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    return int(seed) * 100 + (C.derive_seed(purpose) - C.SEED * 100)


def _round_half_up(num: int, den: int) -> int:
    """round(num / den) with halves rounded up, in exact integer arithmetic (num >= 0, den > 0)."""
    return (2 * int(num) + int(den)) // (2 * int(den))


def _reads_as_number(text: str) -> bool:
    """True when a CSV reader would parse ``text`` as a number (all digits, or digits-e-digits)."""
    try:
        float(text)
    except ValueError:
        return False
    return True


def _random_hex_ids(rng: np.random.Generator, n: int, length: int, *, upper: bool) -> list[str]:
    """``n`` unique random hex strings of ``length`` characters drawn from ``rng``.

    Strings that a CSV reader would parse as a number (for example "0012345678" or "12345e6789")
    are redrawn, so every id reads back as text with its leading zeros, even from a file in which
    all ids happen to be numeric-looking. Each id is an independent draw: it carries no
    information about the row it is attached to.
    """
    fmt = f"0{length}{'X' if upper else 'x'}"
    high = 1 << (4 * length)
    seen: set[str] = set()
    out: list[str] = []
    while len(out) < n:
        for value in rng.integers(0, high, size=n - len(out), dtype=np.uint64).tolist():
            text = format(int(value), fmt)
            if text in seen or _reads_as_number(text):
                continue
            seen.add(text)
            out.append(text)
    return out


def _sha256_bytes(data: bytes) -> str:
    """Hex SHA-256 of ``data``."""
    return hashlib.sha256(data).hexdigest()


def _csv_bytes(df: pd.DataFrame, columns: Sequence[str], name: str) -> bytes:
    """UTF-8 CSV bytes with ``\\n`` line endings, no index and exactly ``columns`` in that order."""
    if list(df.columns) != list(columns):
        raise ValueError(f"{name}: columns must be exactly {list(columns)}, got {list(df.columns)}")
    buf = io.StringIO()
    df.to_csv(buf, index=False, lineterminator="\n")
    return buf.getvalue().encode("utf-8")


def _jsonable(obj: Any) -> Any:
    """Convert numpy scalars, tuples and sets to plain JSON types, recursively."""
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return [_jsonable(v) for v in sorted(obj)]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def _json_bytes(obj: Any) -> bytes:
    """Deterministic JSON bytes (sorted keys, two-space indent, trailing newline)."""
    return (json.dumps(_jsonable(obj), indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def _image_path(image_dir: str | Path, filename: str) -> str:
    """``image_dir/filename`` with forward slashes (valid on Windows and POSIX)."""
    return (Path(image_dir) / filename).as_posix()


# ----------------------------------------------------------------------------- master table (S4)
def _normalise_master(master: pd.DataFrame, frame: str = "master") -> pd.DataFrame:
    """Validate an S4 frame and return a typed copy in canonical order.

    Checks: required columns; integer labels in {0, 1}; ``ngt_incomplete`` implies ``ngt``;
    ``n_other_lines == ngt + cvc + swan``; view in {"AP", "PA"}; unique uid, unique NIH filename
    and unique (NIH patient, follow-up number); each RANZCR patient maps to one NIH patient.
    Rows are sorted by (nih_patient_id, followup, nih_filename) and re-indexed 0..n-1.
    """
    _require_columns(master, MASTER_COLS, frame)
    src = master.loc[:, list(MASTER_COLS)].reset_index(drop=True)
    out = pd.DataFrame(
        {
            col: (_as_str(src[col], frame, col) if col in _MASTER_STR_COLS else _as_int(src[col], frame, col))
            for col in MASTER_COLS
        }
    )
    for col in LABEL_COLS:
        if not bool(out[col].isin([0, 1]).all()):
            raise ValueError(f"{frame}: column '{col}' must be 0/1")
    if bool(((out["ngt_incomplete"] == 1) & (out["ngt"] == 0)).any()):
        raise ValueError(f"{frame}: ngt_incomplete = 1 requires ngt = 1 (incompletely imaged counts as present)")
    expected = out["ngt"] + out["cvc"] + out["swan"]
    if bool((out["n_other_lines"] != expected).any()):
        raise ValueError(f"{frame}: n_other_lines must equal ngt + cvc + swan")
    bad_view = ~out["view"].isin(VIEWS)
    if bool(bad_view.any()):
        raise ValueError(
            f"{frame}: view must be 'AP' or 'PA'; found {sorted(set(out.loc[bad_view, 'view']))}"
        )
    _check_unique(out, ["uid"], frame)
    _check_unique(out, ["nih_filename"], frame)
    _check_unique(out, ["nih_patient_id", "followup"], frame, " (follow-up numbers must be unique per NIH patient)")
    per_patient = out.groupby("ranzcr_patient_id")["nih_patient_id"].nunique()
    if bool((per_patient > 1).any()):
        raise ValueError(
            f"{frame}: {int((per_patient > 1).sum())} RANZCR patient(s) map to more than one NIH patient "
            "(patient consistency must hold before selection)"
        )
    out = out.sort_values(["nih_patient_id", "followup", "nih_filename"]).reset_index(drop=True)
    return out


def build_master_table(match_table: pd.DataFrame, presence: pd.DataFrame, nih_meta: pd.DataFrame) -> pd.DataFrame:
    """Join the matched images with their RANZCR CLiP labels and NIH metadata (S4).

    Args:
        match_table: S3 frame from ``matching.match_all``; only rows with status "matched" are used.
        presence: S1 frame from ``ranzcr.presence`` (RANZCR CLiP labels, 43 trained labellers).
        nih_meta: S2 frame from ``nih.load_nih_meta``.

    Returns:
        The S4 ``master`` frame, one row per matched image, columns ``MASTER_COLS``, sorted by
        (nih_patient_id, followup, nih_filename).

    Raises:
        ValueError: a required column is missing; a matched row lacks an NIH filename or patient
            id; a matched uid has no labels or a matched filename has no NIH metadata; the uid or
            the NIH filename is not one-to-one; the RANZCR patient id or the NIH patient id
            disagrees between tables; or any check of the S4 schema fails.
    """
    _require_columns(match_table, MATCH_COLS, "match_table")
    _require_columns(presence, PRESENCE_COLS, "presence")
    _require_columns(nih_meta, NIH_META_COLS, "nih_meta")

    matched = match_table.loc[match_table["status"].isin(["matched"]), list(MATCH_COLS)].reset_index(drop=True)
    frame = "match_table (matched rows)"
    links = pd.DataFrame(
        {
            "uid": _as_str(matched["uid"], frame, "uid"),
            "ranzcr_patient_id": _as_str(matched["ranzcr_patient_id"], frame, "ranzcr_patient_id"),
            "nih_filename": _as_str(matched["nih_filename"], frame, "nih_filename"),
            "nih_patient_id": _as_int(matched["nih_patient_id"], frame, "nih_patient_id"),
        }
    )
    if bool((links["nih_patient_id"] < 0).any()):
        raise ValueError(f"{frame}: every matched row needs an NIH patient id (found -1)")
    _check_unique(links, ["uid"], frame)
    _check_unique(links, ["nih_filename"], frame, " (one-to-one resolution expected)")

    pres_src = presence.loc[:, list(PRESENCE_COLS)].reset_index(drop=True)
    pres = pd.DataFrame(
        {
            "uid": _as_str(pres_src["uid"], "presence", "uid"),
            "presence_patient_id": _as_str(pres_src["ranzcr_patient_id"], "presence", "ranzcr_patient_id"),
            **{c: _as_int(pres_src[c], "presence", c) for c in (*LABEL_COLS, "n_other_lines")},
        }
    )
    _check_unique(pres, ["uid"], "presence")

    meta_src = nih_meta.loc[:, list(NIH_META_COLS)].reset_index(drop=True)
    meta = pd.DataFrame(
        {
            "nih_filename": _as_str(meta_src["nih_filename"], "nih_meta", "nih_filename"),
            "meta_patient_id": _as_int(meta_src["nih_patient_id"], "nih_meta", "nih_patient_id"),
            "followup": _as_int(meta_src["followup"], "nih_meta", "followup"),
            "age": _as_int(meta_src["age"], "nih_meta", "age"),
            "sex": _as_str(meta_src["sex"], "nih_meta", "sex"),
            "view": _as_str(meta_src["view"], "nih_meta", "view"),
        }
    )
    _check_unique(meta, ["nih_filename"], "nih_meta")

    no_labels = ~links["uid"].isin(pres["uid"])
    if bool(no_labels.any()):
        raise ValueError(f"{frame}: {int(no_labels.sum())} matched uid(s) have no row in presence")
    no_meta = ~links["nih_filename"].isin(meta["nih_filename"])
    if bool(no_meta.any()):
        raise ValueError(f"{frame}: {int(no_meta.sum())} matched NIH filename(s) have no row in nih_meta")

    out = links.merge(pres, on="uid", how="left", validate="one_to_one")
    out = out.merge(meta, on="nih_filename", how="left", validate="one_to_one")
    if bool((out["ranzcr_patient_id"] != out["presence_patient_id"]).any()):
        raise ValueError("match_table and presence disagree on ranzcr_patient_id for some uid(s)")
    if bool((out["nih_patient_id"] != out["meta_patient_id"]).any()):
        raise ValueError("match_table and nih_meta disagree on nih_patient_id for some NIH filename(s)")
    return _normalise_master(out.loc[:, list(MASTER_COLS)], "master")


# ----------------------------------------------------------------------------- selection machinery
def _capped_assign(
    rng: np.random.Generator,
    patients: Sequence[int],
    can_a: Sequence[bool],
    can_b: Sequence[bool],
    n_a: int,
    n_b: int,
) -> dict[int, bool]:
    """Draw ``n_a`` patients for class a and ``n_b`` for class b, one class per patient.

    Patients are visited in a random order and accepted while their class quota allows
    (a patient able to serve both classes is always accepted until the total is reached); the
    flexible patients then fill the remaining slots of each class in a second random order. The
    caller guarantees feasibility: n_a <= #can_a, n_b <= #can_b and n_a + n_b <= #patients.

    Returns:
        {patient: True for class a, False for class b} for the selected patients.
    """
    total = n_a + n_b
    only_a: list[int] = []
    only_b: list[int] = []
    flex: list[int] = []
    for i in rng.permutation(len(patients)).tolist():
        if len(only_a) + len(only_b) + len(flex) == total:
            break
        if can_a[i] and can_b[i]:
            flex.append(i)
        elif can_a[i]:
            if len(only_a) < n_a:
                only_a.append(i)
        elif can_b[i]:
            if len(only_b) < n_b:
                only_b.append(i)
    if len(only_a) + len(only_b) + len(flex) != total:
        raise RuntimeError("internal error: infeasible class quota passed to _capped_assign")
    need_a = n_a - len(only_a)
    out: dict[int, bool] = {int(patients[i]): True for i in only_a}
    out.update({int(patients[i]): False for i in only_b})
    for rank, j in enumerate(rng.permutation(len(flex)).tolist()):
        out[int(patients[flex[j]])] = rank < need_a
    return out


def _class_range(n: int, only_a: int, only_b: int, both: int) -> tuple[int, int]:
    """Feasible range of class-a counts when ``n`` patients (n <= total) are drawn one per patient."""
    lo = max(0, n - (only_b + both))
    hi = min(n, only_a + both)
    return lo, hi


def _joint_ap_counts(
    n_1: int, range_1: tuple[int, int], target_1: int,
    n_2: int, range_2: tuple[int, int], target_2: int,
) -> tuple[int, int]:
    """AP counts (k_1, k_2) for the two H1 contrast strata of sizes n_1 and n_2.

    Lexicographic choice over the feasible ranges: (1) the closest AP shares between the strata,
    |k_1 / n_1 - k_2 / n_2| (equal AP and PA counts whenever the sizes are equal and the ranges
    overlap); (2) the closest to the targets (the B arm's AP share times the stratum size,
    rounded half up); (3) the smaller counts.
    """
    best: tuple[int, int, int, int] | None = None
    for k_1 in range(range_1[0], range_1[1] + 1):
        for k_2 in range(range_2[0], range_2[1] + 1):
            key = (abs(k_1 * n_2 - k_2 * n_1), abs(k_1 - target_1) + abs(k_2 - target_2), k_1, k_2)
            if best is None or key < best:
                best = key
    if best is None:  # pragma: no cover - ranges are never empty (see _class_range)
        raise RuntimeError("internal error: empty AP-count range")
    return best[2], best[3]


def _matched_ap_count(n: int, range_: tuple[int, int], target: int, n_partner: int, k_partner: int) -> int:
    """AP count for a stratum of size ``n`` drawn after its contrast partner (n_partner, k_partner).

    Same criteria as ``_joint_ap_counts`` with the partner fixed: the closest AP share to the
    partner's realised share, then the closest to ``target``, then the smaller count.
    """
    return min(
        range(range_[0], range_[1] + 1),
        key=lambda k: (abs(k * n_partner - k_partner * n), abs(k - target), k),
    )


def _view_availability(pool: Mapping[int, Mapping[str, list[int]]]) -> dict[str, int]:
    """Patients in ``pool`` with AP images only, PA images only, or both views."""
    only_ap = sum(1 for v in pool.values() if v["AP"] and not v["PA"])
    only_pa = sum(1 for v in pool.values() if v["PA"] and not v["AP"])
    both = sum(1 for v in pool.values() if v["AP"] and v["PA"])
    return {"patients": len(pool), "ap_only": only_ap, "pa_only": only_pa, "both_views": both}


def _nearest_absent(rng: np.random.Generator, absent_rows: Sequence[int], followup: np.ndarray, f_a: int) -> int:
    """The ETT-absent row nearest in follow-up number to ``f_a`` (ties broken at random)."""
    dist = [abs(int(followup[r]) - f_a) for r in absent_rows]
    best = min(dist)
    ties = [r for r, d in zip(absent_rows, dist) if d == best]
    return ties[0] if len(ties) == 1 else ties[int(rng.integers(len(ties)))]


def _select_pairs(
    m: pd.DataFrame, n_pairs: int, rng: np.random.Generator
) -> tuple[list[tuple[int, int]], str, dict[str, Any]]:
    """A/B pairs (rows of ``m``), the rung and the pair section of the report."""
    pid = m["nih_patient_id"].to_numpy()
    ett = m["ett"].to_numpy()
    followup = m["followup"].to_numpy()
    present: dict[int, list[int]] = {}
    absent: dict[int, list[int]] = {}
    for r in range(len(m)):  # m is sorted by patient and follow-up, so the lists are too
        (present if ett[r] == 1 else absent).setdefault(int(pid[r]), []).append(r)
    eligible = sorted(p for p in present if p in absent)
    second_ok = [p for p in eligible if len(present[p]) >= 2 and len(absent[p]) >= 2]
    n1, n2 = len(eligible), len(second_ok)
    n_min = C.N_PAIRS_RUNG_MIN
    if n1 >= n_pairs:
        rung, n_target = "standard", n_pairs
    elif n1 + n2 >= n_pairs:
        rung, n_target = "two_per_patient", n_pairs
    elif n1 + n2 >= n_min:
        rung, n_target = "n80", n_min
    else:
        raise ValueError(
            f"pair yield too low: {n1} eligible NIH patient(s) with at least one ETT-present and one "
            f"ETT-absent image and {n2} with room for a second pair give at most {n1 + n2} pairs; "
            f"{n_pairs} are requested and the last rung (n80) needs at least {n_min} "
            "(PROTOCOL section 2 rungs; section 8 fallback branches)"
        )

    order = rng.permutation(n1).tolist()
    if n1 >= n_target:
        first_patients = [eligible[i] for i in order[:n_target]]
        second_patients: list[int] = []
    else:
        first_patients = [eligible[i] for i in order]
        second_patients = [second_ok[i] for i in rng.permutation(n2).tolist()[: n_target - n1]]

    pairs: list[tuple[int, int]] = []
    first: dict[int, tuple[int, int]] = {}
    for p in first_patients:
        a = present[p][int(rng.integers(len(present[p])))]
        b = _nearest_absent(rng, absent[p], followup, int(followup[a]))
        first[p] = (a, b)
        pairs.append((a, b))
    for p in second_patients:
        a1, b1 = first[p]
        rest_present = [r for r in present[p] if r != a1]
        a = rest_present[int(rng.integers(len(rest_present)))]
        b = _nearest_absent(rng, [r for r in absent[p] if r != b1], followup, int(followup[a]))
        pairs.append((a, b))

    info = {
        "n_target": int(n_pairs),
        "n_selected": len(pairs),
        "rung": rung,
        "rung_min": int(n_min),
        "n_eligible_patients": n1,
        "n_second_pair_patients_available": n2,
        "n_pair_patients": len(first_patients),
        "n_patients_with_two_pairs": len(second_patients),
    }
    return pairs, rung, info


def _select_u(
    m: pd.DataFrame,
    excluded: set[int],
    n_ap_b: int,
    n_b: int,
    n_per_stratum: int,
    rng: np.random.Generator,
    h1_override: bool | None,
    keep_noline_descriptive: bool,
) -> tuple[list[tuple[int, str]], tuple[str, ...], dict[str, Any]]:
    """U items as (row, stratum), the stratum order and the U section of the report."""
    pid = m["nih_patient_id"].to_numpy()
    lines = m["n_other_lines"].to_numpy()
    view = m["view"].to_numpy()
    pool_rows = [r for r in np.flatnonzero(m["ett"].to_numpy() == 0).tolist() if int(pid[r]) not in excluded]

    noline_available = {int(pid[r]) for r in pool_rows if lines[r] == 0}
    absent_noline = (m["ett"] == 0) & (m["n_other_lines"] == 0)
    noline_master = int(m.loc[absent_noline, "nih_patient_id"].nunique())
    auto = len(noline_available) < C.H1_SUBSTITUTE_MIN_PATIENTS
    substitute = auto if h1_override is None else bool(h1_override)
    if substitute:
        exposed, reference = "ge2", "eq1"
        descriptive: tuple[str, ...] = ("noline",) if keep_noline_descriptive else ()
    else:
        exposed, reference = "line", "noline"
        descriptive = ()
    strata: tuple[str, ...] = (exposed, reference, *descriptive)

    def stratum_of(k: int) -> str:
        if k == 0:
            return "noline"
        if substitute:
            return "eq1" if k == 1 else "ge2"
        return "line"

    # images per stratum -> patient -> view (rows stay in follow-up order)
    capable: dict[str, dict[int, dict[str, list[int]]]] = {s: {} for s in strata}
    for r in pool_rows:
        s = stratum_of(int(lines[r]))
        if s in capable:
            capable[s].setdefault(int(pid[r]), {"AP": [], "PA": []})[str(view[r])].append(r)

    # Draw order: "noline" first (its patient count drives the H1 substitute rule and it is the
    # scarce stratum by design), then the other strata in ascending order of eligible patients
    # (ties by name). Each stratum draws from its eligible patients not already selected by an
    # earlier stratum, so a patient eligible for two strata stays available to the later one
    # unless the earlier one selected them.
    order = (["noline"] if "noline" in strata else []) + sorted(
        (s for s in strata if s != "noline"), key=lambda name: (len(capable[name]), name)
    )
    partner_of = {exposed: reference, reference: exposed}
    taken: set[int] = set()
    size: dict[str, int] = {}
    ranges: dict[str, tuple[int, int]] = {}
    target: dict[str, int] = {}
    k_ap: dict[str, int] = {}
    availability: dict[str, dict[str, int]] = {}
    selected: list[tuple[int, str]] = []
    for s in order:
        pool = {p: v for p, v in capable[s].items() if p not in taken}
        availability[s] = _view_availability(pool)
        size[s] = min(n_per_stratum, len(pool))
        ranges[s] = _class_range(size[s], availability[s]["ap_only"], availability[s]["pa_only"],
                                 availability[s]["both_views"])
        target[s] = _round_half_up(n_ap_b * size[s], n_b)
        partner = partner_of.get(s)
        if partner is None:  # descriptive stratum: as close to the B arm's AP share as available
            k_ap[s] = min(max(target[s], ranges[s][0]), ranges[s][1])
        elif partner in k_ap:  # partner already drawn: match its realised AP share
            k_ap[s] = _matched_ap_count(size[s], ranges[s], target[s], size[partner], k_ap[partner])
        else:  # plan both contrast strata jointly; the partner's pool can only grow from here
            partner_pool = {p: v for p, v in capable[partner].items() if p not in taken and p not in pool}
            avail_p = _view_availability(partner_pool)
            n_p = min(n_per_stratum, len(partner_pool))
            range_p = _class_range(n_p, avail_p["ap_only"], avail_p["pa_only"], avail_p["both_views"])
            k_ap[s], _ = _joint_ap_counts(
                size[s], ranges[s], target[s], n_p, range_p, _round_half_up(n_ap_b * n_p, n_b)
            )
        patients = sorted(pool)
        chosen = _capped_assign(
            rng, patients, [bool(pool[p]["AP"]) for p in patients], [bool(pool[p]["PA"]) for p in patients],
            k_ap[s], size[s] - k_ap[s],
        )
        for p in sorted(chosen):
            options = pool[p]["AP" if chosen[p] else "PA"]
            selected.append((options[int(rng.integers(len(options)))], s))
        taken |= set(chosen)

    info = {
        "h1_substitute": substitute,
        "h1_substitute_rule": "auto" if h1_override is None else "forced",
        "h1_substitute_auto_decision": auto,
        "h1_substitute_min_patients": int(C.H1_SUBSTITUTE_MIN_PATIENTS),
        "h1_contrast": {"exposed": exposed, "reference": reference},
        "descriptive_strata": list(descriptive),
        "n_noline_patients_available": len(noline_available),
        "n_noline_patients_master": noline_master,
        "target_per_stratum": int(n_per_stratum),
        "target_ap_share": round(n_ap_b / n_b, 6),
        # equal AP shares in the two contrast strata (None when either stratum is empty)
        "view_matched": (
            size[exposed] * k_ap[reference] == size[reference] * k_ap[exposed]
            if size[exposed] and size[reference] else None
        ),
        "short_strata": [s for s in strata if size[s] < n_per_stratum],
        "draw_order": order,
        "strata": {
            s: {
                "eligible_patients": len(capable[s]),
                "available": availability[s],
                "selected": size[s],
                "short": size[s] < n_per_stratum,
                "target_ap": target[s],
                "ap_range": list(ranges[s]),
                "ap": k_ap[s],
                "pa": size[s] - k_ap[s],
            }
            for s in strata
        },
    }
    return selected, strata, info


def _dev_targets(n_pos: int, n_neg: int, only_pos: int, only_neg: int, both: int) -> tuple[int, int]:
    """Feasible (pos, neg) counts: the most items, then the closest to the requested ratio."""
    best: tuple[int, int, int] | None = None
    for p in range(0, min(n_pos, only_pos + both) + 1):
        q = min(n_neg, only_neg + both, only_pos + only_neg + both - p)
        if q < 0:
            continue
        key = (-(p + q), abs(p * n_neg - q * n_pos), p)
        if best is None or key < best:
            best = key
    if best is None:  # pragma: no cover - p = 0 is always feasible
        raise RuntimeError("internal error: no feasible dev-pool counts")
    return best[2], -best[0] - best[2]


def _select_dev(
    m: pd.DataFrame, excluded: set[int], n_pos: int, n_neg: int, rng: np.random.Generator
) -> tuple[list[tuple[int, str]], dict[str, Any]]:
    """DEV items as (row, "pos" | "neg") and the DEV section of the report."""
    pid = m["nih_patient_id"].to_numpy()
    ett = m["ett"].to_numpy()
    pool: dict[int, dict[int, list[int]]] = {}
    for r in range(len(m)):
        p = int(pid[r])
        if p not in excluded:
            pool.setdefault(p, {1: [], 0: []})[int(ett[r])].append(r)
    patients = sorted(pool)
    can_pos = [bool(pool[p][1]) for p in patients]
    can_neg = [bool(pool[p][0]) for p in patients]
    only_pos = sum(1 for a, b in zip(can_pos, can_neg) if a and not b)
    only_neg = sum(1 for a, b in zip(can_pos, can_neg) if b and not a)
    both = sum(1 for a, b in zip(can_pos, can_neg) if a and b)
    k_pos, k_neg = _dev_targets(n_pos, n_neg, only_pos, only_neg, both)
    chosen = _capped_assign(rng, patients, can_pos, can_neg, k_pos, k_neg)
    selected: list[tuple[int, str]] = []
    for p in sorted(chosen):
        options = pool[p][1 if chosen[p] else 0]
        selected.append((options[int(rng.integers(len(options)))], "pos" if chosen[p] else "neg"))
    info = {
        "target": {"pos": int(n_pos), "neg": int(n_neg)},
        "available": {"patients": len(patients), "pos_only": only_pos, "neg_only": only_neg, "both_states": both},
        "selected": {"pos": k_pos, "neg": k_neg},
        "short": k_pos < n_pos or k_neg < n_neg,
    }
    return selected, info


def _item_row(rec: Mapping[str, Any], arm: str, pair_id: str, stratum: str, direction: str, gap: int, rung: str) -> dict[str, Any]:
    """One S5 row (item_id left empty, smoke 0) from a master record."""
    row: dict[str, Any] = {
        "item_id": "",
        "uid": rec["uid"],
        "nih_filename": rec["nih_filename"],
        "nih_patient_id": int(rec["nih_patient_id"]),
        "arm": arm,
        "pair_id": pair_id,
        "stratum": stratum,
    }
    row.update({c: int(rec[c]) for c in (*LABEL_COLS, "n_other_lines")})
    row.update(
        {
            "view": rec["view"],
            "followup": int(rec["followup"]),
            "age": int(rec["age"]),
            "sex": rec["sex"],
            "direction": direction,
            "gap": int(gap),
            "rung": rung,
            "smoke": 0,
        }
    )
    return row


def _typed_items(rows: list[dict[str, Any]] | pd.DataFrame) -> pd.DataFrame:
    """An S5 frame with ``ITEM_COLS`` in order, int64 numbers and string text columns."""
    df = pd.DataFrame(rows, columns=list(ITEM_COLS)) if isinstance(rows, list) else rows.loc[:, list(ITEM_COLS)]
    df = df.reset_index(drop=True)
    return pd.DataFrame(
        {
            c: (_as_str(df[c], "items", c, allow_empty=True) if c in _ITEM_STR_COLS else _as_int(df[c], "items", c))
            for c in ITEM_COLS
        }
    )


# ----------------------------------------------------------------------------- public selection API
def select_core(
    master: pd.DataFrame,
    seed: int = C.SEED,
    n_pairs: int = C.N_PAIRS,
    n_u_per_stratum: int = C.N_U_PER_STRATUM,
    n_dev_pos: int = C.N_DEV_POS,
    n_dev_neg: int = C.N_DEV_NEG,
    *,
    h1_substitute: bool | None = None,
    keep_noline_descriptive: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Select the A/B pairs, the U strata and the dev pool (PROTOCOL section 2).

    Pairs. Eligible NIH patients have at least one ETT-present and one ETT-absent image per the
    RANZCR CLiP labels. Within a selected patient, A is drawn uniformly among the ETT-present
    images and B is the ETT-absent image minimising |followup(B) - followup(A)| (ties at random);
    ``direction`` is "B_after_A" or "B_before_A" and ``gap`` the absolute difference. Rungs, with
    E1 eligible patients and E2 of them holding at least two ETT-present and two ETT-absent images:
    "standard" when E1 >= n_pairs (n_pairs random patients, one pair each); "two_per_patient" when
    E1 < n_pairs <= E1 + E2 (every eligible patient gives a pair and n_pairs - E1 random E2
    patients give a second pair, disjoint from their first); "n80" when E1 + E2 < n_pairs but
    E1 + E2 >= constants.N_PAIRS_RUNG_MIN (80 pairs, one per patient when E1 >= 80, otherwise
    every eligible patient plus second pairs as above). Fewer than 80 possible pairs raises.

    U. Patients outside the pairs; ETT-absent images; one image per patient. Stratum "line" =
    n_other_lines >= 1 and "noline" = n_other_lines == 0. When fewer than
    constants.H1_SUBSTITUTE_MIN_PATIENTS patients outside the pairs have an ETT-absent no-line
    image (or when ``h1_substitute`` forces it), the H1 contrast becomes "ge2" (>= 2 other lines)
    versus "eq1" (exactly 1) and ``h1_substitute`` is recorded as True; following PROTOCOL
    section 2 ("the no-line stratum is descriptive") the few no-line patients are then kept as a
    descriptive "noline" stratum unless ``keep_noline_descriptive`` is False. Strata are drawn in
    turn, "noline" first, then the stratum with fewer eligible patients; each draws from its
    eligible patients not already selected by an earlier stratum, so the scarce stratum is as
    large as possible and a patient eligible for two strata stays available to the later one.
    Each stratum takes min(n_u_per_stratum, available patients); a short stratum takes all its
    patients and is recorded. View matching: the AP counts of the two contrast strata give the
    closest AP shares (equal AP and PA counts when the sizes are equal), then the closest to the
    B arm's AP share times the stratum size (rounded half up), limited by availability (planned
    jointly when the first contrast stratum is drawn, then matched to its realised share when
    the second is drawn); a descriptive stratum gets the feasible AP count closest to its own
    target.

    DEV. Patients outside the pairs and U; one image per patient; n_dev_pos ETT-present ("pos")
    and n_dev_neg ETT-absent ("neg") images; a shortfall takes the most items available, as
    balanced as possible, and is recorded.

    All patient sets are disjoint by nih_patient_id. Random draws use the purpose seeds
    "pairs", "u" and "dev" (``constants.derive_seed`` under the default master seed).

    Args:
        master: S4 frame (``build_master_table``); its row order does not matter.
        seed: master seed (default ``constants.SEED``).
        n_pairs, n_u_per_stratum, n_dev_pos, n_dev_neg: requested counts.
        h1_substitute: None decides the H1 substitute from the data (default); True or False
            forces the decision registered from the Day-1 yield table.
        keep_noline_descriptive: keep the descriptive "noline" U stratum under the substitute.

    Returns:
        (items, report): the S5 items (``item_id`` empty and ``smoke`` 0 until
        ``assign_item_ids`` and ``mark_smoke``), rows ordered A (by pair), B (by pair), U (by
        stratum, filename), DEV ("pos" then "neg", by filename); and the selection report
        (``selection_report`` without file hashes).

    Raises:
        ValueError: an invalid master (see ``build_master_table``), a negative count, or fewer
            than constants.N_PAIRS_RUNG_MIN possible pairs.
    """
    if int(n_pairs) < 1:
        raise ValueError("n_pairs must be at least 1")
    for name, value in (("n_u_per_stratum", n_u_per_stratum), ("n_dev_pos", n_dev_pos), ("n_dev_neg", n_dev_neg)):
        if int(value) < 0:
            raise ValueError(f"{name} must be non-negative")
    m = _normalise_master(master)
    seeds = {p: _purpose_seed(seed, p) for p in ("pairs", "u", "dev")}

    pairs, rung, pair_info = _select_pairs(m, int(n_pairs), np.random.default_rng(seeds["pairs"]))
    pid = m["nih_patient_id"].to_numpy()
    view = m["view"].to_numpy()
    pair_patients = {int(pid[a]) for a, _ in pairs}
    n_ap_b = sum(1 for _, b in pairs if view[b] == "AP")

    u_selected, u_strata, u_info = _select_u(
        m, pair_patients, n_ap_b, len(pairs), int(n_u_per_stratum), np.random.default_rng(seeds["u"]),
        h1_substitute, keep_noline_descriptive,
    )
    u_patients = {int(pid[r]) for r, _ in u_selected}
    dev_selected, dev_info = _select_dev(
        m, pair_patients | u_patients, int(n_dev_pos), int(n_dev_neg), np.random.default_rng(seeds["dev"])
    )

    rec = m.to_dict("records")
    followup = m["followup"].to_numpy()
    a_rows: list[dict[str, Any]] = []
    b_rows: list[dict[str, Any]] = []
    for k, (a, b) in enumerate(pairs, start=1):
        pair_id = f"p{k:03d}"
        direction = "B_after_A" if followup[b] > followup[a] else "B_before_A"
        gap = abs(int(followup[b]) - int(followup[a]))
        a_rows.append(_item_row(rec[a], "A", pair_id, "", direction, gap, rung))
        b_rows.append(_item_row(rec[b], "B", pair_id, "", direction, gap, rung))
    u_rank = {s: i for i, s in enumerate(u_strata)}
    u_rows = [
        _item_row(rec[r], "U", "", s, "", -1, rung)
        for r, s in sorted(u_selected, key=lambda t: (u_rank[t[1]], rec[t[0]]["nih_filename"]))
    ]
    dev_rank = {"pos": 0, "neg": 1}
    dev_rows = [
        _item_row(rec[r], "DEV", "", s, "", -1, rung)
        for r, s in sorted(dev_selected, key=lambda t: (dev_rank[t[1]], rec[t[0]]["nih_filename"]))
    ]
    items = _typed_items(a_rows + b_rows + u_rows + dev_rows)

    core: dict[str, Any] = {
        "seed": int(seed),
        "seeds": seeds,
        "rung": rung,
        "h1_substitute": u_info["h1_substitute"],
        "n_eligible_patients": pair_info["n_eligible_patients"],
        "master": {
            "n_images": len(m),
            "n_patients": int(m["nih_patient_id"].nunique()),
            "n_ett_present_images": int((m["ett"] == 1).sum()),
            "n_ett_absent_images": int((m["ett"] == 0).sum()),
        },
        "pairs": pair_info,
        "u": u_info,
        "dev": dev_info,
        "params": {
            "n_pairs": int(n_pairs),
            "n_u_per_stratum": int(n_u_per_stratum),
            "n_dev_pos": int(n_dev_pos),
            "n_dev_neg": int(n_dev_neg),
            "n_pairs_rung_min": int(C.N_PAIRS_RUNG_MIN),
            "h1_substitute_override": h1_substitute,
            "keep_noline_descriptive": bool(keep_noline_descriptive),
        },
    }
    return items, selection_report(items, core)


def assign_item_ids(items: pd.DataFrame, seed: int = C.derive_seed("ids")) -> pd.DataFrame:
    """Return a copy of ``items`` with fresh opaque ``item_id`` values.

    Each id is 10 lowercase hex characters drawn from ``numpy.random.default_rng(seed)``, unique,
    never numeric-looking (see ``_random_hex_ids``) and unrelated to the arm or the row order.
    """
    _require_columns(items, ITEM_COLS, "items")
    out = _typed_items(items)
    out["item_id"] = pd.Series(
        _random_hex_ids(np.random.default_rng(seed), len(out), ITEM_ID_HEX_LEN, upper=False), dtype="str"
    )
    return out


def mark_smoke(items: pd.DataFrame, n: int = C.N_SMOKE, seed: int = C.derive_seed("smoke")) -> pd.DataFrame:
    """Return a copy of ``items`` with ``smoke`` = 1 on ``n`` random DEV items, else 0.

    The draw is balanced: n - n // 2 "pos" and n // 2 "neg" DEV items (10 / 10 for n = 20); if a
    class has too few items, all of it is taken and the other class fills the remainder. Smoke
    items come from the dev pool only, so no test-item output exists before the blind read.
    """
    _require_columns(items, ITEM_COLS, "items")
    if int(n) < 0:
        raise ValueError("n must be non-negative")
    out = _typed_items(items)
    rng = np.random.default_rng(seed)
    dev = out.loc[out["arm"] == "DEV"].sort_values("nih_filename")
    pos = dev.index[(dev["stratum"] == "pos").to_numpy()].to_numpy(dtype=np.int64)
    neg = dev.index[(dev["stratum"] == "neg").to_numpy()].to_numpy(dtype=np.int64)
    take_pos = min(int(n) - int(n) // 2, len(pos))
    take_neg = min(int(n) // 2, len(neg))
    short = int(n) - take_pos - take_neg
    extra = min(short, len(pos) - take_pos)
    take_pos, short = take_pos + extra, short - extra
    take_neg += min(short, len(neg) - take_neg)
    chosen = np.concatenate([pos[rng.permutation(len(pos))[:take_pos]], neg[rng.permutation(len(neg))[:take_neg]]])
    smoke = np.zeros(len(out), dtype=np.int64)
    smoke[chosen.astype(np.int64)] = 1
    out["smoke"] = smoke
    return out


def _require_ids(items: pd.DataFrame) -> None:
    """Raise ValueError unless every item has a unique, non-empty item_id."""
    ids = items["item_id"].astype(str)
    if bool((ids == "").any()) or bool(items["item_id"].isna().any()):
        raise ValueError("items: item_id is empty; call assign_item_ids first")
    if bool(ids.duplicated().any()):
        raise ValueError("items: item_id values must be unique")


def pod_items(items: pd.DataFrame) -> pd.DataFrame:
    """The S6 pod item list: ``item_id``, ``nih_filename``, ``prompt_set``, ``smoke``.

    ``prompt_set`` is "full" (P0, P1, P2hi) for A, B and U and "p0" for DEV; rows are sorted by
    the opaque item id, so neither the columns nor the row order carry an arm, a label, a uid or
    a patient id. Raises ValueError if ids are missing or a smoke item is not a DEV item.
    """
    _require_columns(items, ("item_id", "nih_filename", "arm", "smoke"), "items")
    _require_ids(items)
    arm = items["arm"].astype(str)
    if not bool(arm.isin(ARMS).all()):
        raise ValueError(f"items: arm must be one of {ARMS}")
    smoke = _as_int(items["smoke"], "items", "smoke")
    if bool(((smoke == 1) & (arm != "DEV")).any()):
        raise ValueError("items: smoke items must come from the dev pool only")
    df = pd.DataFrame(
        {
            "item_id": items["item_id"].astype(str).to_numpy(),
            "nih_filename": items["nih_filename"].astype(str).to_numpy(),
            "prompt_set": arm.map(PROMPT_SET_BY_ARM).to_numpy(),
            "smoke": smoke.to_numpy(),
        }
    )
    return df.sort_values("item_id").reset_index(drop=True).loc[:, list(POD_ITEM_COLS)]


def nih_filelist(items: pd.DataFrame) -> pd.DataFrame:
    """The single-column ``nih_filename`` list of every selected image, sorted (for the export kernel)."""
    _require_columns(items, ("nih_filename",), "items")
    names = items["nih_filename"].astype(str)
    if bool(names.duplicated().any()):
        raise ValueError("items: NIH filenames must be unique")
    return pd.DataFrame({"nih_filename": sorted(names.tolist())}, columns=list(NIH_FILELIST_COLS))


def reading_lists(
    items: pd.DataFrame,
    image_dir: str | Path,
    n_a: int = C.N_READ_A,
    seed: int = C.derive_seed("reading"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The blind reading list and its key (S10).

    The list holds every B, every U and min(``n_a``, number of A) random A items in one random
    order, each under a fresh opaque ``read_id`` (8 uppercase hex characters, unique, unrelated to
    the item id and to the order). The default ``constants.N_READ_A`` reads every A image
    (PROTOCOL v0.4: RANZCR CLiP labels tracheostomy tubes as ETT, so each A image is checked). ``image_path`` is ``image_dir/nih_filename`` with forward slashes. The key maps
    ``read_id`` to ``item_id`` in the same row order and is never shown by the reading tool.

    Returns:
        (reading_list with ``READING_LIST_COLS``, reading_key with ``READING_KEY_COLS``).
    """
    _require_columns(items, ("item_id", "nih_filename", "arm"), "items")
    _require_ids(items)
    if int(n_a) < 0:
        raise ValueError("n_a must be non-negative")
    rng = np.random.default_rng(seed)
    base = items.loc[:, ["item_id", "nih_filename", "arm"]].astype(str).sort_values("nih_filename")
    a_items = base.loc[base["arm"] == "A"]
    pick = rng.permutation(len(a_items))[: min(int(n_a), len(a_items))]
    chosen = pd.concat([base.loc[base["arm"].isin(["B", "U"])], a_items.iloc[np.sort(pick)]])
    chosen = chosen.sort_values("nih_filename").reset_index(drop=True)
    chosen = chosen.iloc[rng.permutation(len(chosen))].reset_index(drop=True)
    read_ids = _random_hex_ids(rng, len(chosen), READ_ID_HEX_LEN, upper=True)
    reading = pd.DataFrame(
        {
            "read_id": read_ids,
            "nih_filename": chosen["nih_filename"].tolist(),
            "image_path": [_image_path(image_dir, f) for f in chosen["nih_filename"]],
        },
        columns=list(READING_LIST_COLS),
    )
    key = pd.DataFrame({"read_id": read_ids, "item_id": chosen["item_id"].tolist()}, columns=list(READING_KEY_COLS))
    return reading, key


def retest_list(reading_list: pd.DataFrame, n: int = C.N_RETEST, seed: int = C.derive_seed("retest")) -> pd.DataFrame:
    """``n`` random reading-list rows in a new random order, with their read ids unchanged.

    The draw starts from the rows sorted by ``read_id``, so it does not depend on the reading
    order; the subset and its order come from one seeded permutation.
    """
    _require_columns(reading_list, READING_LIST_COLS, "reading_list")
    if int(n) < 0:
        raise ValueError("n must be non-negative")
    rng = np.random.default_rng(seed)
    base = reading_list.loc[:, list(READING_LIST_COLS)].sort_values("read_id").reset_index(drop=True)
    pick = rng.permutation(len(base))[: min(int(n), len(base))]
    return base.iloc[pick].reset_index(drop=True)


def practice_seed(seed: int = C.SEED) -> int:
    """Seed of the practice list under the master ``seed`` (master x 100 + ``PRACTICE_SEED_INDEX``)."""
    if int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    return int(seed) * 100 + PRACTICE_SEED_INDEX


def practice_list(
    items: pd.DataFrame,
    image_dir: str | Path,
    n: int = PRACTICE_N,
    seed: int | None = None,
    avoid_read_ids: Sequence[str] = (),
) -> pd.DataFrame:
    """A practice reading list of ``n`` dev-pool images (S10 columns) for the student's practice run.

    Only DEV items are used, so a practice run never shows a test image (A, B or U) before the
    blind read; the draw is balanced (n // 2 ETT-present "pos", the rest "neg"; a short class is
    filled from the other). Each row gets its own fresh ``read_id`` (8 uppercase hex characters),
    never one of ``avoid_read_ids`` (the study reading list), in one random order. No key is
    written: the practice answers are not part of the study.
    """
    _require_columns(items, ("item_id", "nih_filename", "arm", "stratum"), "items")
    if int(n) < 0:
        raise ValueError("n must be non-negative")
    rng = np.random.default_rng(practice_seed() if seed is None else int(seed))
    dev = items.loc[items["arm"].astype(str) == "DEV", ["nih_filename", "stratum"]].astype(str)
    dev = dev.sort_values("nih_filename").reset_index(drop=True)
    pos = dev.index[(dev["stratum"] == "pos").to_numpy()].to_numpy(dtype=np.int64)
    neg = dev.index[(dev["stratum"] == "neg").to_numpy()].to_numpy(dtype=np.int64)
    take_pos = min(int(n) // 2, len(pos))
    take_neg = min(int(n) - take_pos, len(neg))
    take_pos = min(int(n) - take_neg, len(pos))
    chosen = np.concatenate([pos[rng.permutation(len(pos))[:take_pos]], neg[rng.permutation(len(neg))[:take_neg]]])
    chosen = np.sort(chosen)[rng.permutation(len(chosen))]
    names = dev["nih_filename"].to_numpy()[chosen].tolist() if len(chosen) else []
    avoid = {str(r) for r in avoid_read_ids}
    read_ids: list[str] = []
    while len(read_ids) < len(names):
        for rid in _random_hex_ids(rng, len(names) - len(read_ids), READ_ID_HEX_LEN, upper=True):
            if rid not in avoid and rid not in read_ids:
                read_ids.append(rid)
    return pd.DataFrame(
        {"read_id": read_ids, "nih_filename": names, "image_path": [_image_path(image_dir, f) for f in names]},
        columns=list(READING_LIST_COLS),
    )


# ----------------------------------------------------------------------------- report
def _view_share(df: pd.DataFrame) -> dict[str, Any]:
    n = len(df)
    ap = int((df["view"] == "AP").sum())
    return {"n": n, "AP": ap, "PA": n - ap, "ap_share": round(ap / n, 6) if n else None}


def _item_summaries(items: pd.DataFrame) -> dict[str, Any]:
    """Counts, view shares, follow-up gap summary, direction counts and smoke counts from S5 items."""
    arm = items["arm"].astype(str)
    stratum = items["stratum"].astype(str)
    by_arm = {a: int((arm == a).sum()) for a in ARMS}
    by_stratum: dict[str, dict[str, int]] = {}
    shares_by_stratum: dict[str, dict[str, Any]] = {}
    for a in ("U", "DEV"):
        sub = items.loc[arm == a]
        names = [s for s in (U_STRATA if a == "U" else DEV_STRATA) if bool((sub["stratum"] == s).any())]
        by_stratum[a] = {s: int((sub["stratum"] == s).sum()) for s in names}
        for s in names:
            shares_by_stratum[f"{a}:{s}"] = _view_share(sub.loc[sub["stratum"] == s])
    pair_rows = items.loc[arm.isin(["A", "B"])]
    a_rows = items.loc[arm == "A"]
    gaps = a_rows["gap"].to_numpy(dtype=float)
    if len(gaps):
        gap_summary: dict[str, Any] = {
            "n": int(len(gaps)),
            "min": int(gaps.min()),
            "q1": round(float(np.percentile(gaps, 25)), 6),
            "median": round(float(np.median(gaps)), 6),
            "mean": round(float(gaps.mean()), 6),
            "q3": round(float(np.percentile(gaps, 75)), 6),
            "max": int(gaps.max()),
            "n_gap_1": int((gaps == 1).sum()),
        }
    else:
        gap_summary = {"n": 0}
    dev = items.loc[arm == "DEV"]
    smoke = items["smoke"].to_numpy()
    return {
        "counts": {
            "by_arm": by_arm,
            "by_stratum": by_stratum,
            "n_items": int(len(items)),
            "n_test_items": by_arm["A"] + by_arm["B"] + by_arm["U"],
            "n_patients": int(items["nih_patient_id"].nunique()),
            "n_patients_by_set": {
                "pairs": int(pair_rows["nih_patient_id"].nunique()),
                "U": int(items.loc[arm == "U", "nih_patient_id"].nunique()),
                "DEV": int(dev["nih_patient_id"].nunique()),
            },
        },
        "view_shares": {"by_arm": {a: _view_share(items.loc[arm == a]) for a in ARMS}, "by_stratum": shares_by_stratum},
        "followup_gap": gap_summary,
        "direction_counts": {d: int((a_rows["direction"] == d).sum()) for d in DIRECTIONS},
        "smoke": {
            "n": int(smoke.sum()),
            "pos": int(((dev["stratum"] == "pos") & (dev["smoke"] == 1)).sum()),
            "neg": int(((dev["stratum"] == "neg") & (dev["smoke"] == 1)).sum()),
        },
    }


def selection_report(
    items: pd.DataFrame,
    core_report: Mapping[str, Any],
    file_sha256: Mapping[str, str] | None = None,
    reading_list: pd.DataFrame | None = None,
    retest: pd.DataFrame | None = None,
    practice: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """The selection report: ``core_report`` plus summaries recomputed from ``items``.

    Top-level keys: ``counts`` (per arm and per U / DEV stratum), ``rung``, ``h1_substitute``,
    ``view_shares`` (AP / PA counts and AP share per arm and per stratum), ``followup_gap``
    (n, min, q1, median, mean, q3, max, n_gap_1 over pairs), ``direction_counts``,
    ``n_eligible_patients``, ``sha256`` (file name -> hex digest of every written file except the
    report itself), ``smoke``, ``reading`` (list sizes, when given) and the detail sections
    ``pairs``, ``u``, ``dev``, ``master``, ``params``, ``seed`` and ``seeds``. Aggregates only:
    no uid, filename or label of any single image.
    """
    report: dict[str, Any] = json.loads(json.dumps(_jsonable(dict(core_report))))
    report.update(_jsonable(_item_summaries(_typed_items(items))))
    if reading_list is not None:
        report["reading"] = {
            "n_reading_list": int(len(reading_list)),
            "n_retest": None if retest is None else int(len(retest)),
        }
        if practice is not None:
            report["reading"]["n_practice"] = int(len(practice))
    report["sha256"] = {k: str(file_sha256[k]) for k in sorted(file_sha256)} if file_sha256 else {}
    return report


# ----------------------------------------------------------------------------- excluded filenames
_NIH_NAME_RE = re.compile(r"^\d{8}_\d{3}\.png$")


def read_filename_list(path: str | Path) -> tuple[list[str], str]:
    """NIH filenames listed in a text file (one per line; "#" comments and blank lines ignored).

    Returns (unique names in file order, SHA-256 of the file bytes). Raises ValueError for a line
    that is not an NIH filename ("00000001_000.png").
    """
    data = Path(path).read_bytes()
    names: list[str] = []
    for k, raw in enumerate(data.decode("utf-8-sig").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if not _NIH_NAME_RE.fullmatch(line):
            raise ValueError(f"{Path(path).name} line {k}: {line!r} is not an NIH filename")
        if line not in names:
            names.append(line)
    return names, _sha256_bytes(data)


def remove_filenames(master: pd.DataFrame, names: Sequence[str]) -> tuple[pd.DataFrame, dict[str, int]]:
    """``master`` without the rows whose ``nih_filename`` is listed, and the counts (listed, present)."""
    _require_columns(master, ("nih_filename",), "master")
    listed = set(str(n) for n in names)
    present = master["nih_filename"].astype(str).isin(listed)
    return master.loc[~present].reset_index(drop=True), {"n_listed": len(listed), "n_present_in_matched_pool": int(present.sum())}


# ----------------------------------------------------------------------------- end-to-end build and write
@dataclass(frozen=True)
class SelectionOutputs:
    """Every table ``scripts/select_items.py`` writes, plus the report from ``select_core``."""

    items: pd.DataFrame
    pod_items: pd.DataFrame
    nih_filelist: pd.DataFrame
    reading_list: pd.DataFrame
    reading_key: pd.DataFrame
    retest_list: pd.DataFrame
    core_report: dict[str, Any]
    practice_list: pd.DataFrame | None = None


def build_outputs(
    master: pd.DataFrame,
    image_dir: str | Path,
    seed: int = C.SEED,
    n_pairs: int = C.N_PAIRS,
    n_u_per_stratum: int = C.N_U_PER_STRATUM,
    n_dev_pos: int = C.N_DEV_POS,
    n_dev_neg: int = C.N_DEV_NEG,
    n_read_a: int = C.N_READ_A,
    n_retest: int = C.N_RETEST,
    n_smoke: int = C.N_SMOKE,
    *,
    h1_substitute: bool | None = None,
    keep_noline_descriptive: bool = True,
    n_practice: int = PRACTICE_N,
    exclude_filenames: Sequence[str] | None = None,
    exclude_source: Mapping[str, Any] | None = None,
) -> SelectionOutputs:
    """Run the whole selection: ``select_core``, ids, smoke flags and every derived list.

    Purpose seeds come from the master ``seed`` exactly as ``constants.derive_seed`` derives
    them ("ids", "smoke", "reading", "retest" here; "pairs", "u", "dev" inside ``select_core``);
    the practice list uses ``practice_seed(seed)``.

    ``exclude_filenames`` (for example the public NIH images that received model or detector
    outputs during software tests before the protocol freeze) are removed from the matched pool
    before ANY selection, so no arm, stratum or the dev pool can use them. The report records
    ``excluded_filenames`` = {n_listed, n_present_in_matched_pool} plus ``exclude_source`` (the
    list file's name and SHA-256); never the names themselves.
    """
    excluded_info: dict[str, Any] | None = None
    if exclude_filenames is not None:
        master, counts = remove_filenames(master, exclude_filenames)
        excluded_info = {**dict(exclude_source or {}), **counts}
    items, core = select_core(
        master, seed, n_pairs, n_u_per_stratum, n_dev_pos, n_dev_neg,
        h1_substitute=h1_substitute, keep_noline_descriptive=keep_noline_descriptive,
    )
    if excluded_info is not None:
        listed = set(str(n) for n in exclude_filenames or ())
        if listed & set(items["nih_filename"].astype(str)):  # pragma: no cover - guaranteed by the removal
            raise RuntimeError("internal error: an excluded NIH filename was selected")
        core = {**core, "excluded_filenames": excluded_info}
    extra_seeds = {p: _purpose_seed(seed, p) for p in ("ids", "smoke", "reading", "retest")}
    items = assign_item_ids(items, extra_seeds["ids"])
    items = mark_smoke(items, n_smoke, extra_seeds["smoke"])
    reading, key = reading_lists(items, image_dir, n_read_a, extra_seeds["reading"])
    retest = retest_list(reading, n_retest, extra_seeds["retest"])
    practice = practice_list(items, image_dir, n_practice, practice_seed(seed), reading["read_id"].tolist())
    core = dict(core)
    core["seeds"] = {**core["seeds"], **extra_seeds}
    core["params"] = {**core["params"], "n_read_a": int(n_read_a), "n_retest": int(n_retest), "n_smoke": int(n_smoke),
                      "n_practice": int(n_practice), "seed_practice": practice_seed(seed)}
    return SelectionOutputs(items, pod_items(items), nih_filelist(items), reading, key, retest, core, practice)


def write_outputs(outputs: SelectionOutputs, out_dir: str | Path, *, overwrite: bool = False) -> dict[str, Any]:
    """Write the CSV files, the list seed sidecars and ``selection_report.json`` into ``out_dir``.

    CSV files (``OUTPUT_FILES``; the practice list when the outputs carry one) are UTF-8 with
    ``\\n`` line endings and no index; the report is sorted-key JSON and records the SHA-256 of
    every CSV written. The sidecars ``reading_list.meta.json`` = {"seed": derive_seed("reading")},
    ``retest_list.meta.json`` = {"seed": derive_seed("retest")} and ``practice_list.meta.json`` =
    {"seed": practice seed} (under the master seed of the selection) are where the reading tool
    reads the list seed from. Identical outputs give byte-identical files.

    Raises:
        FileExistsError: an output file exists and ``overwrite`` is False.
        ValueError: a table has the wrong columns, or a RANZCR uid would reach the pod list.

    Returns:
        The selection report that was written.
    """
    out = Path(out_dir)
    tables = {
        "items": outputs.items,
        "pod_items": outputs.pod_items,
        "nih_filelist": outputs.nih_filelist,
        "reading_list": outputs.reading_list,
        "reading_key": outputs.reading_key,
        "retest_list": outputs.retest_list,
    }
    if outputs.practice_list is not None:
        tables["practice_list"] = outputs.practice_list
        test_names = set(outputs.items.loc[outputs.items["arm"].astype(str) != "DEV", "nih_filename"].astype(str))
        if test_names & set(outputs.practice_list["nih_filename"].astype(str)):
            raise ValueError("the practice list would show a test image (A, B or U); refusing to write")
    uids = set(outputs.items["uid"].astype(str))
    pod_text = set(outputs.pod_items.astype(str).to_numpy().ravel().tolist())
    if uids & pod_text:
        raise ValueError("pod_items would carry a RANZCR uid; refusing to write")
    payloads = {OUTPUT_FILES[name]: _csv_bytes(df, _TABLE_COLS[name], name) for name, df in tables.items()}
    seeds = dict(outputs.core_report.get("seeds", {}))
    params = dict(outputs.core_report.get("params", {}))
    sidecars: dict[str, bytes] = {}
    for name, seed_value in (("reading_list", seeds.get("reading")), ("retest_list", seeds.get("retest")),
                             ("practice_list", params.get("seed_practice"))):
        if name in tables and seed_value is not None:
            sidecars[META_FILES[name]] = _json_bytes({"seed": int(seed_value)})
    targets = [out / name for name in (*payloads, *sidecars, OUTPUT_FILES["selection_report"])]
    existing = [p.name for p in targets if p.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"{out}: refusing to overwrite {', '.join(existing)} (use overwrite=True / --force)")
    out.mkdir(parents=True, exist_ok=True)
    sha: dict[str, str] = {}
    for name, data in payloads.items():
        (out / name).write_bytes(data)
        sha[name] = _sha256_bytes(data)
    for name, data in sidecars.items():
        (out / name).write_bytes(data)
    report = selection_report(outputs.items, outputs.core_report, sha, outputs.reading_list, outputs.retest_list,
                              outputs.practice_list)
    (out / OUTPUT_FILES["selection_report"]).write_bytes(_json_bytes(report))
    return report


def read_items(path: str | Path) -> pd.DataFrame:
    """Read ``items.csv`` back with its S5 types (empty text fields stay "", ids keep leading zeros)."""
    df = pd.read_csv(
        path,
        dtype={c: str for c in _ITEM_STR_COLS},
        keep_default_na=False,
        na_values=[],
    )
    _require_columns(df, ITEM_COLS, str(path))
    return _typed_items(df)
