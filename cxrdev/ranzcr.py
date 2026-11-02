"""RANZCR CLiP label presence per image and the Day-1 yield table (kernel-side).

Labels of record: RANZCR CLiP labels (43 trained labellers: radiologists, radiology trainees,
hospital medical officers), Kaggle competition ``ranzcr-clip-catheter-line-classification``
(``train.csv``). A device class is present when any of its columns is 1 (PROTOCOL section 1);
"NGT - Incompletely Imaged" counts as NGT present and is flagged separately.

Data boundary (PROTOCOL section 9): ``train.csv`` and every row-level frame derived from it stay
on Kaggle and in ``data/`` on the student's machine. ``yield_table`` returns aggregate counts only.

Runs inside the private Kaggle kernels and locally (pandas, Python 3.10 syntax). Private helpers
carry the ``_rz_`` prefix.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

from cxrdev.constants import CVC_COLS, ETT_COLS, NGT_COLS, NGT_INCOMPLETE_COL, RANZCR_PATIENT, RANZCR_UID, SWAN_COL

PRESENCE_COLUMNS = ("uid", "ranzcr_patient_id", "ett", "ngt", "ngt_incomplete", "cvc", "swan", "n_other_lines")

#: metrics guaranteed in ``yield_table`` (PIPELINE_SPEC section 3), followed by extra descriptive rows
YIELD_METRICS = (
    "n_images", "n_patients", "ett_present_images", "ett_absent_images", "patients_with_both_ett_states",
    "ett_absent_line_images", "ett_absent_line_patients", "ett_absent_noline_images",
    "ett_absent_noline_patients", "ett_absent_ge2_patients", "ett_absent_eq1_patients",
    "ngt_incomplete_images",
)


def _rz_required_columns() -> list[str]:
    cols = [RANZCR_UID, RANZCR_PATIENT, *ETT_COLS, *NGT_COLS, *CVC_COLS, SWAN_COL]
    out: list[str] = []
    for c in cols:
        if c not in out:
            out.append(c)
    return out


def _rz_flag(df: pd.DataFrame, cols) -> np.ndarray:
    """1 where any of ``cols`` is >= 0.5 (labels are 0/1; blanks and text count as 0), else 0."""
    block = np.zeros((len(df), len(cols)), dtype=np.float64)
    for j, c in enumerate(cols):
        block[:, j] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    return (block >= 0.5).any(axis=1).astype(np.int64)


def presence(train_df: pd.DataFrame) -> pd.DataFrame:
    """Schema S1: one row per RANZCR image with 0/1 device-class presence and ``n_other_lines``.

    Columns: uid, ranzcr_patient_id (str), ett, ngt, ngt_incomplete, cvc, swan (int 0/1),
    n_other_lines (= ngt + cvc + swan). Raises ValueError on missing columns or duplicated UIDs.
    """
    missing = [c for c in _rz_required_columns() if c not in train_df.columns]
    if missing:
        raise ValueError(f"train.csv lacks columns {missing}")
    uid = train_df[RANZCR_UID].astype(str).str.strip()
    if uid.duplicated().any():
        raise ValueError(f"train.csv has {int(uid.duplicated().sum())} duplicated {RANZCR_UID} values")
    if (uid == "").any():
        raise ValueError(f"train.csv has empty {RANZCR_UID} values")
    out = pd.DataFrame({
        "uid": uid.to_numpy(),
        "ranzcr_patient_id": train_df[RANZCR_PATIENT].astype(str).str.strip().to_numpy(),
        "ett": _rz_flag(train_df, ETT_COLS),
        "ngt": _rz_flag(train_df, NGT_COLS),
        "ngt_incomplete": _rz_flag(train_df, (NGT_INCOMPLETE_COL,)),
        "cvc": _rz_flag(train_df, CVC_COLS),
        "swan": _rz_flag(train_df, (SWAN_COL,)),
    })
    out["n_other_lines"] = (out["ngt"] + out["cvc"] + out["swan"]).astype("int64")
    return out.loc[:, list(PRESENCE_COLUMNS)].reset_index(drop=True)


def _rz_patients(df: pd.DataFrame, mask: pd.Series) -> set:
    return set(df.loc[mask, "ranzcr_patient_id"].tolist())


def yield_table(presence_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate yield counts from an S1 frame, as a two-column frame (metric, value).

    The first rows are ``YIELD_METRICS`` (images and patients by ETT state, the U strata by other
    labelled lines: line = at least one of NGT, CVC, Swan-Ganz; noline = none; ge2 / eq1 = two or more /
    exactly one), followed by descriptive rows used to choose a rung (patients with at least two
    ETT-present and at least two ETT-absent images) and the per-class image counts.
    """
    missing = [c for c in PRESENCE_COLUMNS if c not in presence_df.columns]
    if missing:
        raise ValueError(f"presence frame lacks columns {missing}")
    p = presence_df
    ett = p["ett"].astype("int64") == 1
    absent = ~ett
    other = p["n_other_lines"].astype("int64")
    present_patients = _rz_patients(p, ett)
    absent_patients = _rz_patients(p, absent)
    both = present_patients & absent_patients
    line = absent & (other >= 1)
    noline = absent & (other == 0)
    per_patient = p.assign(_present=ett.astype("int64"), _absent=absent.astype("int64")).groupby(
        "ranzcr_patient_id")[["_present", "_absent"]].sum()
    ge2_both = int(((per_patient["_present"] >= 2) & (per_patient["_absent"] >= 2)).sum())
    noline_patients = _rz_patients(p, noline)
    line_patients = _rz_patients(p, line)
    rows = [
        ("n_images", len(p)),
        ("n_patients", int(p["ranzcr_patient_id"].nunique())),
        ("ett_present_images", int(ett.sum())),
        ("ett_absent_images", int(absent.sum())),
        ("patients_with_both_ett_states", len(both)),
        ("ett_absent_line_images", int(line.sum())),
        ("ett_absent_line_patients", len(line_patients)),
        ("ett_absent_noline_images", int(noline.sum())),
        ("ett_absent_noline_patients", len(noline_patients)),
        ("ett_absent_ge2_patients", len(_rz_patients(p, absent & (other >= 2)))),
        ("ett_absent_eq1_patients", len(_rz_patients(p, absent & (other == 1)))),
        ("ngt_incomplete_images", int((p["ngt_incomplete"].astype("int64") == 1).sum())),
        ("ett_present_patients", len(present_patients)),
        ("ett_absent_patients", len(absent_patients)),
        ("patients_ge2_present_ge2_absent", ge2_both),
        ("ett_absent_line_patients_not_both_states", len(line_patients - both)),
        ("ett_absent_noline_patients_not_both_states", len(noline_patients - both)),
        ("ngt_present_images", int((p["ngt"].astype("int64") == 1).sum())),
        ("cvc_present_images", int((p["cvc"].astype("int64") == 1).sum())),
        ("swan_present_images", int((p["swan"].astype("int64") == 1).sum())),
    ]
    out = pd.DataFrame(rows, columns=["metric", "value"])
    out["value"] = out["value"].astype("int64")
    return out


def is_train_csv(path: str | Path) -> bool:
    """True when ``path`` is a CSV whose header holds the RANZCR UID and PatientID columns."""
    try:
        with open(path, newline="", encoding="utf-8-sig") as fh:
            header = next(csv.reader(fh), [])
    except (OSError, UnicodeDecodeError):
        return False
    names = {h.strip() for h in header}
    return RANZCR_UID in names and RANZCR_PATIENT in names


def read_train_csv(path: str | Path) -> pd.DataFrame:
    """Read ``train.csv`` with the UID and PatientID columns as strings."""
    return pd.read_csv(path, dtype={RANZCR_UID: str, RANZCR_PATIENT: str}, keep_default_na=False)
