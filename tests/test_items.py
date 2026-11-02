"""Tests for cxrdev/items.py and scripts/select_items.py.

Synthetic data only: every frame below is generated here (no RANZCR or NIH row is used). The
helpers build S4 master frames directly, or S1/S2/S3 frames whose matched join is a known master.
"""
from __future__ import annotations

import dataclasses
import hashlib
import importlib
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cxrdev import constants as C
from cxrdev import items as I

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


# ============================================================================ synthetic data helpers
def img(pid: int, fu: int, ett: int, lines: int = 1, view: str = "AP", *, ngt_incomplete: int = 0,
        age: int = 60, sex: str = "F") -> dict:
    """One synthetic S4 row; ``lines`` other lines are set as cvc, then ngt, then swan."""
    cvc, ngt, swan = int(lines >= 1), int(lines >= 2), int(lines >= 3)
    return {
        "uid": f"synuid-{pid}-{fu}",
        "ranzcr_patient_id": f"synrp{pid:06d}",
        "nih_filename": f"{pid:08d}_{fu:03d}.png",
        "nih_patient_id": pid,
        "followup": fu,
        "view": view,
        "age": age,
        "sex": sex,
        "ett": ett,
        "ngt": ngt,
        "ngt_incomplete": ngt_incomplete if ngt else 0,
        "cvc": cvc,
        "swan": swan,
        "n_other_lines": lines,
    }


def frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=list(I.MASTER_COLS))


def random_master(n_patients: int = 1500, seed: int = 11) -> pd.DataFrame:
    """1-6 studies per patient; 35% of patients mixed ETT states, 50% always absent, 15% always present."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for pid in range(1, n_patients + 1):
        k = int(rng.integers(1, 7))
        fus = sorted(rng.choice(40, size=k, replace=False).tolist())
        u = rng.random()
        if u < 0.35:
            states = rng.integers(0, 2, size=k).tolist()
            if k >= 2 and len(set(states)) == 1:
                states[0] = 1 - states[0]
        elif u < 0.85:
            states = [0] * k
        else:
            states = [1] * k
        age, sex = int(rng.integers(18, 90)), ("M" if rng.random() < 0.55 else "F")
        for fu, ett in zip(fus, states):
            lines = int(rng.choice(4, p=[0.25, 0.40, 0.25, 0.10]))
            view = "AP" if rng.random() < 0.65 else "PA"
            rows.append(img(pid, fu, ett, lines, view, ngt_incomplete=int(rng.random() < 0.2), age=age, sex=sex))
    return frame(rows)


def eligible_master(n_eligible: int, n_second: int, *, start_pid: int = 1) -> pd.DataFrame:
    """``n_eligible`` patients with both ETT states; the first ``n_second`` have two of each."""
    rows: list[dict] = []
    for k in range(n_eligible):
        pid = start_pid + k
        if k < n_second:
            rows += [img(pid, 0, 1), img(pid, 1, 0), img(pid, 2, 1), img(pid, 3, 0)]
        else:
            rows += [img(pid, 0, 1), img(pid, 1, 0)]
    return frame(rows)


def h1_master(n_noline: int, *, n_eligible: int = 130, n_eq1: int = 100, n_ge2: int = 100) -> pd.DataFrame:
    """Pair patients whose ETT-absent images carry a line, plus single-image U candidates."""
    rows: list[dict] = []
    pid = 0
    for _ in range(n_eligible):
        pid += 1
        rows += [img(pid, 0, 1, 1), img(pid, 1, 0, 1)]
    for _ in range(n_noline):
        pid += 1
        rows.append(img(pid, 0, 0, 0, "AP" if pid % 2 else "PA"))
    for _ in range(n_eq1):
        pid += 1
        rows.append(img(pid, 0, 0, 1, "AP" if pid % 3 else "PA"))
    for _ in range(n_ge2):
        pid += 1
        rows.append(img(pid, 0, 0, 2 + pid % 2, "AP" if pid % 3 else "PA"))
    return frame(rows)


def view_master(noline_ap: int, noline_pa: int, *, noline_both: int = 0, line_ap: int = 100, line_pa: int = 100,
                b_ap: int = 90, b_pa: int = 30) -> pd.DataFrame:
    """Exactly b_ap + b_pa pair patients (B view fixed) plus single-view or two-view U candidates."""
    rows: list[dict] = []
    pid = 0
    for k in range(b_ap + b_pa):
        pid += 1
        rows += [img(pid, 0, 1, 1, "AP"), img(pid, 1, 0, 1, "AP" if k < b_ap else "PA")]
    for _ in range(noline_ap):
        pid += 1
        rows.append(img(pid, 0, 0, 0, "AP"))
    for _ in range(noline_pa):
        pid += 1
        rows.append(img(pid, 0, 0, 0, "PA"))
    for _ in range(noline_both):
        pid += 1
        rows += [img(pid, 0, 0, 0, "AP"), img(pid, 1, 0, 0, "PA")]
    for _ in range(line_ap):
        pid += 1
        rows.append(img(pid, 0, 0, 1, "AP"))
    for _ in range(line_pa):
        pid += 1
        rows.append(img(pid, 0, 0, 2, "PA"))
    return frame(rows)


def dev_master(pos_only: int, neg_only: int, extra_eligible: int) -> pd.DataFrame:
    """120 + extra_eligible pair-eligible patients, plus single-state patients for the dev pool."""
    rows = eligible_master(120 + extra_eligible, 0).to_dict("records")
    pid = 10_000
    for _ in range(pos_only):
        pid += 1
        rows.append(img(pid, 0, 1))
    for _ in range(neg_only):
        pid += 1
        rows.append(img(pid, 0, 0))
    return frame(rows)


def split_master(master: pd.DataFrame, seed: int = 3) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Synthetic S3 match table, S1 presence and S2 nih_meta whose matched join is ``master``."""
    rng = np.random.default_rng(seed)
    label_cols = ["uid", "ranzcr_patient_id", *I.LABEL_COLS, "n_other_lines"]
    extra_pres = pd.DataFrame(
        {
            "uid": [f"synuid-x-{i}" for i in range(40)],
            "ranzcr_patient_id": [f"synrpx{i % 15}" for i in range(40)],
            "ett": rng.integers(0, 2, 40), "ngt": 0, "ngt_incomplete": 0, "cvc": 1, "swan": 0, "n_other_lines": 1,
        }
    )
    presence = pd.concat([master[label_cols], extra_pres], ignore_index=True)
    n = len(master)
    matched = pd.DataFrame(
        {
            "uid": master["uid"], "ranzcr_patient_id": master["ranzcr_patient_id"],
            "jpg_w": 2048, "jpg_h": 2048, "nih_filename": master["nih_filename"],
            "nih_patient_id": master["nih_patient_id"], "d1": rng.integers(0, 10, n), "d2": rng.integers(40, 90, n),
            "ncc1": 0.99, "ncc2": 0.5, "n_candidates": 3, "key_used": "dim", "route": "margin", "status": "matched",
        }
    )
    statuses = ["unmatched", "ambiguous", "duplicate_conflict", "patient_conflict"]
    other = pd.DataFrame(
        {
            "uid": extra_pres["uid"], "ranzcr_patient_id": extra_pres["ranzcr_patient_id"],
            "jpg_w": 2500, "jpg_h": 2048,
            # a duplicate_conflict row may name a filename that another uid holds; it must be ignored
            "nih_filename": [master["nih_filename"].iloc[i] if statuses[i % 4] == "duplicate_conflict" else ""
                             for i in range(40)],
            "nih_patient_id": [int(master["nih_patient_id"].iloc[i]) if statuses[i % 4] == "duplicate_conflict" else -1
                               for i in range(40)],
            "d1": 80, "d2": -1, "ncc1": np.nan, "ncc2": np.nan, "n_candidates": 1, "key_used": "global",
            "route": "none", "status": [statuses[i % 4] for i in range(40)],
        }
    )
    match_table = pd.concat([matched, other], ignore_index=True)
    match_table = match_table.iloc[rng.permutation(len(match_table))].reset_index(drop=True)
    meta = master[["nih_filename", "nih_patient_id", "followup", "age", "sex", "view"]].copy()
    extra_meta = pd.DataFrame(
        {
            "nih_filename": [f"{900000 + i:08d}_000.png" for i in range(25)],
            "nih_patient_id": [900000 + i for i in range(25)], "followup": 0, "age": 40, "sex": "M", "view": "PA",
        }
    )
    nih_meta = pd.concat([meta, extra_meta], ignore_index=True)
    nih_meta["orig_w"], nih_meta["orig_h"], nih_meta["finding_labels"] = 2048, 2048, "No Finding"
    return match_table, presence, nih_meta


# ============================================================================ check helpers
def n_runs(seq: list) -> int:
    return 1 + sum(1 for x, y in zip(seq, seq[1:]) if x != y)


def expected_runs(counts: list[int]) -> float:
    n = sum(counts)
    return 1 + (n - 1) - sum(c * (c - 1) for c in counts) / n


def rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def assert_core_invariants(items: pd.DataFrame, master: pd.DataFrame) -> None:
    """Every rule of select_core that can be checked from its output and the master table."""
    assert list(items.columns) == list(I.ITEM_COLS)
    assert set(items["arm"]) <= set(I.ARMS)
    assert items["nih_filename"].is_unique
    assert items["rung"].nunique() == 1 and items["rung"].iloc[0] in I.RUNGS
    # every row is a faithful copy of its master row
    m = master.set_index("nih_filename")
    for col in ("uid", "nih_patient_id", "followup", "view", "age", "sex", *I.LABEL_COLS, "n_other_lines"):
        assert (items[col].to_numpy() == m.loc[items["nih_filename"], col].to_numpy()).all(), col
    a = items[items["arm"] == "A"].set_index("pair_id")
    b = items[items["arm"] == "B"].set_index("pair_id")
    assert a.index.is_unique and b.index.is_unique and set(a.index) == set(b.index)
    assert sorted(a.index) == [f"p{k:03d}" for k in range(1, len(a) + 1)]
    b = b.loc[a.index]
    assert (a["ett"] == 1).all() and (b["ett"] == 0).all()
    assert (a["nih_patient_id"].to_numpy() == b["nih_patient_id"].to_numpy()).all()
    gap = (b["followup"].to_numpy() - a["followup"].to_numpy())
    assert (a["gap"].to_numpy() == np.abs(gap)).all() and (b["gap"].to_numpy() == np.abs(gap)).all()
    want_dir = np.where(gap > 0, "B_after_A", "B_before_A")
    assert (a["direction"].to_numpy() == want_dir).all() and (b["direction"].to_numpy() == want_dir).all()
    assert (a["stratum"] == "").all() and (b["stratum"] == "").all() and (a["gap"] >= 1).all()
    # B is nearest in follow-up number: only the B of an earlier pair of the patient may be closer
    for p, grp in a.groupby("nih_patient_id"):
        earlier_b: set[str] = set()
        for pair_id in sorted(grp.index):
            fa, fb = a.at[pair_id, "followup"], b.at[pair_id, "followup"]
            absent = master[(master["nih_patient_id"] == p) & (master["ett"] == 0)]
            closer = set(absent.loc[(absent["followup"] - fa).abs() < abs(fb - fa), "nih_filename"])
            assert closer <= earlier_b, (p, pair_id)
            earlier_b.add(b.at[pair_id, "nih_filename"])
    # non-pair rows
    rest = items[items["arm"].isin(["U", "DEV"])]
    assert (rest["pair_id"] == "").all() and (rest["direction"] == "").all() and (rest["gap"] == -1).all()
    u = items[items["arm"] == "U"]
    dev = items[items["arm"] == "DEV"]
    assert (u["ett"] == 0).all() and u["nih_patient_id"].is_unique and dev["nih_patient_id"].is_unique
    lines = u["n_other_lines"]
    for s, ok in (("line", lines >= 1), ("noline", lines == 0), ("ge2", lines >= 2), ("eq1", lines == 1)):
        assert ok[u["stratum"] == s].all(), s
    u_strata = set(u["stratum"])
    assert u_strata <= {"line", "noline"} or u_strata <= {"ge2", "eq1", "noline"}, u_strata
    assert ((dev["stratum"] == "pos") == (dev["ett"] == 1)).all() and set(dev["stratum"]) <= {"pos", "neg"}
    pairs_p = set(a["nih_patient_id"])
    u_p, dev_p = set(u["nih_patient_id"]), set(dev["nih_patient_id"])
    assert not (pairs_p & u_p) and not (pairs_p & dev_p) and not (u_p & dev_p)


@pytest.fixture(scope="module")
def big_master() -> pd.DataFrame:
    return random_master()


@pytest.fixture(scope="module")
def big_outputs(big_master: pd.DataFrame) -> I.SelectionOutputs:
    return I.build_outputs(big_master, "data/images")


# ============================================================================ build_master_table
def test_build_master_table_joins_matched_rows_only() -> None:
    master = random_master(300, seed=5)
    match_table, presence, nih_meta = split_master(master)
    built = I.build_master_table(match_table, presence, nih_meta)
    assert list(built.columns) == list(I.MASTER_COLS)
    want = master.sort_values(["nih_patient_id", "followup"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(built, I._normalise_master(want), check_dtype=True)
    assert len(built) == len(master)
    for col in ("nih_patient_id", "followup", "age", *I.LABEL_COLS, "n_other_lines"):
        assert built[col].dtype == np.int64
    assert built["nih_filename"].is_unique and not built["uid"].str.startswith("synuid-x").any()


def _break(kind: str, mt: pd.DataFrame, pr: pd.DataFrame, nm: pd.DataFrame):
    matched = mt.index[mt["status"] == "matched"]
    if kind == "missing_column":
        pr = pr.drop(columns=["swan"])
    elif kind == "uid_without_labels":
        pr = pr[pr["uid"] != mt.at[matched[0], "uid"]]
    elif kind == "filename_without_meta":
        nm = nm[nm["nih_filename"] != mt.at[matched[0], "nih_filename"]]
    elif kind == "duplicate_filename":
        mt.loc[matched[1], "nih_filename"] = mt.at[matched[0], "nih_filename"]
    elif kind == "patient_id_disagrees":
        mt.loc[matched[0], "nih_patient_id"] = 777777
    elif kind == "lines_inconsistent":
        pr.loc[pr.index[0], "n_other_lines"] = (int(pr.at[pr.index[0], "n_other_lines"]) + 1) % 4
    elif kind == "bad_view":
        nm.loc[nm.index[0], "view"] = "LL"
    elif kind == "empty_filename":
        mt.loc[matched[0], "nih_filename"] = ""
    elif kind == "rpid_two_nih_patients":
        first = mt.at[matched[0], "ranzcr_patient_id"]
        other = mt.loc[matched][mt.loc[matched, "ranzcr_patient_id"] != first].index[0]
        mt.loc[other, "ranzcr_patient_id"] = first
        pr.loc[pr["uid"] == mt.at[other, "uid"], "ranzcr_patient_id"] = first
    return mt, pr, nm


@pytest.mark.parametrize(
    ("kind", "message"),
    [
        ("missing_column", "swan"),
        ("uid_without_labels", "no row in presence"),
        ("filename_without_meta", "no row in nih_meta"),
        ("duplicate_filename", "duplicated 'nih_filename'"),
        ("patient_id_disagrees", "disagree on nih_patient_id"),
        ("lines_inconsistent", "n_other_lines"),
        ("bad_view", "view must be"),
        ("empty_filename", "empty"),
        ("rpid_two_nih_patients", "more than one NIH patient"),
    ],
)
def test_build_master_table_rejects_inconsistent_inputs(kind: str, message: str) -> None:
    master = random_master(60, seed=9)
    mt, pr, nm = split_master(master)
    mt, pr, nm = _break(kind, mt.copy(), pr.copy(), nm.copy())
    with pytest.raises(ValueError, match=message):
        I.build_master_table(mt, pr, nm)


def test_master_validation_rejects_duplicate_followup_and_incomplete_without_ngt() -> None:
    rows = [img(1, 0, 1), img(1, 1, 0)]
    dup = frame(rows + [dict(img(1, 1, 0), nih_filename="00000001_999.png", uid="synuid-dup")])
    with pytest.raises(ValueError, match="follow-up numbers must be unique"):
        I.select_core(dup, n_pairs=1)
    bad = frame([img(1, 0, 1), dict(img(1, 1, 0, 1), ngt_incomplete=1)])
    with pytest.raises(ValueError, match="ngt_incomplete"):
        I.select_core(bad, n_pairs=1)


# ============================================================================ select_core: standard rung
def test_select_core_standard_rung_invariants(big_master: pd.DataFrame) -> None:
    items, report = I.select_core(big_master)
    assert_core_invariants(items, big_master)
    counts = items.groupby(["arm", "stratum"]).size().to_dict()
    assert counts == {("A", ""): 120, ("B", ""): 120, ("U", "line"): 60, ("U", "noline"): 60,
                      ("DEV", "pos"): 150, ("DEV", "neg"): 150}
    assert report["rung"] == "standard" and (items["rung"] == "standard").all()
    assert items.loc[items["arm"] == "A", "nih_patient_id"].is_unique  # one pair per patient
    assert (items["item_id"] == "").all() and (items["smoke"] == 0).all()
    assert report["h1_substitute"] is False and report["u"]["descriptive_strata"] == []
    assert report["seeds"] == {p: C.derive_seed(p) for p in ("pairs", "u", "dev")}
    # eligible-patient count recomputed independently
    g = big_master.groupby("nih_patient_id")["ett"].agg(["min", "max"])
    assert report["n_eligible_patients"] == int(((g["min"] == 0) & (g["max"] == 1)).sum())
    assert report["sha256"] == {}


def test_select_core_view_matching_follows_b_arm_share(big_master: pd.DataFrame) -> None:
    items, report = I.select_core(big_master)
    b = items[items["arm"] == "B"]
    n_ap_b = int((b["view"] == "AP").sum())
    target = (2 * n_ap_b * 60 + len(b)) // (2 * len(b))
    u = items[items["arm"] == "U"]
    ap = {s: int(((u["stratum"] == s) & (u["view"] == "AP")).sum()) for s in ("line", "noline")}
    assert ap["line"] == ap["noline"] == target
    assert report["u"]["view_matched"] is True
    assert report["u"]["target_ap_share"] == round(n_ap_b / len(b), 6)


def test_select_core_is_independent_of_row_order(big_master: pd.DataFrame) -> None:
    shuffled = big_master.sample(frac=1.0, random_state=1).reset_index(drop=True)
    shuffled = shuffled[list(reversed(shuffled.columns))]
    a, ra = I.select_core(big_master)
    b, rb = I.select_core(shuffled)
    pd.testing.assert_frame_equal(a, b)
    assert ra == rb


def test_select_core_different_seed_changes_selection(big_master: pd.DataFrame) -> None:
    a, _ = I.select_core(big_master)
    b, _ = I.select_core(big_master, seed=C.SEED + 1)
    assert set(a["nih_filename"]) != set(b["nih_filename"])


def test_a_is_drawn_at_random_and_b_is_nearest() -> None:
    # three ETT-present images, one ETT-absent image: every A occurs across seeds; B is fixed
    m = frame([img(1, 0, 1), img(1, 10, 1), img(1, 20, 1), img(1, 30, 0)])
    seen_a = set()
    for seed in range(60):
        items, _ = I.select_core(m, seed=seed, n_pairs=1, n_u_per_stratum=0, n_dev_pos=0, n_dev_neg=0)
        a = items[items["arm"] == "A"].iloc[0]
        b = items[items["arm"] == "B"].iloc[0]
        seen_a.add(int(a["followup"]))
        assert int(b["followup"]) == 30 and b["direction"] == "B_after_A" and b["gap"] == 30 - a["followup"]
    assert seen_a == {0, 10, 20}
    # nearest absent image wins over farther ones, before or after A
    m2 = frame([img(1, 10, 1), img(1, 2, 0), img(1, 9, 0), img(1, 30, 0)])
    items, _ = I.select_core(m2, n_pairs=1, n_u_per_stratum=0, n_dev_pos=0, n_dev_neg=0)
    b = items[items["arm"] == "B"].iloc[0]
    assert int(b["followup"]) == 9 and b["direction"] == "B_before_A" and b["gap"] == 1


def test_b_ties_are_broken_at_random() -> None:
    m = frame([img(1, 5, 1), img(1, 3, 0), img(1, 7, 0)])
    seen = set()
    for seed in range(40):
        items, _ = I.select_core(m, seed=seed, n_pairs=1, n_u_per_stratum=0, n_dev_pos=0, n_dev_neg=0)
        b = items[items["arm"] == "B"].iloc[0]
        seen.add((int(b["followup"]), b["direction"]))
        assert b["gap"] == 2
    assert seen == {(3, "B_before_A"), (7, "B_after_A")}


# ============================================================================ rungs
def _pairs_only(m: pd.DataFrame, **kw):
    return I.select_core(m, n_u_per_stratum=0, n_dev_pos=0, n_dev_neg=0, **kw)


@pytest.mark.parametrize(
    ("n_eligible", "n_second", "rung", "n_pairs", "n_two"),
    [
        (130, 0, "standard", 120, 0),
        (120, 0, "standard", 120, 0),
        (119, 1, "two_per_patient", 120, 1),
        (100, 25, "two_per_patient", 120, 20),
        (90, 5, "n80", 80, 0),
        (80, 0, "n80", 80, 0),
        (60, 30, "n80", 80, 20),
        (70, 10, "n80", 80, 10),
    ],
)
def test_rungs(n_eligible: int, n_second: int, rung: str, n_pairs: int, n_two: int) -> None:
    m = eligible_master(n_eligible, n_second)
    items, report = _pairs_only(m)
    assert_core_invariants(items, m)
    a = items[items["arm"] == "A"]
    assert report["rung"] == rung and (items["rung"] == rung).all()
    assert len(a) == n_pairs and report["pairs"]["n_selected"] == n_pairs
    per_patient = a["nih_patient_id"].value_counts()
    assert int((per_patient == 2).sum()) == n_two == report["pairs"]["n_patients_with_two_pairs"]
    assert per_patient.max() <= 2
    assert report["pairs"]["n_eligible_patients"] == n_eligible == report["n_eligible_patients"]
    assert report["pairs"]["n_second_pair_patients_available"] == n_second
    assert report["u"]["view_matched"] is None  # no U requested: nothing to match
    if n_two:  # second pairs only from patients with two images of each state
        assert set(per_patient[per_patient == 2].index) <= set(range(1, n_second + 1))
    if rung == "n80" and n_eligible >= 80:
        assert a["nih_patient_id"].is_unique


@pytest.mark.parametrize(("n_eligible", "n_second"), [(70, 9), (50, 20), (0, 0), (79, 0)])
def test_fewer_than_80_possible_pairs_raises(n_eligible: int, n_second: int) -> None:
    m = eligible_master(n_eligible, n_second) if n_eligible else frame([img(1, 0, 0), img(2, 0, 1)])
    with pytest.raises(ValueError, match=r"at least 80"):
        _pairs_only(m)


def test_rung_two_per_patient_second_pairs_are_disjoint_and_nearest() -> None:
    rows: list[dict] = []
    for pid in range(1, 101):
        if pid <= 40:  # 3 present / 3 absent, interleaved
            rows += [img(pid, f, 1 if f % 2 == 0 else 0) for f in range(6)]
        else:
            rows += [img(pid, 0, 1), img(pid, 1, 0)]
    m = frame(rows)
    items, report = _pairs_only(m)
    assert report["rung"] == "two_per_patient" and report["pairs"]["n_patients_with_two_pairs"] == 20
    assert_core_invariants(items, m)


def test_invalid_counts_raise() -> None:
    m = eligible_master(5, 0)
    with pytest.raises(ValueError, match="n_pairs"):
        I.select_core(m, n_pairs=0)
    with pytest.raises(ValueError, match="n_dev_pos"):
        I.select_core(m, n_pairs=1, n_dev_pos=-1)
    with pytest.raises(ValueError, match="seed"):
        I.select_core(m, seed=-1, n_pairs=1)


# ============================================================================ U strata and the H1 substitute
def test_h1_substitute_switches_below_30_noline_patients() -> None:
    m29 = h1_master(29)
    items, report = I.select_core(m29, n_dev_pos=0, n_dev_neg=0)
    assert_core_invariants(items, m29)
    u = items[items["arm"] == "U"]
    assert report["h1_substitute"] is True and report["u"]["h1_substitute_rule"] == "auto"
    assert report["u"]["h1_contrast"] == {"exposed": "ge2", "reference": "eq1"}
    assert report["u"]["descriptive_strata"] == ["noline"]
    assert report["u"]["n_noline_patients_available"] == 29
    assert u["stratum"].value_counts().to_dict() == {"ge2": 60, "eq1": 60, "noline": 29}
    assert "noline" in report["u"]["short_strata"]

    m30 = h1_master(30)
    items30, report30 = I.select_core(m30, n_dev_pos=0, n_dev_neg=0)
    u30 = items30[items30["arm"] == "U"]
    assert report30["h1_substitute"] is False
    assert report30["u"]["h1_contrast"] == {"exposed": "line", "reference": "noline"}
    assert u30["stratum"].value_counts().to_dict() == {"line": 60, "noline": 30}
    assert report30["u"]["short_strata"] == ["noline"] and report30["u"]["strata"]["noline"]["short"] is True


def test_h1_substitute_can_drop_descriptive_noline_or_be_forced() -> None:
    items, report = I.select_core(h1_master(29), n_dev_pos=0, n_dev_neg=0, keep_noline_descriptive=False)
    assert set(items.loc[items["arm"] == "U", "stratum"]) == {"ge2", "eq1"}
    assert report["u"]["descriptive_strata"] == []
    forced_off, rep_off = I.select_core(h1_master(29), n_dev_pos=0, n_dev_neg=0, h1_substitute=False)
    assert rep_off["h1_substitute"] is False and rep_off["u"]["h1_substitute_rule"] == "forced"
    assert rep_off["u"]["h1_substitute_auto_decision"] is True
    assert set(forced_off.loc[forced_off["arm"] == "U", "stratum"]) == {"line", "noline"}
    forced_on, rep_on = I.select_core(h1_master(45), n_dev_pos=0, n_dev_neg=0, h1_substitute=True)
    assert rep_on["h1_substitute"] is True
    assert set(forced_on.loc[forced_on["arm"] == "U", "stratum"]) == {"ge2", "eq1", "noline"}


def test_noline_count_excludes_pair_patients() -> None:
    # 130 pair-eligible patients whose ETT-absent images have no line, 25 other no-line patients:
    # the 120 pair patients do not count, so the available no-line patients are 25 + (130 - 120)
    rows: list[dict] = []
    for pid in range(1, 131):
        rows += [img(pid, 0, 1, 1), img(pid, 1, 0, 0)]
    for pid in range(200, 225):
        rows.append(img(pid, 0, 0, 0))
    for pid in range(300, 400):
        rows.append(img(pid, 0, 0, 1 + pid % 2))
    m = frame(rows)
    _, report = I.select_core(m, n_dev_pos=0, n_dev_neg=0)
    assert report["u"]["n_noline_patients_available"] == 35
    assert report["u"]["n_noline_patients_master"] == 155
    assert report["h1_substitute"] is False


def test_noline_has_priority_for_patients_eligible_for_both_strata() -> None:
    # 40 patients each have one no-line and one line ETT-absent image: all go to "noline"
    rows = eligible_master(120, 0).to_dict("records")
    for pid in range(500, 540):
        rows += [img(pid, 0, 0, 0, "AP"), img(pid, 1, 0, 2, "AP")]
    for pid in range(600, 700):
        rows.append(img(pid, 0, 0, 1, "AP"))
    m = frame(rows)
    items, report = I.select_core(m, n_dev_pos=0, n_dev_neg=0)
    u = items[items["arm"] == "U"]
    assert (u["stratum"] == "noline").sum() == 40 and report["u"]["strata"]["noline"]["available"]["patients"] == 40
    assert set(u.loc[u["stratum"] == "noline", "nih_patient_id"]) == set(range(500, 540))
    assert u["nih_patient_id"].is_unique


def test_patients_eligible_for_two_strata_stay_available_to_the_later_stratum() -> None:
    # 100 patients each with one no-line and one line ETT-absent image and no other U candidate:
    # "noline" draws 60 of them, and "line" can still draw the 40 it did not select
    rows = eligible_master(120, 0).to_dict("records")
    for pid in range(500, 600):
        rows += [img(pid, 0, 0, 0, "AP"), img(pid, 1, 0, 2, "AP")]
    m = frame(rows)
    items, report = I.select_core(m, n_dev_pos=0, n_dev_neg=0)
    assert_core_invariants(items, m)
    u = items[items["arm"] == "U"]
    assert u["stratum"].value_counts().to_dict() == {"noline": 60, "line": 40}
    assert set(u["nih_patient_id"]) == set(range(500, 600))
    strata = report["u"]["strata"]
    assert strata["line"]["eligible_patients"] == 100 and strata["line"]["available"]["patients"] == 40
    assert report["u"]["draw_order"] == ["noline", "line"] and report["u"]["short_strata"] == ["line"]


def test_substitute_draws_the_scarcer_contrast_stratum_first() -> None:
    rows = eligible_master(120, 0).to_dict("records")
    for pid in range(1000, 1050):  # eligible for both eq1 and ge2
        rows += [img(pid, 0, 0, 1, "AP"), img(pid, 1, 0, 2, "AP")]
    for pid in range(1100, 1120):  # eq1 only
        rows.append(img(pid, 0, 0, 1, "AP"))
    for pid in range(1200, 1300):  # ge2 only
        rows.append(img(pid, 0, 0, 3, "AP"))
    m = frame(rows)
    items, report = I.select_core(m, n_dev_pos=0, n_dev_neg=0)
    assert report["h1_substitute"] is True and report["u"]["descriptive_strata"] == ["noline"]
    assert report["u"]["draw_order"] == ["noline", "eq1", "ge2"]
    assert report["u"]["strata"]["eq1"]["eligible_patients"] == 70
    assert report["u"]["strata"]["ge2"]["eligible_patients"] == 150
    u = items[items["arm"] == "U"]
    assert u["stratum"].value_counts().to_dict() == {"eq1": 60, "ge2": 60}
    assert u["nih_patient_id"].is_unique
    assert_core_invariants(items, m)


def _u_ap(items: pd.DataFrame) -> dict[str, tuple[int, int]]:
    u = items[items["arm"] == "U"]
    return {s: (int(((u["stratum"] == s) & (u["view"] == "AP")).sum()), int(((u["stratum"] == s) & (u["view"] == "PA")).sum()))
            for s in sorted(set(u["stratum"]))}


def test_view_matching_hits_the_b_share_when_available() -> None:
    items, report = I.select_core(view_master(100, 100), n_dev_pos=0, n_dev_neg=0)
    assert report["u"]["target_ap_share"] == 0.75
    assert _u_ap(items) == {"line": (45, 15), "noline": (45, 15)}
    assert report["u"]["view_matched"] is True


def test_view_matching_is_limited_by_availability() -> None:
    # no-line stratum: 40 AP-only and 30 PA-only patients -> AP count range [30, 40]; target 45
    items, report = I.select_core(view_master(40, 30), n_dev_pos=0, n_dev_neg=0)
    assert _u_ap(items) == {"line": (40, 20), "noline": (40, 20)}
    assert report["u"]["strata"]["noline"]["ap_range"] == [30, 40]
    assert report["u"]["view_matched"] is True


def test_short_stratum_takes_all_and_shares_are_matched() -> None:
    items, report = I.select_core(view_master(30, 15), n_dev_pos=0, n_dev_neg=0)
    assert _u_ap(items) == {"line": (40, 20), "noline": (30, 15)}
    assert report["u"]["short_strata"] == ["noline"] and report["u"]["view_matched"] is True
    assert report["u"]["strata"]["noline"]["selected"] == 45


def test_view_matching_with_two_view_patients() -> None:
    items, report = I.select_core(view_master(0, 0, noline_both=70), n_dev_pos=0, n_dev_neg=0)
    assert _u_ap(items) == {"line": (45, 15), "noline": (45, 15)}
    noline = items[(items["arm"] == "U") & (items["stratum"] == "noline")]
    assert noline["nih_patient_id"].is_unique and report["u"]["strata"]["noline"]["available"]["both_views"] == 70


def test_view_matching_reports_an_unavoidable_mismatch() -> None:
    items, report = I.select_core(view_master(60, 0, line_ap=5, line_pa=100), n_dev_pos=0, n_dev_neg=0)
    assert _u_ap(items) == {"line": (5, 55), "noline": (60, 0)}
    assert report["u"]["view_matched"] is False


def test_round_half_up() -> None:
    assert I._round_half_up(61 * 60, 120) == 31  # 30.5 -> 31
    assert I._round_half_up(59 * 60, 120) == 30  # 29.5 -> 30
    assert I._round_half_up(0, 7) == 0 and I._round_half_up(7, 7) == 1


# ============================================================================ dev pool
def test_dev_shortfall_takes_the_most_items_as_balanced_as_possible() -> None:
    m = dev_master(pos_only=5, neg_only=100, extra_eligible=10)
    items, report = I.select_core(m, n_u_per_stratum=0)
    dev = items[items["arm"] == "DEV"]
    assert dev["stratum"].value_counts().to_dict() == {"neg": 100, "pos": 15}
    assert report["dev"]["short"] is True and report["dev"]["selected"] == {"pos": 15, "neg": 100}
    assert report["dev"]["available"] == {"patients": 115, "pos_only": 5, "neg_only": 100, "both_states": 10}
    assert_core_invariants(items, m)


def test_dev_full_targets() -> None:
    m = dev_master(pos_only=200, neg_only=200, extra_eligible=0)
    items, report = I.select_core(m, n_u_per_stratum=0)
    dev = items[items["arm"] == "DEV"]
    assert dev["stratum"].value_counts().to_dict() == {"pos": 150, "neg": 150} and report["dev"]["short"] is False


def test_capped_assign_fills_exact_quotas() -> None:
    rng = np.random.default_rng(0)
    patients = list(range(30))
    can_a = [i < 20 for i in patients]  # 0-9 a only, 10-19 both, 20-29 b only
    can_b = [i >= 10 for i in patients]
    for n_a, n_b in ((20, 10), (10, 20), (5, 5), (0, 0), (15, 15)):
        got = I._capped_assign(rng, patients, can_a, can_b, n_a, n_b)
        assert sum(got.values()) == n_a and len(got) - sum(got.values()) == n_b
        assert all(can_a[p] if cls else can_b[p] for p, cls in got.items())


# ============================================================================ ids and smoke
def test_item_ids_are_opaque_unique_and_deterministic(big_outputs: I.SelectionOutputs) -> None:
    items = big_outputs.items
    ids = items["item_id"]
    assert ids.str.fullmatch(r"[0-9a-f]{10}").all() and ids.is_unique
    assert not any(I._reads_as_number(x) for x in ids)
    again = I.assign_item_ids(items, C.derive_seed("ids"))
    assert (again["item_id"] == ids).all()
    other = I.assign_item_ids(items, C.derive_seed("ids") + 1)
    assert not set(other["item_id"]) & set(ids)
    # sorting by id does not group arms, and ids are unrelated to the row order
    by_id = items.sort_values("item_id")["arm"].tolist()
    counts = items["arm"].value_counts().tolist()
    assert n_runs(by_id) >= 0.8 * expected_runs(counts)
    assert abs(rank_corr(np.arange(len(items)), items["item_id"].to_numpy())) < 0.15
    ranks = np.argsort(np.argsort(items["item_id"].to_numpy()))
    half = (len(items) - 1) / 2
    for arm in I.ARMS:
        assert abs(ranks[(items["arm"] == arm).to_numpy()].mean() - half) < 0.2 * half, arm


def test_random_hex_ids_redraw_numeric_looking_values() -> None:
    ids = I._random_hex_ids(np.random.default_rng(1), 3000, 4, upper=True)
    assert len(set(ids)) == 3000
    assert all(len(x) == 4 and set(x) <= set("0123456789ABCDEF") for x in ids)
    assert not any(I._reads_as_number(x) for x in ids)
    assert I._reads_as_number("0012") and I._reads_as_number("12E4") and not I._reads_as_number("12AB")


def test_mark_smoke_is_balanced_dev_only_and_deterministic(big_outputs: I.SelectionOutputs) -> None:
    items = big_outputs.items
    smoke = items[items["smoke"] == 1]
    assert len(smoke) == 20 and (smoke["arm"] == "DEV").all()
    assert smoke["stratum"].value_counts().to_dict() == {"pos": 10, "neg": 10}
    again = I.mark_smoke(items, 20, C.derive_seed("smoke"))
    assert (again["smoke"] == items["smoke"]).all()
    odd = I.mark_smoke(items, 5, 123)
    assert odd.loc[odd["smoke"] == 1, "stratum"].value_counts().to_dict() == {"pos": 3, "neg": 2}


def test_mark_smoke_fills_from_the_other_class_when_short() -> None:
    m = dev_master(pos_only=3, neg_only=50, extra_eligible=0)
    items, _ = I.select_core(m, n_u_per_stratum=0)
    items = I.mark_smoke(I.assign_item_ids(items), 20)
    smoke = items[items["smoke"] == 1]
    assert len(smoke) == 20 and smoke["stratum"].value_counts().to_dict() == {"neg": 17, "pos": 3}


# ============================================================================ pod list and NIH list
def test_pod_items_carry_no_labels(big_outputs: I.SelectionOutputs) -> None:
    pod, items = big_outputs.pod_items, big_outputs.items
    assert list(pod.columns) == ["item_id", "nih_filename", "prompt_set", "smoke"]
    assert len(pod) == len(items) and pod["item_id"].is_monotonic_increasing
    merged = pod.merge(items, on="item_id", suffixes=("", "_items"))
    assert (merged["nih_filename"] == merged["nih_filename_items"]).all()
    want = np.where(merged["arm"] == "DEV", "p0", "full")
    assert (merged["prompt_set"].to_numpy() == want).all()
    assert (merged["smoke"] == merged["smoke_items"]).all()
    cells = set(pod.astype(str).to_numpy().ravel())
    assert not cells & set(items["uid"]) and not cells & set(items["pair_id"].replace("", np.nan).dropna())
    # row order of the pod list does not group arms
    arm_seq = merged.set_index("item_id").loc[pod["item_id"], "arm"].tolist()
    assert n_runs(arm_seq) >= 0.8 * expected_runs(items["arm"].value_counts().tolist())


def test_pod_items_refuses_missing_ids_and_test_smoke(big_outputs: I.SelectionOutputs) -> None:
    items = big_outputs.items.copy()
    with pytest.raises(ValueError, match="assign_item_ids"):
        I.pod_items(items.assign(item_id=""))
    bad = items.copy()
    bad.loc[bad.index[(bad["arm"] == "A").to_numpy()][0], "smoke"] = 1
    with pytest.raises(ValueError, match="dev pool only"):
        I.pod_items(bad)


def test_nih_filelist_lists_every_selected_image(big_outputs: I.SelectionOutputs) -> None:
    fl = big_outputs.nih_filelist
    assert list(fl.columns) == ["nih_filename"] and fl["nih_filename"].is_monotonic_increasing
    assert set(fl["nih_filename"]) == set(big_outputs.items["nih_filename"]) and len(fl) == 660


# ============================================================================ reading and retest lists
def test_reading_list_composition_and_key(big_outputs: I.SelectionOutputs) -> None:
    items, reading, key = big_outputs.items, big_outputs.reading_list, big_outputs.reading_key
    assert list(reading.columns) == ["read_id", "nih_filename", "image_path"]
    assert list(key.columns) == ["read_id", "item_id"]
    assert len(reading) == 360 and (reading["read_id"] == key["read_id"]).all()
    assert reading["read_id"].str.fullmatch(r"[0-9A-F]{8}").all() and reading["read_id"].is_unique
    joined = key.merge(items, on="item_id")
    assert (joined["nih_filename"].to_numpy() == reading["nih_filename"].to_numpy()).all()
    arms = joined["arm"].value_counts().to_dict()
    assert arms == {"B": 120, "U": 120, "A": 120}  # every A is read (PROTOCOL v0.4)
    assert set(items.loc[items["arm"].isin(["B", "U"]), "item_id"]) <= set(key["item_id"])
    assert (reading["image_path"] == "data/images/" + reading["nih_filename"]).all()
    # one random order: arms not grouped by reading order or by read_id
    counts = list(arms.values())
    assert n_runs(joined["arm"].tolist()) >= 0.8 * expected_runs(counts)
    by_read_id = joined.sort_values("read_id")["arm"].tolist()
    assert n_runs(by_read_id) >= 0.8 * expected_runs(counts)
    assert abs(rank_corr(np.arange(len(reading)), reading["read_id"].to_numpy())) < 0.2
    # read ids are unrelated to item ids
    assert not (joined["item_id"].str[:8].str.upper() == joined["read_id"]).any()
    again, key2 = I.reading_lists(items, "data/images", C.N_READ_A, C.derive_seed("reading"))
    pd.testing.assert_frame_equal(again, reading)
    pd.testing.assert_frame_equal(key2, key)


def test_reading_list_image_dir_and_small_a_pool(big_outputs: I.SelectionOutputs, tmp_path: Path) -> None:
    reading, _ = I.reading_lists(big_outputs.items, tmp_path / "png dir", 30, 5)
    assert reading["image_path"].str.startswith((tmp_path / "png dir").as_posix() + "/").all()
    a_only = big_outputs.items[big_outputs.items["arm"] == "A"].head(10)
    small, small_key = I.reading_lists(a_only, "x", 30, 5)  # fewer A items than n_a: take all
    assert len(small) == 10 and set(small_key["item_id"]) == set(a_only["item_id"])


def test_retest_list_is_a_reordered_subset(big_outputs: I.SelectionOutputs) -> None:
    reading, retest = big_outputs.reading_list, big_outputs.retest_list
    assert list(retest.columns) == list(I.READING_LIST_COLS) and len(retest) == 100
    assert retest["read_id"].is_unique and set(retest["read_id"]) <= set(reading["read_id"])
    pd.testing.assert_frame_equal(
        retest.sort_values("read_id").reset_index(drop=True),
        reading[reading["read_id"].isin(retest["read_id"])].sort_values("read_id").reset_index(drop=True),
    )
    position = {r: i for i, r in enumerate(reading["read_id"])}
    original_order = sorted(retest["read_id"], key=position.get)
    assert retest["read_id"].tolist() != original_order
    assert abs(rank_corr(np.arange(100), np.array([position[r] for r in retest["read_id"]]))) < 0.3
    pd.testing.assert_frame_equal(I.retest_list(reading, 100, C.derive_seed("retest")), retest)
    assert len(I.retest_list(reading.head(40), 100, 1)) == 40


# ============================================================================ report, files, determinism
def test_write_outputs_files_report_and_hashes(big_outputs: I.SelectionOutputs, tmp_path: Path) -> None:
    out = tmp_path / "items"
    report = I.write_outputs(big_outputs, out)
    names = sorted(p.name for p in out.iterdir())
    assert names == sorted([*I.OUTPUT_FILES.values(), *I.META_FILES.values()])
    assert not any(("ph" + "ash") in n or ("map" + "ping") in n for n in names)
    for name, digest in report["sha256"].items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == digest
    assert sorted(report["sha256"]) == sorted(v for k, v in I.OUTPUT_FILES.items() if k != "selection_report")
    on_disk = json.loads((out / "selection_report.json").read_text(encoding="utf-8"))
    assert on_disk == report
    for name in report["sha256"]:
        data = (out / name).read_bytes()
        assert b"\r\n" not in data and data.endswith(b"\n")
    assert (out / "pod_items.csv").read_text(encoding="utf-8").splitlines()[0] == "item_id,nih_filename,prompt_set,smoke"
    assert (out / "nih_filelist.csv").read_text(encoding="utf-8").splitlines()[0] == "nih_filename"
    assert "synuid" not in (out / "pod_items.csv").read_text(encoding="utf-8")
    text = (out / "selection_report.json").read_text(encoding="utf-8")
    assert "synuid" not in text and ".png" not in text  # aggregates only


def test_selection_report_contents(big_outputs: I.SelectionOutputs, tmp_path: Path) -> None:
    report = I.write_outputs(big_outputs, tmp_path)
    items = big_outputs.items
    for key in ("counts", "rung", "h1_substitute", "view_shares", "followup_gap", "direction_counts",
                "n_eligible_patients", "sha256", "pairs", "u", "dev", "smoke", "reading", "seeds", "params"):
        assert key in report, key
    assert report["counts"]["by_arm"] == {"A": 120, "B": 120, "U": 120, "DEV": 300}
    assert report["counts"]["by_stratum"] == {"U": {"line": 60, "noline": 60}, "DEV": {"pos": 150, "neg": 150}}
    assert report["counts"]["n_test_items"] == 360 and report["counts"]["n_items"] == 660
    a = items[items["arm"] == "A"]
    gaps = a["gap"].to_numpy()
    fg = report["followup_gap"]
    assert fg["n"] == 120 and fg["min"] == gaps.min() and fg["max"] == gaps.max()
    assert fg["median"] == pytest.approx(np.median(gaps)) and fg["mean"] == pytest.approx(gaps.mean(), abs=1e-6)
    assert fg["q1"] == pytest.approx(np.percentile(gaps, 25)) and fg["n_gap_1"] == int((gaps == 1).sum())
    assert report["direction_counts"] == a["direction"].value_counts().reindex(list(I.DIRECTIONS), fill_value=0).to_dict()
    for arm in I.ARMS:
        sub = items[items["arm"] == arm]
        share = report["view_shares"]["by_arm"][arm]
        assert share["AP"] == int((sub["view"] == "AP").sum()) and share["AP"] + share["PA"] == len(sub)
    assert report["view_shares"]["by_stratum"]["U:line"]["AP"] == report["view_shares"]["by_stratum"]["U:noline"]["AP"]
    assert report["smoke"] == {"n": 20, "pos": 10, "neg": 10}
    assert report["reading"] == {"n_reading_list": 360, "n_retest": 100, "n_practice": 10}
    assert report["seeds"] == {p: C.derive_seed(p) for p in ("pairs", "u", "dev", "ids", "smoke", "reading", "retest")}
    assert report["params"]["n_pairs"] == 120 and report["params"]["n_read_a"] == C.N_READ_A


def test_identical_inputs_give_byte_identical_files(big_master: pd.DataFrame, tmp_path: Path) -> None:
    I.write_outputs(I.build_outputs(big_master, "data/images"), tmp_path / "one")
    shuffled = big_master.sample(frac=1.0, random_state=7).reset_index(drop=True)
    I.write_outputs(I.build_outputs(shuffled, "data/images"), tmp_path / "two")
    for name in I.OUTPUT_FILES.values():
        assert (tmp_path / "one" / name).read_bytes() == (tmp_path / "two" / name).read_bytes(), name
    I.write_outputs(I.build_outputs(big_master, "data/images", seed=C.SEED + 5), tmp_path / "three")
    assert (tmp_path / "one" / "items.csv").read_bytes() != (tmp_path / "three" / "items.csv").read_bytes()


def test_byte_identical_across_processes(tmp_path: Path) -> None:
    """Same inputs in a fresh interpreter with another hash seed give the same bytes."""
    master = random_master(600, seed=41)
    master.to_csv(tmp_path / "master.csv", index=False)
    I.write_outputs(I.build_outputs(master, "data/images"), tmp_path / "here")
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); import pandas as pd; from cxrdev import items as I; "
        "m = pd.read_csv(sys.argv[2], dtype={c: str for c in ('uid', 'ranzcr_patient_id', 'nih_filename', 'view', 'sex')}); "
        "I.write_outputs(I.build_outputs(m, 'data/images'), sys.argv[3])"
    )
    env = dict(os.environ, PYTHONHASHSEED="12345")
    subprocess.run([PYTHON, "-c", code, str(ROOT), str(tmp_path / "master.csv"), str(tmp_path / "there")],
                   env=env, check=True, timeout=300, capture_output=True)
    for name in I.OUTPUT_FILES.values():
        assert (tmp_path / "here" / name).read_bytes() == (tmp_path / "there" / name).read_bytes(), name


def test_write_outputs_refuses_to_overwrite(big_outputs: I.SelectionOutputs, tmp_path: Path) -> None:
    I.write_outputs(big_outputs, tmp_path)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        I.write_outputs(big_outputs, tmp_path)
    I.write_outputs(big_outputs, tmp_path, overwrite=True)


def test_read_items_round_trip(big_outputs: I.SelectionOutputs, tmp_path: Path) -> None:
    I.write_outputs(big_outputs, tmp_path)
    back = I.read_items(tmp_path / "items.csv")
    pd.testing.assert_frame_equal(back, big_outputs.items)
    assert (back.loc[back["arm"] == "U", "pair_id"] == "").all()


def test_build_outputs_through_s1_s2_s3_frames(tmp_path: Path) -> None:
    master = random_master(1500, seed=21)
    mt, pr, nm = split_master(master)
    built = I.build_master_table(mt, pr, nm)
    a = I.build_outputs(built, "imgs")
    b = I.build_outputs(master, "imgs")
    pd.testing.assert_frame_equal(a.items, b.items)
    assert_core_invariants(a.items, master)


# ============================================================================ script
def _fake_modules() -> tuple[types.ModuleType, types.ModuleType]:
    ranzcr = types.ModuleType("cxrdev.ranzcr")

    def presence(train_df: pd.DataFrame) -> pd.DataFrame:
        ett = train_df[list(C.ETT_COLS)].max(axis=1).astype(int)
        ngt = train_df[list(C.NGT_COLS)].max(axis=1).astype(int)
        cvc = train_df[list(C.CVC_COLS)].max(axis=1).astype(int)
        swan = train_df[C.SWAN_COL].astype(int)
        return pd.DataFrame(
            {
                "uid": train_df[C.RANZCR_UID], "ranzcr_patient_id": train_df[C.RANZCR_PATIENT], "ett": ett,
                "ngt": ngt, "ngt_incomplete": train_df[C.NGT_INCOMPLETE_COL].astype(int), "cvc": cvc,
                "swan": swan, "n_other_lines": ngt + cvc + swan,
            }
        )

    ranzcr.presence = presence
    ranzcr.read_train_csv = lambda path: pd.read_csv(
        path, dtype={C.RANZCR_UID: str, C.RANZCR_PATIENT: str}, keep_default_na=False)
    nih = types.ModuleType("cxrdev.nih")
    nih.load_nih_meta = lambda path: pd.read_csv(path, dtype={"nih_filename": str, "sex": str, "view": str})
    return ranzcr, nih


def _write_script_inputs(tmp_path: Path, master: pd.DataFrame) -> dict[str, Path]:
    mt, pr, nm = split_master(master)
    train = pd.DataFrame({C.RANZCR_UID: pr["uid"]})
    for col in (*C.ETT_COLS, *C.NGT_COLS, *C.CVC_COLS, C.SWAN_COL):
        train[col] = 0
    train["ETT - Normal"] = pr["ett"]
    train["NGT - Normal"] = pr["ngt"] * (1 - pr["ngt_incomplete"])
    train[C.NGT_INCOMPLETE_COL] = pr["ngt_incomplete"]
    train["CVC - Normal"] = pr["cvc"]
    train[C.SWAN_COL] = pr["swan"]
    train[C.RANZCR_PATIENT] = pr["ranzcr_patient_id"]
    paths = {"match": tmp_path / "match_table.csv", "train": tmp_path / "train.csv", "nih": tmp_path / "nih_entries.csv"}
    mt.to_csv(paths["match"], index=False)
    train.to_csv(paths["train"], index=False)
    nm.to_csv(paths["nih"], index=False)
    return paths


def test_script_main_end_to_end(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    ranzcr, nih = _fake_modules()
    monkeypatch.setitem(sys.modules, "cxrdev.ranzcr", ranzcr)
    monkeypatch.setitem(sys.modules, "cxrdev.nih", nih)
    script = importlib.import_module("scripts.select_items")
    master = random_master(1500, seed=31)
    paths = _write_script_inputs(tmp_path, master)
    out = tmp_path / "items"
    argv = ["--match", str(paths["match"]), "--train", str(paths["train"]), "--nih", str(paths["nih"]),
            "--images-dir", "data/images", "--out", str(out)]
    assert script.main(argv) == 0
    assert sorted(p.name for p in out.iterdir()) == sorted([*I.OUTPUT_FILES.values(), *I.META_FILES.values()])
    reading = pd.read_csv(out / "reading_list.csv", dtype=str)
    assert (reading["image_path"] == "data/images/" + reading["nih_filename"]).all()
    printed = capsys.readouterr().out
    assert "rung: standard" in printed and "synuid" not in printed
    # same inputs through the library give the same bytes
    expected = I.build_outputs(I.build_master_table(*split_master(master)), "data/images")
    I.write_outputs(expected, tmp_path / "lib")
    for name in I.OUTPUT_FILES.values():
        assert (out / name).read_bytes() == (tmp_path / "lib" / name).read_bytes(), name
    # no silent overwrite; --force overwrites
    assert script.main(argv) == 2
    assert "refusing to overwrite" in capsys.readouterr().err
    assert script.main([*argv, "--force", "--n-pairs", "80", "--h1-substitute", "yes"]) == 0
    report = json.loads((out / "selection_report.json").read_text(encoding="utf-8"))
    assert report["pairs"]["n_selected"] == 80 and report["h1_substitute"] is True


def test_script_reports_a_short_yield(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    ranzcr, nih = _fake_modules()
    monkeypatch.setitem(sys.modules, "cxrdev.ranzcr", ranzcr)
    monkeypatch.setitem(sys.modules, "cxrdev.nih", nih)
    script = importlib.import_module("scripts.select_items")
    paths = _write_script_inputs(tmp_path, eligible_master(50, 10))
    argv = ["--match", str(paths["match"]), "--train", str(paths["train"]), "--nih", str(paths["nih"]),
            "--images-dir", "d", "--out", str(tmp_path / "out")]
    assert script.main(argv) == 2
    assert "at least 80" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_script_runs_as_a_file() -> None:
    result = subprocess.run([PYTHON, str(ROOT / "scripts" / "select_items.py"), "--help"],
                            capture_output=True, text=True, cwd=str(ROOT / "tests"), timeout=120)
    assert result.returncode == 0 and "--images-dir" in result.stdout


# ============================================================================ integration gaps (a, b)
def test_practice_list_dev_only_balanced_own_read_ids(big_outputs: I.SelectionOutputs) -> None:
    practice, items, reading = big_outputs.practice_list, big_outputs.items, big_outputs.reading_list
    assert practice is not None and list(practice.columns) == list(I.READING_LIST_COLS) and len(practice) == 10
    arm_of = dict(zip(items["nih_filename"], items["arm"]))
    stratum_of = dict(zip(items["nih_filename"], items["stratum"]))
    assert {arm_of[f] for f in practice["nih_filename"]} == {"DEV"}  # never a test image
    assert sorted(stratum_of[f] for f in practice["nih_filename"]) == ["neg"] * 5 + ["pos"] * 5
    assert practice["read_id"].str.fullmatch(r"[0-9A-F]{8}").all() and practice["read_id"].is_unique
    assert not set(practice["read_id"]) & set(reading["read_id"])
    assert (practice["image_path"] == "data/images/" + practice["nih_filename"]).all()
    again = I.practice_list(items, "data/images", 10, I.practice_seed(), reading["read_id"].tolist())
    pd.testing.assert_frame_equal(again, practice)
    assert I.practice_seed() == C.SEED * 100 + I.PRACTICE_SEED_INDEX
    assert I.practice_seed() not in {C.derive_seed(k) for k in C._PURPOSES}
    no_dev = items[items["arm"] != "DEV"]
    assert len(I.practice_list(no_dev, "x")) == 0


def test_write_outputs_seed_sidecars_and_practice_list(big_outputs: I.SelectionOutputs, tmp_path: Path) -> None:
    report = I.write_outputs(big_outputs, tmp_path)
    for name, purpose in (("reading_list", "reading"), ("retest_list", "retest")):
        meta = json.loads((tmp_path / I.META_FILES[name]).read_text(encoding="utf-8"))
        assert meta == {"seed": C.derive_seed(purpose)}
    assert json.loads((tmp_path / "practice_list.meta.json").read_text(encoding="utf-8")) == {"seed": I.practice_seed()}
    assert report["sha256"]["practice_list.csv"] == hashlib.sha256((tmp_path / "practice_list.csv").read_bytes()).hexdigest()
    # the reading tool reads these seeds back
    rc = importlib.import_module("tools.reading_core")
    assert rc.read_list_seed(tmp_path / "reading_list.csv") == C.derive_seed("reading")
    assert rc.read_list_seed(tmp_path / "retest_list.csv") == C.derive_seed("retest")
    assert rc.read_list_seed(tmp_path / "practice_list.csv") == I.practice_seed()
    rc.check_practice_list(tmp_path / "practice_list.csv")  # accepted for --practice
    with pytest.raises(rc.ListError):
        rc.check_practice_list(tmp_path / "reading_list.csv")
    # a practice list that would show a test image is refused
    items = big_outputs.items
    bad = big_outputs.practice_list.copy()
    bad.loc[0, "nih_filename"] = items.loc[items["arm"] == "A", "nih_filename"].iloc[0]
    with pytest.raises(ValueError, match="practice list"):
        I.write_outputs(dataclasses.replace(big_outputs, practice_list=bad), tmp_path / "bad")


def test_select_items_reads_train_csv_ids_as_text(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Gap (a): the real ``cxrdev.ranzcr.read_train_csv`` keeps digit-only and e-notation ids as text."""
    _ranzcr, nih = _fake_modules()
    monkeypatch.setitem(sys.modules, "cxrdev.nih", nih)  # the NIH reader is tested elsewhere
    script = importlib.import_module("scripts.select_items")
    master = random_master(400, seed=13)
    odd = {pid: (f"{pid:09d}" if pid % 3 == 0 else (f"{pid:05d}e{pid % 7}" if pid % 3 == 1 else f"NA{pid}"))
           for pid in master["nih_patient_id"].unique()}
    master["ranzcr_patient_id"] = master["nih_patient_id"].map(odd)
    master["uid"] = [f"1.2.840.{i:06d}" for i in range(len(master))]
    paths = _write_script_inputs(tmp_path, master)
    match_table, presence, _meta = script.load_inputs(paths["match"], paths["train"], paths["nih"])
    assert presence["ranzcr_patient_id"].map(type).eq(str).all()
    assert set(odd.values()) <= set(presence["ranzcr_patient_id"])
    built = I.build_master_table(match_table, presence, _meta)
    assert len(built) == len(master) and set(built["ranzcr_patient_id"]) == set(odd.values())


def test_exclude_filenames_removed_before_any_selection(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
                                                        capsys: pytest.CaptureFixture) -> None:
    """--exclude-filenames: listed NIH images never enter pairs, U or the dev pool; counts and sha256 reported."""
    ranzcr, nih = _fake_modules()
    monkeypatch.setitem(sys.modules, "cxrdev.ranzcr", ranzcr)
    monkeypatch.setitem(sys.modules, "cxrdev.nih", nih)
    script = importlib.import_module("scripts.select_items")
    master = random_master(1500, seed=31)
    paths = _write_script_inputs(tmp_path, master)
    base = ["--match", str(paths["match"]), "--train", str(paths["train"]), "--nih", str(paths["nih"]),
            "--images-dir", "data/images"]
    assert script.main([*base, "--out", str(tmp_path / "plain")]) == 0
    plain = I.read_items(tmp_path / "plain" / "items.csv")
    # exclude two selected images of each arm, two unselected matched images and one name outside the pool
    chosen = []
    for arm in ("A", "B", "U", "DEV"):
        chosen += plain.loc[plain["arm"] == arm, "nih_filename"].head(2).tolist()
    unselected = sorted(set(master["nih_filename"]) - set(plain["nih_filename"]))[:2]
    listed = chosen + unselected + ["99999999_000.png"]
    text = "# probes seen before the protocol freeze\n\n" + "\n".join(listed) + "\n" + chosen[0] + "\n"
    probe = tmp_path / "probe_seen_filenames.txt"
    probe.write_bytes(text.encode("utf-8"))
    assert script.main([*base, "--out", str(tmp_path / "excl"), "--exclude-filenames", str(probe)]) == 0
    items = I.read_items(tmp_path / "excl" / "items.csv")
    assert not set(listed) & set(items["nih_filename"])
    for name in ("pod_items.csv", "nih_filelist.csv", "reading_list.csv", "retest_list.csv", "practice_list.csv"):
        body = (tmp_path / "excl" / name).read_text(encoding="utf-8")
        assert not any(n in body for n in listed), name
    report = json.loads((tmp_path / "excl" / "selection_report.json").read_text(encoding="utf-8"))
    ex = report["excluded_filenames"]
    assert ex == {"file": "probe_seen_filenames.txt", "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                  "n_listed": 11, "n_present_in_matched_pool": 10}
    assert "99999999_000.png" not in json.dumps(report) and chosen[0] not in json.dumps(report)  # counts only
    assert report["counts"]["by_arm"] == {"A": 120, "B": 120, "U": 120, "DEV": 300}  # still the registered n
    assert "excluded filenames: 11 listed, 10 were in the matched pool" in capsys.readouterr().out
    assert "excluded_filenames" not in json.loads((tmp_path / "plain" / "selection_report.json").read_text(encoding="utf-8"))
    # library: the same exclusion through build_outputs; a malformed line is refused
    names, sha = I.read_filename_list(probe)
    assert names == listed and sha == ex["sha256"]
    out = I.build_outputs(I.build_master_table(*split_master(master)), "data/images", exclude_filenames=names)
    pd.testing.assert_frame_equal(out.items, items)
    bad = tmp_path / "bad.txt"
    bad.write_text("00000001_000.png\nnot-a-file.png\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        I.read_filename_list(bad)
    assert script.main([*base, "--out", str(tmp_path / "bad"), "--exclude-filenames", str(bad)]) == 2


# ============================================================================ wording
def test_source_wording() -> None:
    banned = ["counter" + "factual", "verb" + "atim", "label-" + "cl" + "ean", "un" + "seen",
              "radiologist " + "label", "inter" + "-rater"]
    for path in (ROOT / "cxrdev" / "items.py", ROOT / "scripts" / "select_items.py"):
        text = path.read_text(encoding="utf-8").lower()
        for word in banned:
            assert word not in text, (path.name, word)
        assert "43 trained labellers" in text
