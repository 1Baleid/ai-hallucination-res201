"""Tests for cxrdev.hypotheses (H1, H3, H5, PLUS H2, Holm families). Synthetic data only."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from cxrdev import constants as C
from cxrdev import hypotheses as H
from cxrdev import metrics as M
from cxrdev import parse, stats
from cxrdev.items import ITEM_COLS
from scripts import synthetic_dryrun as SD


# ----------------------------------------------------------------------------- builders
def make_items(specs: list[tuple]) -> pd.DataFrame:
    """S5 rows from (item_id, arm, pair_id, stratum, patient, n_other_lines)."""
    rows = []
    for item_id, arm, pair_id, stratum, patient, n_other in specs:
        ett = 1 if arm == "A" or (arm == "DEV" and stratum == "pos") else 0
        rows.append({
            "item_id": item_id, "uid": f"synthetic.{item_id}", "nih_filename": f"{patient:08d}_{item_id}.png",
            "nih_patient_id": patient, "arm": arm, "pair_id": pair_id, "stratum": stratum, "ett": ett, "ngt": 0,
            "ngt_incomplete": 0, "cvc": 0, "swan": 0, "n_other_lines": n_other, "view": "AP", "followup": 0, "age": 50,
            "sex": "M", "direction": "", "gap": -1, "rung": "standard", "smoke": 0,
        })
    return pd.DataFrame(rows, columns=list(ITEM_COLS))


def make_frame(items: pd.DataFrame, outputs: list[tuple]) -> pd.DataFrame:
    """Scoring frame from (model, prompt, item_id, raw_output) outputs (z follows the answer)."""
    fname = dict(zip(items["item_id"], items["nih_filename"]))
    rows = []
    for model, prompt, item_id, raw in outputs:
        p = parse.parse_answer(raw)
        rows.append({
            "run_id": "r", "model_key": model, "model_id": "m", "revision": "r", "transformers_version": "t",
            "torch_version": "t", "dtype": "bf16", "device": "cpu", "item_id": item_id, "nih_filename": fname[item_id],
            "image_sha256": "", "condition": "image", "prompt_id": prompt, "prompt_sha256": "", "prompt_text": "q",
            "visual_tokens": 1, "seed": 1, "max_new_tokens": 16, "raw_output": raw, "n_generated_tokens": 2,
            "ended_with_eos": True, "parsed": p, "parser_version": parse.PARSER_VERSION, "logp_yes": -1.0, "logp_no": -1.0,
            "logp_abstain": float("nan"), "z": {"yes": 1.0, "no": -1.0}.get(p, 0.0), "coverage": 0.9,
            "signal": "first_token", "wall_s": 0.1, "timestamp_utc": "2026-11-11T00:00:00Z", "truncated": False,
        })
    parsed = M.coerce_parsed(pd.DataFrame(rows, columns=list(M.S8_COLUMNS)))
    return M.scoring_frame(parsed, items)


def u_frame(n_line: int, yes_line: int, n_noline: int, yes_noline: int, strata=("line", "noline"), extra=None):
    specs, outs = [], []
    k = 0
    for stratum, n, n_yes in ((strata[0], n_line, yes_line), (strata[1], n_noline, yes_noline), *(extra or [])):
        for i in range(n):
            k += 1
            specs.append((f"u{k:03d}", "U", "", stratum, 1000 + k, 1))
            outs.append(("m1", "P0", f"u{k:03d}", "Yes" if i < n_yes else "No"))
    items = make_items(specs)
    return make_frame(items, outs)


# ----------------------------------------------------------------------------- H1
def test_h1_known_difference_with_fixed_strata(monkeypatch):
    calls = []
    real = stats.clustered_ci

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(stats, "clustered_ci", spy)
    df = u_frame(40, 30, 40, 8)
    r = H.h1(df, "m1", n_boot=300, seed=C.derive_seed("bootstrap"))
    assert r["estimate"] == pytest.approx(30 / 40 - 8 / 40)
    assert (r["exposed"], r["reference"], r["substitute"]) == ("line", "noline", False)
    assert (r["n_exposed"], r["n_reference"], r["n"]) == (40, 40, 80)
    assert r["rate_exposed"]["k"] == 30 and r["rate_reference"]["k"] == 8
    assert r["lo"] < r["estimate"] < r["hi"] and r["lo"] > 0
    assert r["tested"] and r["p_value"] <= 0.01 and r["sided"] == "two-sided"
    assert calls and calls[0]["strata_col"] == "st"  # stratum sizes fixed in every resample


def test_h1_substitute_contrast_ignores_descriptive_noline():
    df = u_frame(35, 25, 35, 10, strata=("ge2", "eq1"), extra=[("noline", 10, 10)])
    r = H.h1(df, "m1", n_boot=100)
    assert (r["exposed"], r["reference"], r["substitute"]) == ("ge2", "eq1", True)
    assert r["n"] == 70 and r["estimate"] == pytest.approx(15 / 35)
    assert r["contrast"] == "U ge2 minus U eq1"


def test_h1_small_cells_are_not_tested_and_n80_rung_is_tested():
    small = H.h1(u_frame(20, 15, 40, 8), "m1", n_boot=100)
    assert small["tested"] is False and math.isnan(small["p_value"]) and "fewer than 30" in small["reason"]
    assert math.isfinite(small["lo"]) and math.isfinite(small["p_value_descriptive"])
    # PROTOCOL v0.4: the U arm stays 60 + 60 under every rung, so H1 is the registered test under n80 too
    n80 = H.h1(u_frame(40, 30, 40, 8), "m1", n_boot=100, rung="n80")
    std = H.h1(u_frame(40, 30, 40, 8), "m1", n_boot=100, rung="standard")
    assert n80["tested"] is True and n80["reason"] == "" and n80["rung"] == "n80"
    assert n80["p_value"] == std["p_value"] and n80["estimate"] == std["estimate"] and n80["lo"] == std["lo"]
    missing = H.h1(u_frame(40, 30, 40, 8, strata=("line", "other")), "m1", n_boot=10)
    assert missing["tested"] is False and math.isnan(missing["estimate"])


def test_h1_uses_valid_outputs_and_exclusions():
    df = u_frame(40, 30, 40, 8)
    df.loc[df["item_id"] == "u001", ["parsed", "valid"]] = ["unparsable", False]  # a "yes" in line
    r = H.h1(df, "m1", n_boot=50, exclusions=["u041"])  # a "yes" in noline
    assert (r["n_exposed"], r["rate_exposed"]["k"]) == (39, 29)
    assert (r["n_reference"], r["rate_reference"]["k"]) == (39, 7)


def test_h1_secondary():
    specs = [(f"b{i}", "B", f"p{i:03d}", "", i, 0) for i in range(30)] + [(f"u{i}", "U", "", "line", 100 + i, 1) for i in range(30)]
    items = make_items(specs)
    outs = [("m1", "P0", f"b{i}", "Yes" if i < 6 else "No") for i in range(30)] + \
           [("m1", "P0", f"u{i}", "Yes" if i < 15 else "No") for i in range(30)]
    sec = H.h1_secondary(make_frame(items, outs), "m1", n_boot=100)
    assert sec["far_b_minus_far_u"]["estimate"] == pytest.approx(6 / 30 - 15 / 30)
    assert (sec["far_b_minus_far_u"]["n_B"], sec["far_b_minus_far_u"]["n_U"]) == (30, 30)


# ----------------------------------------------------------------------------- H3
def h3_frame(n_b=40, n_u=40, p1_abst=range(0, 4), p2_abst=range(2, 30), drop_p1=(79,), missing_p2=(78,), two_b_patients=0):
    specs = []
    for i in range(n_b):
        patient = i // 2 if i < 2 * two_b_patients else 500 + i
        specs.append((f"i{i:03d}", "B", f"p{i:03d}", "", patient, 0))
    for i in range(n_b, n_b + n_u):
        specs.append((f"i{i:03d}", "U", "", "line", 900 + i, 1))
    items = make_items(specs)
    outs = []
    for i in range(n_b + n_u):
        iid = f"i{i:03d}"
        outs.append(("m1", "P1", iid, "Hmm" if i in drop_p1 else ("I don't know" if i in p1_abst else "No")))
        if i not in missing_p2:
            outs.append(("m1", "P2hi", iid, "I don't know." if i in p2_abst else "No"))
    return make_frame(items, outs)


def test_h3_mcnemar_counts_and_dropped():
    r = H.h3(h3_frame(), "m1", "standard", n_boot=100)
    assert (r["b"], r["c"], r["n"]) == (26, 2, 78)
    assert r["estimate"] == pytest.approx(24 / 78)
    assert r["p_value"] == pytest.approx(stats.mcnemar_exact(26, 2)) and r["p_method"] == "exact McNemar"
    assert r["n_candidates"] == 80 and r["n_dropped"] == 2
    assert r["dropped"] == {"missing_P1": 0, "missing_P2hi": 1, "invalid_P1": 1, "invalid_P2hi": 0}
    assert (r["abstention_ref"]["k"], r["abstention_pen"]["k"]) == (4, 28)
    assert r["ci_role"] == "descriptive" and r["lo"] <= r["estimate"] <= r["hi"]
    ex = H.h3(h3_frame(), "m1", "standard", n_boot=0, exclusions=["i005"])
    assert (ex["b"], ex["n"], ex["n_candidates"]) == (25, 77, 79)


def test_h3_no_change_and_small_n():
    r = H.h3(h3_frame(p1_abst=(), p2_abst=()), "m1", n_boot=50)
    assert (r["b"], r["c"]) == (0, 0) and r["estimate"] == 0.0 and r["p_value"] == 1.0
    small = H.h3(h3_frame(n_b=10, n_u=10, p2_abst=range(0, 8), drop_p1=(), missing_p2=()), "m1", n_boot=50)
    assert small["tested"] is False and math.isnan(small["p_value"]) and small["n"] == 20


def test_h3_two_per_patient_rung():
    df = h3_frame(n_b=40, n_u=20, p1_abst=range(0, 3), p2_abst=range(0, 40), drop_p1=(), missing_p2=(), two_b_patients=20)
    r = H.h3(df, "m1", "two_per_patient", n_boot=300)
    assert r["p_method"].startswith("clustered BCa") and r["ci_role"] == "primary"
    assert r["n_clusters"] == 40  # 20 patients with two B items + 20 U patients
    chk = r["mcnemar_check"]
    assert chk["n"] == 40  # one B per patient plus every U item
    assert chk["p_value"] == pytest.approx(stats.mcnemar_exact(chk["b"], chk["c"]))
    assert r["tested"] and r["p_value"] < 0.01 and r["mcnemar_p"] == pytest.approx(stats.mcnemar_exact(r["b"], r["c"]))
    again = H.h3(df, "m1", "two_per_patient", n_boot=300)
    assert again["mcnemar_check"] == chk and again["p_value"] == r["p_value"]  # seeded


# ----------------------------------------------------------------------------- H5, Holm, secondary
def test_h5_from_arm1_result():
    res = {"model_key": "m1", "n_A": 100, "n_B": 98, "skipped_reason": "",
           "gain": {"estimate": 0.07, "lo": 0.03, "hi": 0.11, "method": "bca"},
           "shuffled_expert": {"observed_gain": 0.07, "p_value": 0.002, "n_perm": 9999, "null_mean": -0.01,
                               "null_q025": -0.04, "null_q975": 0.02, "ba_fused": 0.8, "ba_beta0": 0.73, "beta": 1.0,
                               "gamma": 5.0}}
    r = H.h5(res)
    assert r["estimate"] == 0.07 and r["p_value"] == 0.002 and r["tested"] and r["n"] == 198
    assert r["sided"] == "one-sided (greater)" and r["ci_role"] == "descriptive" and (r["lo"], r["hi"]) == (0.03, 0.11)
    small = H.h5({**res, "n_B": 20})
    assert small["tested"] is False and math.isnan(small["p_value"]) and small["p_value_descriptive"] == 0.002
    assert H.h5({"model_key": "m1", "skipped_reason": "no detector"})["tested"] is False
    assert H.h5(None, "m2")["model_key"] == "m2"


def test_holm_family():
    fam = H.holm_family([{"model_key": "a", "p_value": 0.04, "tested": True},
                         {"model_key": "b", "p_value": 0.01, "tested": True},
                         {"model_key": "c", "p_value": float("nan"), "tested": False}])
    assert [f["p_holm"] for f in fam[:2]] == pytest.approx([0.04, 0.02])
    assert math.isnan(fam[2]["p_holm"]) and all(f["family_m"] == 2 for f in fam)
    one = H.holm_family([{"p_value": 0.03, "tested": True}])
    assert one[0]["p_holm"] == pytest.approx(0.03)


def test_h3_secondary():
    s = {"P0": {"accuracy": {"estimate": 0.7}, "FAR_B": {"estimate": 0.2}, "ER_0.9": {"estimate": -1.0}},
         "P2hi": {"accuracy": {"estimate": 0.5}, "FAR_B": {"estimate": 0.1}, "ER_0.9": {"estimate": -0.2},
                  "abstention": {"estimate": 0.3}}}
    out = H.h3_secondary(s)
    assert list(out["ER_0.9"]) == ["P0", "P2hi"] and out["P0_accuracy"]["estimate"] == 0.7
    assert out["FAR_B"]["P2hi"]["estimate"] == 0.1


# ----------------------------------------------------------------------------- PLUS H2
def test_h2_paired_differences():
    specs = [(f"a{i}", "A", f"p{i:03d}", "", i, 0) for i in range(30)] + \
            [(f"b{i}", "B", f"p{i:03d}", "", i, 0) for i in range(30)] + \
            [(f"u{i}", "U", "", "line", 100 + i, 1) for i in range(30)]
    items = make_items(specs)
    outs = []
    for i in range(30):
        outs += [("chexone", "P0", f"a{i}", "Yes" if i < 24 else "No"), ("qwen25vl3b", "P0", f"a{i}", "Yes" if i < 27 else "No")]
        outs += [("chexone", "P0", f"b{i}", "No"), ("qwen25vl3b", "P0", f"b{i}", "Yes" if i < 9 else "No")]
        outs += [("chexone", "P0", f"u{i}", "Hmm" if i == 0 else "No"), ("qwen25vl3b", "P0", f"u{i}", "Yes" if i < 6 else "No")]
    r = H.h2(make_frame(items, outs), "chexone", "qwen25vl3b", n_boot=200)
    # u0 is unparsable for CheXOne, so it leaves both models: 59 paired B and U items
    assert r["n"] == 59
    assert r["hfr_a"]["k"] == 59 and r["hfr_b"]["k"] == 59 - 9 - 5
    assert r["estimate"] == pytest.approx((9 + 5) / 59)
    assert r["tested"] and r["p_value"] < 0.01
    ba_a = (24 / 30 + 1.0) / 2
    ba_b = (27 / 30 + 21 / 30) / 2
    assert r["ba_diff"]["estimate"] == pytest.approx(ba_a - ba_b) and r["ba_diff"]["n"] == 60


# ----------------------------------------------------------------------------- with the synthetic generator
def test_h1_substitute_from_synthetic_items():
    items = SD.make_items(h1_substitute=True)
    parsed, truth = SD.make_parsed(items)
    frame = M.scoring_frame(parsed, items)
    r = H.h1(frame, "qwen3vl2b", n_boot=99)
    assert r["substitute"] and r["contrast"] == "U ge2 minus U eq1"
    planted = truth["qwen3vl2b"]["P0"]
    assert r["estimate"] == pytest.approx(planted["U_1"]["n_yes"] / planted["U_1"]["n_valid"]
                                          - planted["U_0"]["n_yes"] / planted["U_0"]["n_valid"])


def test_h3_two_per_patient_from_synthetic_items():
    items = SD.make_items(n_two_pair_patients=20)
    assert items.loc[items["arm"] == "B", "nih_patient_id"].nunique() == 100
    parsed, _ = SD.make_parsed(items)
    r = H.h3(M.scoring_frame(parsed, items), "medgemma15", "two_per_patient", n_boot=99)
    assert r["p_method"].startswith("clustered") and r["mcnemar_check"]["n"] <= r["n"]
    assert r["estimate"] > 0.2 and r["p_value"] <= 0.01


# ----------------------------------------------------------------------------- integration gaps (d)
def test_h1_registered_contrast_from_selection_report_keeps_noline_descriptive():
    """Under the H1 substitute the contrast is U(ge2) vs U(eq1); no-line U items are never pooled."""
    df = u_frame(35, 25, 35, 10, strata=("ge2", "eq1"), extra=[("noline", 12, 12)])
    contrast = {"exposed": "ge2", "reference": "eq1"}  # selection_report.json u.h1_contrast
    r = H.h1(df, "m1", n_boot=100, contrast=contrast)
    assert (r["exposed"], r["reference"], r["substitute"]) == ("ge2", "eq1", True)
    assert r["contrast_source"] == "selection report"
    assert (r["n_exposed"], r["n_reference"], r["n"]) == (35, 35, 70)
    assert r["estimate"] == pytest.approx(25 / 35 - 10 / 35)  # the 12 "yes" no-line items do not enter
    assert r["n_descriptive"] == 12 and r["descriptive_strata"] == ["noline"]
    # inferred contrast gives the same answer
    inferred = H.h1(df, "m1", n_boot=100)
    assert inferred["estimate"] == r["estimate"] and inferred["contrast_source"] == "item strata"
    # a registered contrast whose strata are missing is not tested (never silently swapped)
    wrong = H.h1(df, "m1", n_boot=10, contrast=("line", "noline"))
    assert wrong["tested"] is False and math.isnan(wrong["estimate"])
    with pytest.raises(ValueError):
        H.resolve_h1_contrast(["ge2", "eq1"], ("ge2", "noline"))


def test_h3_clustered_whenever_a_patient_contributes_two_pairs():
    """PROTOCOL v0.4: the clustered analysis applies whenever any patient has two pairs, whatever the rung."""
    df = h3_frame(n_b=40, n_u=20, p1_abst=range(0, 3), p2_abst=range(0, 40), drop_p1=(), missing_p2=(), two_b_patients=20)
    assert H.max_pairs_per_patient(df) == 2
    for rung in ("n80", "standard"):
        r = H.h3(df, "m1", rung, n_boot=300)
        assert r["clustered_primary"] is True and r["max_pairs_per_patient"] == 2
        assert r["p_method"].startswith("clustered BCa") and r["ci_role"] == "primary"
        assert r["mcnemar_check"]["n"] == 40
        ref = H.h3(df, "m1", "two_per_patient", n_boot=300)
        assert r["p_value"] == ref["p_value"] and r["mcnemar_check"] == ref["mcnemar_check"]
    one = H.h3(h3_frame(), "m1", "n80", n_boot=50)
    assert H.max_pairs_per_patient(h3_frame()) == 1
    assert one["clustered_primary"] is False and one["p_method"] == "exact McNemar"
