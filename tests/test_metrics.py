"""Tests for cxrdev.metrics (PROTOCOL sections 3, 5 and 6). Synthetic data only."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from cxrdev import constants as C
from cxrdev import fileio, parse
from cxrdev import metrics as M
from cxrdev.items import ITEM_COLS


# ----------------------------------------------------------------------------- builders
def make_items(specs: list[tuple]) -> pd.DataFrame:
    """S5 rows from (item_id, arm, pair_id, stratum, patient, n_other_lines)."""
    rows = []
    for item_id, arm, pair_id, stratum, patient, n_other in specs:
        ett = 1 if arm == "A" or (arm == "DEV" and stratum == "pos") else 0
        rows.append({
            "item_id": item_id, "uid": f"synthetic.{item_id}", "nih_filename": f"{patient:08d}_{item_id}.png",
            "nih_patient_id": patient, "arm": arm, "pair_id": pair_id, "stratum": stratum, "ett": ett,
            "ngt": int(n_other >= 1), "ngt_incomplete": 0, "cvc": int(n_other >= 2), "swan": int(n_other >= 3),
            "n_other_lines": n_other, "view": "AP", "followup": 0, "age": 50, "sex": "F",
            "direction": "B_after_A" if pair_id else "", "gap": 1 if pair_id else -1, "rung": "standard", "smoke": 0,
        })
    return pd.DataFrame(rows, columns=list(ITEM_COLS))


def make_parsed(outputs: list[tuple], items: pd.DataFrame, condition: str = "image") -> pd.DataFrame:
    """S8 rows from (model, prompt, item_id, raw_output, truncated, z)."""
    fname = dict(zip(items["item_id"], items["nih_filename"]))
    rows = []
    for model, prompt, item_id, raw, truncated, z in outputs:
        rows.append({
            "run_id": "r", "model_key": model, "model_id": "synthetic/model", "revision": "rev", "transformers_version": "t",
            "torch_version": "t", "dtype": "bfloat16", "device": "cpu", "item_id": item_id,
            "nih_filename": fname.get(item_id, ""), "image_sha256": "", "condition": condition, "prompt_id": prompt,
            "prompt_sha256": "", "prompt_text": "q", "visual_tokens": 1, "seed": 1, "max_new_tokens": 16, "raw_output": raw,
            "n_generated_tokens": 16 if truncated else 2, "ended_with_eos": not truncated, "parsed": parse.parse_answer(raw),
            "parser_version": parse.PARSER_VERSION, "logp_yes": -0.5, "logp_no": -1.0, "logp_abstain": float("nan"),
            "z": z, "coverage": 0.95, "signal": "first_token", "wall_s": 0.3, "timestamp_utc": "2026-11-11T00:00:00Z",
            "truncated": truncated,
        })
    return M.coerce_parsed(pd.DataFrame(rows, columns=list(M.S8_COLUMNS)))


SPECS = [
    ("a1", "A", "p001", "", 1, 0), ("b1", "B", "p001", "", 1, 1),
    ("a2", "A", "p002", "", 2, 0), ("b2", "B", "p002", "", 2, 0),
    ("a3", "A", "p003", "", 3, 1), ("b3", "B", "p003", "", 3, 2),
    ("a4", "A", "p004", "", 4, 0), ("b4", "B", "p004", "", 4, 0),
    ("u1", "U", "", "line", 11, 1), ("u2", "U", "", "line", 12, 2),
    ("u3", "U", "", "noline", 13, 0), ("u4", "U", "", "noline", 14, 0),
]
OUTPUTS = [
    ("m1", "P0", "a1", "Yes", False, 2.0), ("m1", "P0", "b1", "No.", False, -1.0),
    ("m1", "P0", "a2", "yes", False, 1.5), ("m1", "P0", "b2", "Yes", False, 0.5),
    ("m1", "P0", "a3", "No", False, -0.5), ("m1", "P0", "b3", "I don't know.", False, 0.1),
    ("m1", "P0", "a4", "Yes, the tube is visible and its tip is", True, 0.8), ("m1", "P0", "b4", "Hmm, maybe.", False, -0.2),
    ("m1", "P0", "u1", "Yes", False, 1.0), ("m1", "P0", "u2", "No", False, -1.0),
    ("m1", "P0", "u3", "No", False, -2.0), ("m1", "P0", "u4", "Yes", False, 0.3),
]


@pytest.fixture()
def toy() -> pd.DataFrame:
    items = make_items(SPECS)
    return M.scoring_frame(make_parsed(OUTPUTS, items), items)


# ----------------------------------------------------------------------------- small helpers
def test_rate_and_er_cost():
    r = M.rate(3, 10)
    assert r["estimate"] == 0.3 and r["k"] == 3 and r["n"] == 10 and r["method"] == "wilson"
    assert r["lo"] < 0.3 < r["hi"]
    empty = M.rate(0, 0)
    assert math.isnan(empty["estimate"]) and math.isnan(empty["lo"]) and empty["n"] == 0
    assert M.er_cost(0.9) == 9.0 and M.er_cost(0.5) == 1.0
    for bad in (0.0, 1.0, -0.1):
        with pytest.raises(ValueError):
            M.er_cost(bad)


def test_ordering_helpers():
    assert M.ordered_models(["zz", "medgemma15", "chexone", "qwen3vl2b"]) == ["qwen3vl2b", "medgemma15", "chexone", "zz"]
    assert M.ordered_prompts(["P2hi", "P9", "P0", "P1", "P0"]) == ["P0", "P1", "P2hi", "P9"]


def test_h1_contrast_selection():
    assert M.h1_contrast(["line", "noline"]) == ("line", "noline", False)
    assert M.h1_contrast(["ge2", "eq1", "noline"]) == ("ge2", "eq1", True)
    assert M.h1_contrast(["line"]) is None


# ----------------------------------------------------------------------------- the H0 gate
def test_h0_gate_order_and_boundaries():
    # NO-GO comes first, even when the GO conditions also hold
    assert M.h0_gate(0.59, 0.95, 0.5) == "NO-GO"
    assert M.h0_gate(0.95, 0.95, 0.095) == "NO-GO"
    assert M.h0_gate(0.95, 0.95, 0.91) == "NO-GO"
    # GO thresholds are inclusive
    assert M.h0_gate(0.70, 0.70, 0.25) == "GO"
    assert M.h0_gate(0.70, 0.70, 0.75) == "GO"
    assert M.h0_gate(0.60, 0.99, 0.5) == "CONDITIONAL"
    assert M.h0_gate(0.80, 0.69, 0.5) == "CONDITIONAL"
    assert M.h0_gate(0.80, 0.80, 0.80) == "CONDITIONAL"
    assert M.h0_gate(0.80, 0.80, 0.10) == "CONDITIONAL"
    # floating-point error at a threshold does not flip the class
    assert M.h0_gate((84 / 120 + 0.7) / 2, 0.7, 0.3) == "GO"
    assert M.h0_gate(0.7 - 1e-15, 0.7, 0.3) == "GO"
    with pytest.raises(ValueError):
        M.h0_gate(float("nan"), 0.7, 0.5)


# ----------------------------------------------------------------------------- reading S8
def test_read_parsed_roundtrip_keeps_text_verbatim(tmp_path):
    items = make_items(SPECS[:4])
    tricky = ["None", "NA", "", "a,b\n\"c\"", "  Yes  "]
    outs = [("m1", "P0", iid, raw, False, z) for iid, raw, z in zip(["a1", "b1", "a2", "b2"], tricky, [1.0, np.nan, -1, 0])]
    df = make_parsed(outs, items)
    df.loc[1, "logp_abstain"] = float("nan")
    path = tmp_path / "parsed_results.csv"
    fileio.write_csv(df, path)
    back = M.read_parsed(path)
    assert list(back.columns) == list(M.S8_COLUMNS)
    assert back["raw_output"].tolist() == tricky[:4]
    assert back["nih_filename"].tolist() == df["nih_filename"].tolist()
    assert back["ended_with_eos"].dtype == bool and back["truncated"].dtype == bool
    assert back["visual_tokens"].dtype == np.int64
    assert math.isnan(back.loc[1, "z"]) and back.loc[0, "z"] == 1.0
    assert back["parser_version"].tolist() == [parse.PARSER_VERSION] * 4


def test_coerce_parsed_booleans():
    items = make_items(SPECS[:2])
    df = make_parsed([("m1", "P0", "a1", "Yes", False, 1.0), ("m1", "P0", "b1", "No", True, -1.0)], items)
    df["truncated"] = ["false", "1"]
    df["ended_with_eos"] = [1, 0]
    out = M.coerce_parsed(df)
    assert out["truncated"].tolist() == [False, True] and out["ended_with_eos"].tolist() == [True, False]
    df["truncated"] = ["maybe", "0"]
    with pytest.raises(ValueError, match="non-boolean"):
        M.coerce_parsed(df)
    with pytest.raises(ValueError, match="missing"):
        M.coerce_parsed(df.drop(columns=["z"]))


def test_read_exclusions_rejects_unknown_reason(tmp_path):
    p = tmp_path / "exclusions.csv"
    p.write_text("item_id,reason\na1,skip\nb1,flag_ambiguous\n", encoding="utf-8")
    assert M.read_exclusions(p)["reason"].tolist() == ["skip", "flag_ambiguous"]
    p.write_text("item_id,reason\na1,model_said_so\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown exclusion reason"):
        M.read_exclusions(p)


# ----------------------------------------------------------------------------- scoring frame
def test_scoring_frame_join_and_validity(toy):
    assert len(toy) == 12
    row = toy.set_index("item_id")
    assert row.loc["a1", "arm"] == "A" and row.loc["a1", "y"] == 1 and row.loc["b1", "y"] == 0
    assert row.loc["a1", "set"] == "pair" and row.loc["u1", "set"] == "U"
    assert not row.loc["a4", "valid"]  # truncated, although it parses as yes
    assert row.loc["a4", "parsed"] == "yes"
    assert not row.loc["b4", "valid"] and row.loc["b4", "parsed"] == "unparsable"
    assert row.loc["b3", "valid"] and row.loc["b3", "parsed"] == "abstain"
    assert "uid" not in toy.columns  # the RANZCR uid is never carried into scoring


def test_scoring_frame_rejects_inconsistent_inputs():
    items = make_items(SPECS)
    parsed = make_parsed(OUTPUTS, items)
    with pytest.raises(ValueError, match="not in the item list"):
        M.scoring_frame(parsed, items.iloc[1:])
    bad_name = parsed.copy()
    bad_name.loc[0, "nih_filename"] = "99999999_999.png"
    with pytest.raises(ValueError, match="NIH filename"):
        M.scoring_frame(bad_name, items)
    bad_label = items.copy()
    bad_label.loc[bad_label["item_id"] == "b1", "ett"] = 1
    with pytest.raises(ValueError, match="contradicts"):
        M.scoring_frame(parsed, bad_label)
    dup = pd.concat([parsed, parsed.iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicated"):
        M.scoring_frame(dup, items)
    with pytest.raises(ValueError, match="unknown condition"):
        M.scoring_frame(parsed, items, "video")


def test_effective_answers(toy):
    sub = toy.set_index("item_id").loc[["a3", "a4", "b3", "b4"]].reset_index()
    assert M.effective_answers(sub).tolist() == ["no", None, "abstain", None]
    assert M.effective_answers(sub, unparsable_as_wrong=True).tolist() == ["no", "no", "abstain", "yes"]


# ----------------------------------------------------------------------------- summarise
def test_summarise_hand_computed(toy):
    s = M.summarise(toy, "m1", "P0", n_boot=50)
    assert (s["hit_A"]["k"], s["hit_A"]["n"]) == (2, 3)
    assert (s["FAR_B"]["k"], s["FAR_B"]["n"]) == (1, 3)
    assert (s["FAR_U"]["k"], s["FAR_U"]["n"]) == (2, 4)
    assert (s["FAR_U_line"]["k"], s["FAR_U_line"]["n"]) == (1, 2)
    assert (s["FAR_U_noline"]["k"], s["FAR_U_noline"]["n"]) == (1, 2)
    assert (s["FAR_B_line"]["k"], s["FAR_B_line"]["n"]) == (0, 2)  # b1 (no), b3 (abstain)
    assert (s["FAR_B_noline"]["k"], s["FAR_B_noline"]["n"]) == (1, 1)  # b2 (yes); b4 unparsable
    assert s["ba"]["estimate"] == pytest.approx(0.5)
    assert s["ba"]["n"] == 6
    assert (s["yes_rate"]["k"], s["yes_rate"]["n"]) == (5, 10)
    assert (s["abstention"]["k"], s["abstention"]["n"]) == (1, 10)
    assert (s["abstention_B"]["k"], s["abstention_B"]["n"]) == (1, 3)
    assert (s["accuracy"]["k"], s["accuracy"]["n"]) == (5, 10)
    assert (s["answered"]["k"], s["answered"]["n"]) == (9, 10)
    assert (s["risk"]["k"], s["risk"]["n"]) == (4, 9)
    assert s["ER_0.9"]["estimate"] == pytest.approx((5 - 9 * 4) / 10)
    assert s["ER_0.5"]["estimate"] == pytest.approx((5 - 4) / 10)
    assert s["ER_0.9"]["c"] == 9.0 and s["ER_0.9"]["n"] == 10
    assert (s["persistence"]["k"], s["persistence"]["n"]) == (1, 2)
    assert (s["pair_correct"]["k"], s["pair_correct"]["n"]) == (1, 3)
    assert (s["unparsable_rate"]["k"], s["unparsable_rate"]["n"]) == (1, 12)
    assert (s["truncated_rate"]["k"], s["truncated_rate"]["n"]) == (1, 12)
    assert (s["excluded_rate"]["k"], s["excluded_rate"]["n"]) == (2, 12)
    assert (s["unparsable_rate_B"]["k"], s["truncated_rate_A"]["k"]) == (1, 1)
    assert s["h0_class"] == "NO-GO"
    assert s["flag_unparsable"] is False  # 1/12 is below the 10% flag
    assert s["n_rows"] == 12 and s["n_rows_excluded"] == 0
    assert s["signal"] == "first_token" and s["coverage_median"] == pytest.approx(0.95)


def test_summarise_auroc_uses_every_finite_z(toy):
    s = M.summarise(toy, "m1", "P0", n_boot=60)
    z = dict(zip(toy["item_id"], toy["z"]))
    a = [z[i] for i in ("a1", "a2", "a3", "a4")]
    b = [z[i] for i in ("b1", "b2", "b3", "b4")]
    u = [z[i] for i in ("u1", "u2", "u3", "u4")]
    assert s["auroc_AB"]["estimate"] == pytest.approx(roc_auc_score([1] * 4 + [0] * 4, a + b))
    assert s["auroc_AU"]["estimate"] == pytest.approx(roc_auc_score([1] * 4 + [0] * 4, a + u))
    assert (s["auroc_AB"]["n_pos"], s["auroc_AB"]["n_neg"]) == (4, 4)
    for key in ("auroc_AB", "ba", "ER_0.9"):
        r = s[key]
        assert r["method"] in ("bca", "percentile")
        assert r["lo"] <= r["estimate"] + 1e-12 and r["estimate"] - 1e-12 <= r["hi"]
    s1 = M.summarise(toy, "m1", "P0", n_boot=0)
    assert s1["ba"]["method"] == "none" and math.isnan(s1["ba"]["lo"]) and s1["ba"]["estimate"] == pytest.approx(0.5)
    assert "auroc_AB" not in M.summarise(toy, "m1", "P0", n_boot=0, auroc_prompts=())


def test_summarise_sensitivity_counts_excluded_outputs_as_wrong(toy):
    s = M.summarise(toy, "m1", "P0", unparsable_as_wrong=True, n_boot=0)
    assert (s["hit_A"]["k"], s["hit_A"]["n"]) == (2, 4)  # truncated A counted as "no"
    assert (s["FAR_B"]["k"], s["FAR_B"]["n"]) == (2, 4)  # unparsable B counted as "yes"
    assert s["ba"]["estimate"] == pytest.approx((0.5 + 0.25) / 2)
    assert (s["pair_correct"]["k"], s["pair_correct"]["n"]) == (1, 4)
    assert (s["persistence"]["k"], s["persistence"]["n"]) == (1, 2)
    # the rates of excluded outputs do not depend on the scoring mode; the gate is not applied
    assert (s["unparsable_rate"]["k"], s["unparsable_rate"]["n"]) == (1, 12)
    assert s["h0_class"] is None and s["unparsable_as_wrong"] is True


def test_exclusions_item_level_and_pair_level(toy):
    s = M.summarise(toy, "m1", "P0", exclusions=pd.DataFrame({"item_id": ["b2"], "reason": ["skip"]}), n_boot=0)
    assert (s["FAR_B"]["k"], s["FAR_B"]["n"]) == (0, 2)
    assert (s["hit_A"]["k"], s["hit_A"]["n"]) == (2, 3)  # a2 stays for item-level metrics
    assert (s["persistence"]["k"], s["persistence"]["n"]) == (0, 1)  # pair p002 is dropped
    assert (s["pair_correct"]["k"], s["pair_correct"]["n"]) == (1, 2)
    assert s["n_rows"] == 11 and s["n_rows_excluded"] == 1
    s2 = M.summarise(toy, "m1", "P0", exclusions=["a1"], n_boot=0)
    assert (s2["hit_A"]["k"], s2["hit_A"]["n"]) == (1, 2)
    assert (s2["persistence"]["k"], s2["persistence"]["n"]) == (1, 1)
    assert M.apply_exclusions(toy, None) is toy


def test_flag_unparsable_over_ten_percent():
    specs = [(f"a{i}", "A", f"p{i:03d}", "", i, 0) for i in range(10)] + \
            [(f"b{i}", "B", f"p{i:03d}", "", i, 0) for i in range(10)]
    items = make_items(specs)
    outs = [("m1", "P0", f"a{i}", "Yes" if i > 2 else "the tube", False, 1.0) for i in range(10)] + \
           [("m1", "P0", f"b{i}", "No", False, -1.0) for i in range(10)]
    s = M.summarise(M.scoring_frame(make_parsed(outs, items), items), "m1", "P0", n_boot=0)
    assert s["unparsable_rate"]["k"] == 3 and s["flag_unparsable"] is True
    s1 = M.summarise(M.scoring_frame(make_parsed([(m, "P1", i, r, t, z) for m, _, i, r, t, z in outs], items), items),
                     "m1", "P1", n_boot=0)
    assert s1["flag_unparsable"] is False and "h0_class" not in s1  # flag and gate are P0 only


# ----------------------------------------------------------------------------- blind rows and risk-coverage
def test_blind_rows():
    items = make_items(SPECS[:2])
    blind = make_parsed([("m1", "P1", "BLIND-TEXT", "I don't know", False, -0.4),
                         ("m1", "P0", "BLIND-GREY", "No", False, -2.0),
                         ("m1", "P0", "BLIND-TEXT", "Yes", False, 1.0),
                         ("m2", "P0", "BLIND-TEXT", "Yes", False, 3.0)], items)
    blind["condition"] = ["BLIND-text", "BLIND-grey", "BLIND-text", "BLIND-text"]
    b = M.blind_rows(blind, "m1")
    assert list(b.columns) == ["model_key", "prompt_id", "condition", "z", "p_yes_norm", "coverage", "greedy", "truncated"]
    assert b[["prompt_id", "condition"]].values.tolist() == [["P0", "BLIND-text"], ["P0", "BLIND-grey"], ["P1", "BLIND-text"]]
    assert b["p_yes_norm"].tolist() == pytest.approx([1 / (1 + math.exp(-1.0)), 1 / (1 + math.exp(2.0)), 1 / (1 + math.exp(0.4))])
    assert b["greedy"].tolist() == ["yes", "no", "abstain"]
    dup = pd.concat([blind, blind.iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError, match="more than once"):
        M.blind_rows(dup, "m1")
    assert M.blind_rows(blind, "nobody").empty


def test_risk_coverage(toy):
    rc = M.risk_coverage(toy, "m1", "P0")
    z = toy["z"].to_numpy()
    y = toy["y"].to_numpy()
    for r in rc.itertuples(index=False):
        m = np.abs(z) >= r.threshold
        assert r.n_answered == m.sum()
        assert r.coverage == pytest.approx(m.mean())
        assert r.risk == pytest.approx(np.mean((z[m] > 0).astype(int) != y[m]))
    assert rc["coverage"].iloc[-1] == 1.0
    assert rc["threshold"].is_monotonic_decreasing
    # ties in |z| share one row (u2 and b1 both have |z| = 1.0)
    assert (rc["threshold"] == 1.0).sum() == 1
    assert M.risk_coverage(toy, "nobody").empty


def test_rows_for_filters_and_excludes(toy):
    sub = M.rows_for(toy, "m1", "P0", ("U",), exclusions=["u1"])
    assert sorted(sub["item_id"]) == ["u2", "u3", "u4"]
    assert M.rows_for(toy, "m1", "P1").empty


def test_summarise_design_effect_and_stratified_bootstrap_runs(toy):
    s = M.summarise(toy, "m1", "P0", n_boot=40, seed=C.derive_seed("bootstrap"))
    de = s["design_effect"]
    # patient 4 has no valid output (truncated A, unparsable B): 7 clusters hold the 10 valid items
    assert de["n_clusters"] == 7 and de["mean_cluster_size"] == pytest.approx(10 / 7)
    assert s["ER_0.9"]["n_clusters"] == 7
    again = M.summarise(toy, "m1", "P0", n_boot=40, seed=C.derive_seed("bootstrap"))
    assert again["ER_0.9"]["lo"] == s["ER_0.9"]["lo"] and again["ba"]["hi"] == s["ba"]["hi"]  # deterministic
