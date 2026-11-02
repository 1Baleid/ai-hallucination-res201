"""Tests for cxrdev.arm1 (detector-gated answers and controls, PROTOCOL section 7). Synthetic data only."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from cxrdev import arm1 as A1
from cxrdev import constants as C
from cxrdev import metrics as M
from scripts import synthetic_dryrun as SD


# ----------------------------------------------------------------------------- the fusion rule
def test_expert_logodds_and_fuse():
    assert A1.expert_logodds([0.5])[0] == pytest.approx(0.0)
    assert A1.expert_logodds([0.75])[0] == pytest.approx(math.log(3.0))
    lo = A1.expert_logodds([0.0, 1.0])
    assert lo[0] == pytest.approx(-math.log((1 - 1e-6) / 1e-6)) and lo[1] == pytest.approx(-lo[0])
    assert math.isnan(A1.expert_logodds([float("nan")])[0])
    f = A1.fuse([0.5, -0.5, 0.2], [0.9, 0.1, 1.0], beta=1.0)
    assert f[0] == pytest.approx(0.5 + math.log(9.0)) and f[1] == pytest.approx(-0.5 - math.log(9.0))
    assert f[2] == pytest.approx(0.2 + C.GAMMA)  # clipped at gamma
    assert A1.fuse([0.3], [0.99], beta=2.0, gamma=1.0)[0] == pytest.approx(2.3)
    z = np.array([0.1, -2.0])
    assert np.array_equal(A1.fuse(z, [float("nan"), 0.3], beta=0.0), z)  # beta = 0 is the logit rule
    with pytest.raises(ValueError):
        A1.fuse([1.0, 2.0], [0.5], beta=1.0)


def test_balanced_accuracy():
    yes = np.array([True, True, False, False])
    y = np.array([1, 1, 1, 0])
    assert A1.balanced_accuracy(yes, y) == pytest.approx((2 / 3 + 1.0) / 2)
    assert A1.balanced_accuracy(yes, y, no=np.zeros(4, dtype=bool)) == pytest.approx((2 / 3 + 0.0) / 2)  # abstention
    assert math.isnan(A1.balanced_accuracy([True, False], [1, 1]))
    with pytest.raises(ValueError):
        A1.balanced_accuracy([True], [2])
    with pytest.raises(ValueError):
        A1.balanced_accuracy([True, False], [1])


def test_youden_tau_matches_brute_force_and_tie_rule():
    rng = np.random.default_rng(C.derive_seed("model"))
    for _ in range(5):
        y = rng.integers(0, 2, 60)
        y[:2] = [0, 1]
        s = np.round(rng.uniform(0, 1, 60), 2)  # ties on purpose
        tau = A1.youden_tau(s, y)
        u = np.unique(s)
        cands = (u[:-1] + u[1:]) / 2
        j = [np.mean(s[y == 1] > c) - np.mean(s[y == 0] > c) for c in cands]
        assert tau in cands
        assert np.mean(s[y == 1] > tau) - np.mean(s[y == 0] > tau) == pytest.approx(max(j))
    # J ties at 0.2 and 0.75: the candidate closest to 0.5 wins
    assert A1.youden_tau([0.3, 0.9, 0.1, 0.6], [1, 1, 0, 0]) == pytest.approx(0.75)
    assert A1.youden_tau([0.4, 0.4], [1, 0]) == 0.4
    with pytest.raises(ValueError):
        A1.youden_tau([0.2, 0.3], [1, 1])
    with pytest.raises(ValueError):
        A1.youden_tau([0.2, float("nan")], [1, 0])


def test_select_beta_dev():
    z = np.array([-0.8] * 5 + [1.5] * 5)
    s = np.array([0.75] * 5 + [0.25] * 5)
    y = np.array([1] * 5 + [0] * 5)
    assert A1.select_beta_dev(z, s, y) == 2.0
    big = np.array([20.0, -20.0] * 5)
    yb = np.array([1, 0] * 5)
    sb = np.full(10, 0.5)
    assert A1.select_beta_dev(big, sb, yb) == 1.0  # every beta ties: closest to 1
    assert A1.select_beta_dev(big, sb, yb, grid=(2.0, 0.5)) == 0.5
    with pytest.raises(ValueError):
        A1.select_beta_dev(big, sb, np.ones(10, dtype=int))


# ----------------------------------------------------------------------------- shuffled expert
def _informative(n: int = 200, seed: int = 1):
    rng = np.random.default_rng([C.derive_seed("model"), seed])
    y = np.r_[np.ones(n // 2, dtype=int), np.zeros(n // 2, dtype=int)]
    z = rng.normal(np.where(y == 1, 0.3, -0.3), 1.0)
    s = 1 / (1 + np.exp(-rng.normal(np.where(y == 1, 1.2, -1.2), 1.0)))
    return z, s, y


def test_shuffled_expert_informative_and_constant():
    z, s, y = _informative()
    r = A1.shuffled_expert_test(z, s, y, n_perm=499, seed=7)
    assert r["observed_gain"] == pytest.approx(A1.balanced_accuracy(A1.fuse(z, s, 1.0) > 0, y) - A1.balanced_accuracy(z > 0, y))
    assert r["observed_gain"] > 0.05 and r["p_value"] <= 0.01
    assert (r["p_value"] * (1 + 499)) == pytest.approx(round(r["p_value"] * 500))
    assert r["null_q025"] <= r["null_q500"] <= r["null_q975"] and r["n"] == 200 and r["n_pos"] == 100
    again = A1.shuffled_expert_test(z, s, y, n_perm=499, seed=7)
    assert again == r
    other = A1.shuffled_expert_test(z, s, y, n_perm=499, seed=8)
    assert other["null_mean"] != r["null_mean"]
    const = A1.shuffled_expert_test(z, np.full(200, 0.8), y, n_perm=99, seed=7)
    # every permutation reproduces the observed gain
    assert const["p_value"] == 1.0 and const["null_sd"] == pytest.approx(0.0, abs=1e-12)
    assert const["null_mean"] == pytest.approx(const["observed_gain"], abs=1e-12)


def test_shuffled_expert_null_matches_independent_simulation():
    z, s, y = _informative(n=120, seed=2)
    r = A1.shuffled_expert_test(z, s, y, n_perm=1500, seed=11)
    rng = np.random.default_rng(12345)
    ba0 = A1.balanced_accuracy(z > 0, y)
    sims = [A1.balanced_accuracy(A1.fuse(z, rng.permutation(s), 1.0) > 0, y) - ba0 for _ in range(1500)]
    assert r["null_mean"] == pytest.approx(np.mean(sims), abs=0.006)
    assert r["null_sd"] == pytest.approx(np.std(sims, ddof=1), rel=0.12)


def test_shuffled_expert_noise_is_not_significant():
    z, _, y = _informative(seed=3)
    rng = np.random.default_rng([C.derive_seed("model"), 99])
    noise = rng.uniform(0.05, 0.95, z.size)
    r = A1.shuffled_expert_test(z, noise, y, n_perm=499, seed=7)
    assert r["p_value"] > 0.05


def test_shuffled_expert_input_checks():
    with pytest.raises(ValueError):
        A1.shuffled_expert_test([0.1, float("nan")], [0.5, 0.5], [1, 0], n_perm=9)
    with pytest.raises(ValueError):
        A1.shuffled_expert_test([0.1, 0.2], [0.5, 0.5], [1, 1], n_perm=9)
    with pytest.raises(ValueError):
        A1.shuffled_expert_test([0.1, 0.2], [0.5, 0.5], [1, 0], n_perm=0)


# ----------------------------------------------------------------------------- constant bias, disagreement, oracle
def test_constant_bias_curve_and_matched_bias():
    z, _, y = _informative(n=100, seed=4)
    curve = A1.constant_bias_curve(z, y)
    assert list(curve.columns) == ["b", "yes_rate", "ba"] and len(curve) == 401
    assert curve["yes_rate"].is_monotonic_increasing
    for i in (0, 150, 200, 400):
        b = curve["b"].iloc[i]
        assert curve["ba"].iloc[i] == pytest.approx(A1.balanced_accuracy(z + b > 0, y))
        assert curve["yes_rate"].iloc[i] == pytest.approx(np.mean(z + b > 0))
    zz, yy = np.array([-1.0, 1.0]), np.array([0, 1])
    m = A1.matched_bias(zz, yy, 0.5)
    assert m["b"] == 0.0 and m["yes_rate"] == 0.5 and m["ba"] == 1.0 and m["abs_diff"] == 0.0
    assert A1.matched_bias(zz, yy, 0.3)["b"] == 0.0
    m2 = A1.matched_bias(zz, yy, 0.9, b_grid=[-3.0, 1.5, 3.0])
    assert m2["b"] == 1.5 and m2["yes_rate"] == 1.0


def test_disagreement_abstain_and_oracle_rows():
    d = A1.disagreement_abstain([1, -1, 1, -1], [0.9, 0.9, 0.1, 0.1], [1, 1, 0, 0], tau=0.5)
    assert d["n"] == 4 and d["n_answered"] == 2
    assert d["coverage"]["estimate"] == 0.5 and d["risk"]["estimate"] == 0.0 and d["tau"] == 0.5
    d2 = A1.disagreement_abstain([1, 1], [0.9, 0.9], [0, 1], tau=0.5)
    assert (d2["risk"]["k"], d2["risk"]["n"]) == (1, 2)
    o = A1.oracle_rows([6.0, -6.0, 1.0, -1.0], [0, 1, 1, 0])
    assert o["oracle_label_ba"] == 1.0 and o["clipped_oracle_ba"] == pytest.approx(0.5)
    assert o["share_abs_z_gt_gamma"] == 0.5 and o["n"] == 4


def test_join_and_read_detector(tmp_path):
    df = pd.DataFrame({"item_id": ["a", "b", "c"], "x": [1, 2, 3]})
    det = pd.DataFrame({"item_id": ["a", "b"], "s": [0.2, 0.9]})
    j = A1.join_detector(df, det)
    assert j["s"].tolist()[:2] == [0.2, 0.9] and math.isnan(j["s"].iloc[2])
    with pytest.raises(ValueError, match="more than once"):
        A1.join_detector(df, pd.concat([det, det]))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        A1.join_detector(df, det.assign(s=[0.2, 1.5]))
    with pytest.raises(ValueError, match="already has"):
        A1.join_detector(j, det)
    path = tmp_path / "detector_scores.csv"
    full = det.assign(nih_filename=["x.png", "y.png"], logit_with=[1.0, 2.0], logit_without=[0.0, 0.0],
                      model_id="google/medsiglip-448", revision="r")
    full.to_csv(path, index=False)
    back = A1.read_detector(path)
    assert back["s"].tolist() == [0.2, 0.9] and back["item_id"].tolist() == ["a", "b"]


# ----------------------------------------------------------------------------- run_arm1 on synthetic data
@pytest.fixture(scope="module")
def synth():
    items = SD.make_items()
    parsed, _ = SD.make_parsed(items)
    frame = M.scoring_frame(parsed, items)
    return {"items": items, "frame": frame, "det": SD.make_detector(items, True), "noise": SD.make_detector(items, False)}


def test_run_arm1_rows_gain_and_controls(synth):
    jf = A1.join_detector(synth["frame"], synth["det"])
    r = A1.run_arm1(jf, jf, "qwen3vl2b", n_perm=199, n_boot=99)
    ids = [row["row"] for row in r["rows"]]
    assert ids == ["greedy_text", "beta0", "fused_beta1", "dev_beta", "detector_alone", "shuffled_null", "matched_bias",
                   "disagree_abstain", "oracle_label", "clipped_oracle"]
    rows = {row["row"]: row for row in r["rows"]}
    t = M.rows_for(jf, "qwen3vl2b", "P0", condition="image")
    ab = t["arm"].isin(["A", "B"]).to_numpy()
    z, s, y = t["z"].to_numpy(), t["s"].to_numpy(), t["y"].to_numpy()
    ba0 = A1.balanced_accuracy(z[ab] > 0, y[ab])
    ba1 = A1.balanced_accuracy(A1.fuse(z[ab], s[ab], 1.0) > 0, y[ab])
    assert rows["beta0"]["ba"]["estimate"] == pytest.approx(ba0)
    assert rows["fused_beta1"]["ba"]["estimate"] == pytest.approx(ba1)
    assert r["gain"]["estimate"] == pytest.approx(ba1 - ba0) == pytest.approx(r["shuffled_expert"]["observed_gain"])
    assert r["gain"]["estimate"] > 0.03 and r["shuffled_expert"]["p_value"] <= 0.02
    assert rows["fused_beta1"]["p_value"] == r["shuffled_expert"]["p_value"]
    assert (r["n_A"], r["n_B"], r["n_U"]) == (120, 120, 120) and rows["beta0"]["n"] == 360
    # greedy text vs beta = 0: the planted flipped-z items are the only disagreements (no abstentions under P0)
    ag = r["agreement_greedy_beta0"]
    assert ag["n"] - ag["k"] == SD.N_FLIPPED_Z
    # matched constant bias reproduces the fused yes-rate on A and B
    fused_yr = np.mean(A1.fuse(z[ab], s[ab], 1.0) > 0)
    assert abs(np.mean(z[ab] + r["matched_bias"]["b"] > 0) - fused_yr) <= 0.02
    assert "fused_minus_matched" in rows["matched_bias"]
    # detector-alone and the U arm scored with the same rule
    u = t["arm"].to_numpy() == "U"
    assert rows["detector_alone"]["FAR_U"]["k"] == int(np.sum(s[u] > r["tau"]))
    assert rows["fused_beta1"]["FAR_U"]["k"] == int(np.sum(A1.fuse(z[u], s[u], 1.0) > 0))
    dev = jf.loc[(jf["arm"] == "DEV") & (jf["model_key"] == "qwen3vl2b")]
    assert r["tau"] == pytest.approx(A1.youden_tau(dev["s"], dev["y"]))
    assert r["beta_dev"] in C.BETA_GRID
    assert rows["disagree_abstain"]["coverage"]["n"] == 360 and 0 < rows["disagree_abstain"]["coverage"]["estimate"] < 1
    assert rows["oracle_label"]["ba"]["estimate"] == 1.0
    assert rows["clipped_oracle"]["ba"]["estimate"] == pytest.approx(r["oracle"]["clipped_oracle_ba"])


def test_run_arm1_noise_detector_gives_no_gain(synth):
    jn = A1.join_detector(synth["frame"], synth["noise"])
    r = A1.run_arm1(jn, jn, "qwen3vl2b", n_perm=199, n_boot=0)
    assert abs(r["gain"]["estimate"]) < 0.05
    assert r["shuffled_expert"]["p_value"] > 0.05


def test_run_arm1_without_dev_and_with_missing_scores(synth):
    det = synth["det"]
    test_ids = synth["items"].loc[synth["items"]["arm"] == "B", "item_id"].head(5)
    jf = A1.join_detector(synth["frame"], det.loc[~det["item_id"].isin(test_ids)])
    r = A1.run_arm1(jf, None, "medgemma15", n_perm=49, n_boot=0, with_oracle=False)
    ids = [row["row"] for row in r["rows"]]
    assert ids == ["greedy_text", "beta0", "fused_beta1", "shuffled_null", "matched_bias"]
    assert r["skipped"] and r["tau"] is None and r["beta_dev"] is None and r["dev"] is None
    assert r["n_dropped_no_score"] == 5 and r["n_B"] == 115
    r2 = A1.run_arm1(jf, jf, "medgemma15", n_perm=49, n_boot=0, exclusions=list(test_ids[:2]) + ["nonexistent"])
    assert r2["n_retained"] == 358 and r2["n_dropped_no_score"] == 3
    with pytest.raises(ValueError, match="join it"):
        A1.run_arm1(synth["frame"], None, "medgemma15", n_perm=9, n_boot=0)
    empty = A1.run_arm1(jf, jf, "nobody", n_perm=9, n_boot=0)
    assert empty["skipped_reason"]


def test_detector_summary(synth):
    items, det = synth["items"], synth["det"]
    out = A1.detector_summary(items, det, exclusions=None, n_boot=99)
    dev = items.loc[items["arm"] == "DEV"].merge(det[["item_id", "s"]], on="item_id")
    assert out["dev_auroc"]["estimate"] == pytest.approx(roc_auc_score(dev["ett"], dev["s"]))
    assert 0.75 < out["dev_auroc"]["estimate"] < 0.93  # planted about 0.85
    assert out["tau"] == pytest.approx(A1.youden_tau(dev["s"], dev["ett"]))
    a = items.loc[items["arm"] == "A"].merge(det[["item_id", "s"]], on="item_id")
    assert out["agreement"]["A"]["k"] == int(np.sum(a["s"] > out["tau"])) and out["agreement"]["A"]["n"] == 120
    ex = A1.detector_summary(items, det, exclusions=list(a["item_id"].head(3)), n_boot=0)
    assert ex["agreement"]["A"]["n"] == 117 and out["n_missing"] == 0
    none = A1.detector_summary(items.loc[items["arm"] != "DEV"], det, n_boot=0)
    assert none["dev_auroc"] is None and none["skipped_reason"]
