"""Tests for cxrdev.stats (PROTOCOL.md sections 5 and 6). Synthetic data only."""
from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd
import pytest
import scipy.stats
from sklearn.metrics import cohen_kappa_score, roc_auc_score

from cxrdev import stats
from cxrdev.constants import derive_seed

SEED = derive_seed("bootstrap")


def _mean_x(d: pd.DataFrame) -> float:
    return float(d["x"].to_numpy().mean())  # numpy on the column: the fast stat_fn pattern


def _singletons(x: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"x": np.asarray(x, dtype=float), "cid": np.arange(len(x))})


# ----------------------------------------------------------------------------- alpha grid
def test_alpha_grid():
    g = stats.ALPHA_GRID
    assert isinstance(g, np.ndarray)
    assert g.size == 9 + 99
    assert g[0] == 0.001 and g[-1] == 0.5
    assert np.all(np.diff(g) > 0)
    np.testing.assert_allclose(g[:10], [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01])
    for a in (0.01, 0.015, 0.05, 0.1, 0.25, 0.495):
        assert np.any(g == a), a
    assert np.array_equal(g, np.round(g, 4))


# ----------------------------------------------------------------------------- Wilson
def test_wilson_reference_values():
    lo, hi = stats.wilson_ci(0, 10)
    assert lo == 0.0
    assert hi == pytest.approx(0.2775, abs=1e-4)
    lo, hi = stats.wilson_ci(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-4)
    assert hi == pytest.approx(0.7634, abs=1e-4)
    lo, hi = stats.wilson_ci(10, 10)
    assert hi == 1.0 and lo == pytest.approx(1 - 0.2775, abs=1e-4)
    assert all(math.isnan(v) for v in stats.wilson_ci(0, 0))


def test_wilson_is_score_test_inversion():
    z = scipy.stats.norm.ppf(0.975)
    for k, n in [(1, 10), (3, 17), (40, 118), (59, 60), (12, 300)]:
        phat = k / n
        for bound in stats.wilson_ci(k, n):
            assert abs(phat - bound) / math.sqrt(bound * (1 - bound) / n) == pytest.approx(z, rel=1e-9)
    # a wider interval at a smaller alpha
    lo90, hi90 = stats.wilson_ci(30, 100, alpha=0.10)
    lo99, hi99 = stats.wilson_ci(30, 100, alpha=0.01)
    assert lo99 < lo90 and hi99 > hi90


def test_wilson_validation():
    with pytest.raises(ValueError):
        stats.wilson_ci(11, 10)
    with pytest.raises(ValueError):
        stats.wilson_ci(-1, 10)
    with pytest.raises(ValueError):
        stats.wilson_ci(1, 10, alpha=0.0)
    with pytest.raises(ValueError):
        stats.wilson_ci(1.5, 10)
    assert stats.wilson_ci(np.int64(5), np.int64(10)) == stats.wilson_ci(5, 10)


def test_wilson_halfwidth():
    assert stats.wilson_halfwidth(0.5, 120) == pytest.approx(0.088, abs=1e-3)   # "about 0.09"
    assert stats.wilson_halfwidth(0.5, 200) == pytest.approx(0.069, abs=1e-3)   # "about 0.07"
    lo, hi = stats.wilson_ci(60, 120)
    assert stats.wilson_halfwidth(0.5, 120) == pytest.approx((hi - lo) / 2, rel=1e-12)
    assert math.isnan(stats.wilson_halfwidth(0.5, 0))
    with pytest.raises(ValueError):
        stats.wilson_halfwidth(1.2, 10)


# ----------------------------------------------------------------------------- McNemar, Holm
def test_mcnemar_exact_matches_binomial():
    for b in range(0, 15):
        for c in range(0, 15):
            p = stats.mcnemar_exact(b, c)
            if b + c == 0:
                assert p == 1.0
            else:
                assert p == pytest.approx(min(1.0, 2 * scipy.stats.binom.cdf(min(b, c), b + c, 0.5)), rel=1e-12)
            assert p == stats.mcnemar_exact(c, b)
            assert 0.0 < p <= 1.0
    assert stats.mcnemar_exact(0, 5) == pytest.approx(0.0625)
    assert stats.mcnemar_exact(10, 10) == 1.0
    assert stats.mcnemar_exact(2, 20) == pytest.approx(scipy.stats.binomtest(2, 22, 0.5).pvalue, rel=1e-9)
    with pytest.raises(ValueError):
        stats.mcnemar_exact(-1, 3)


def test_holm_known_example():
    assert stats.holm([0.01, 0.04, 0.03, 0.005]) == pytest.approx([0.03, 0.06, 0.06, 0.02])
    # NaN stays NaN and is excluded from m
    out = stats.holm([0.01, float("nan"), 0.04])
    assert out[0] == pytest.approx(0.02) and math.isnan(out[1]) and out[2] == pytest.approx(0.04)
    out = stats.holm([None, 0.2])
    assert math.isnan(out[0]) and out[1] == pytest.approx(0.2)
    out = stats.holm(pd.Series([0.03, pd.NA, 0.01], dtype="Float64"))
    assert out[0] == pytest.approx(0.03) and math.isnan(out[1]) and out[2] == pytest.approx(0.02)
    assert all(math.isnan(v) for v in stats.holm([float("nan"), float("nan")]))
    assert stats.holm([]) == []
    assert stats.holm([0.5, 0.6]) == [1.0, 1.0]              # capped at 1
    assert stats.holm([0.02, 0.02, 0.02]) == pytest.approx([0.06, 0.06, 0.06])  # ties
    with pytest.raises(ValueError):
        stats.holm([0.1, 1.5])


def test_holm_monotone_and_not_below_raw():
    rng = np.random.default_rng(SEED)
    p = rng.uniform(0, 0.2, size=12)
    adj = np.array(stats.holm(p))
    order = np.argsort(p)
    assert np.all(np.diff(adj[order]) >= 0)
    assert np.all(adj >= p) and np.all(adj <= 1)


# ----------------------------------------------------------------------------- kappa, AUROC
def test_cohen_kappa_matches_sklearn():
    rng = np.random.default_rng(SEED)
    for n_cat in (2, 3):
        a = rng.integers(0, n_cat, size=200)
        flip = rng.random(200) < 0.25
        b = np.where(flip, rng.integers(0, n_cat, size=200), a)
        assert stats.cohen_kappa(a, b) == pytest.approx(cohen_kappa_score(a, b), abs=1e-12)
    a = ["present", "absent", "present", "present", "absent", "absent", "present"]
    b = ["present", "absent", "absent", "present", "absent", "present", "present"]
    assert stats.cohen_kappa(a, b) == pytest.approx(cohen_kappa_score(a, b), abs=1e-12)
    assert stats.cohen_kappa(a, a) == 1.0


def test_cohen_kappa_degenerate_and_validation():
    assert math.isnan(stats.cohen_kappa(["present"] * 5, ["present"] * 5))  # expected agreement 1
    assert math.isnan(stats.cohen_kappa([], []))
    # one rater constant, the other not: p_e < 1, kappa defined (0 here)
    assert stats.cohen_kappa(["present"] * 4, ["present", "absent", "present", "absent"]) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        stats.cohen_kappa([1, 0], [1])
    with pytest.raises(ValueError):
        stats.cohen_kappa(["present", None], ["present", "absent"])
    with pytest.raises(ValueError):
        stats.cohen_kappa([1.0, float("nan")], [1.0, 0.0])


def test_auroc_matches_sklearn_with_ties():
    rng = np.random.default_rng(SEED)
    pos = rng.integers(0, 6, size=57).astype(float)   # many ties
    neg = rng.integers(0, 5, size=43).astype(float)
    y = np.r_[np.ones(pos.size), np.zeros(neg.size)]
    assert stats.auroc(pos, neg) == pytest.approx(roc_auc_score(y, np.r_[pos, neg]), abs=1e-12)
    pos_c, neg_c = rng.normal(1, 1, 80), rng.normal(0, 1, 70)
    y = np.r_[np.ones(80), np.zeros(70)]
    assert stats.auroc(pos_c, neg_c) == pytest.approx(roc_auc_score(y, np.r_[pos_c, neg_c]), abs=1e-12)


def test_auroc_edges():
    assert stats.auroc([3, 4], [1, 2]) == 1.0
    assert stats.auroc([1, 2], [3, 4]) == 0.0
    assert stats.auroc([1, 1], [1, 1, 1]) == 0.5
    assert stats.auroc([2.0], [1.0, 2.0, 3.0]) == pytest.approx((1 + 0.5) / 3)
    assert stats.auroc([np.inf, 0.0], [-np.inf, 0.0]) == pytest.approx((1 + 1 + 1 + 0.5) / 4)
    assert math.isnan(stats.auroc([], [1.0]))
    assert math.isnan(stats.auroc([1.0], []))
    with pytest.raises(ValueError):
        stats.auroc([np.nan, 1.0], [0.0])


# ----------------------------------------------------------------------------- design effect, power
def test_design_effect_hand_computed():
    # clusters a (1, 1, 0), b (0, 0), c (1, 0, 1, 1): N = 9, k = 3
    # MSB = 29/72, MSW = 17/72, n0 = 26/9 -> ICC = 54/275, deff = 1 + 2 * 54/275 = 383/275
    y = [1, 1, 0, 0, 0, 1, 0, 1, 1]
    cl = ["a", "a", "a", "b", "b", "c", "c", "c", "c"]
    res = stats.design_effect(y, cl)
    assert set(res) >= {"icc", "mean_cluster_size", "deff", "n_clusters"}
    assert res["icc"] == pytest.approx(54 / 275, rel=1e-12)
    assert res["mean_cluster_size"] == pytest.approx(3.0)
    assert res["deff"] == pytest.approx(383 / 275, rel=1e-12)
    assert res["n_clusters"] == 3
    # row order does not matter
    perm = [8, 0, 4, 2, 6, 1, 5, 3, 7]
    res_p = stats.design_effect([y[i] for i in perm], pd.Series([cl[i] for i in perm], index=range(100, 109)))
    assert res_p["icc"] == pytest.approx(res["icc"], rel=1e-12)


def test_design_effect_special_cases():
    perfect = stats.design_effect([1, 1, 0, 0], [1, 1, 2, 2])
    assert perfect["icc"] == pytest.approx(1.0) and perfect["deff"] == pytest.approx(2.0)
    anti = stats.design_effect([1, 0, 1, 0], [1, 1, 2, 2])
    assert anti["icc"] == pytest.approx(-1.0) and anti["deff"] == pytest.approx(0.0)  # raw ICC reported
    single = stats.design_effect([1, 0, 1], [1, 2, 3])
    assert math.isnan(single["icc"]) and single["deff"] == 1.0 and single["mean_cluster_size"] == 1.0
    const = stats.design_effect([1, 1, 1, 1], [1, 1, 2, 2])
    assert math.isnan(const["icc"]) and math.isnan(const["deff"])
    one = stats.design_effect([1, 0, 1], [7, 7, 7])
    assert one["n_clusters"] == 1 and math.isnan(one["icc"])
    with pytest.raises(ValueError):
        stats.design_effect([1, 0], [1])


def test_power_two_proportions_reference_values():
    assert round(stats.power_two_proportions(0.50, 0.70, 60, 60), 3) == 0.611
    assert round(stats.power_two_proportions(0.50, 0.75, 60, 60), 3) == 0.816
    assert round(stats.power_two_proportions(0.50, 0.70, 100, 100), 3) == 0.828
    assert round(stats.power_two_proportions(0.50, 0.75, 40, 40), 2) == 0.64  # n = 80 rung note
    assert stats.power_two_proportions(0.70, 0.50, 60, 60) == pytest.approx(
        stats.power_two_proportions(0.50, 0.70, 60, 60))
    assert stats.power_two_proportions(0.5, 0.5, 60, 60) == pytest.approx(0.05)  # size under H0
    assert stats.power_two_proportions(0.5, 0.7, 200, 200) > stats.power_two_proportions(0.5, 0.7, 100, 100)
    with pytest.raises(ValueError):
        stats.power_two_proportions(0.5, 1.2, 60, 60)
    with pytest.raises(ValueError):
        stats.power_two_proportions(0.5, 0.7, 0, 60)


# ----------------------------------------------------------------------------- cluster bootstrap
def _zero_sum_clusters() -> pd.DataFrame:
    """Clusters of unequal size whose values sum to zero within each cluster."""
    rows = []
    for cid, size in enumerate([1, 2, 3, 4, 2, 3, 1, 5, 2, 4]):
        vals = [size - 1] + [-1] * (size - 1)
        for j, v in enumerate(vals):
            rows.append({"cid": f"c{cid:02d}", "v": float(v), "j": j, "size": size})
    df = pd.DataFrame(rows).sample(frac=1.0, random_state=1)  # rows of a cluster not adjacent
    df["row_id"] = np.arange(len(df))
    df.index = np.arange(1000, 1000 + len(df))  # non-default index: resampling must use positions
    return df


def test_cluster_bootstrap_keeps_clusters_together():
    df = _zero_sum_clusters()
    boot = stats.cluster_bootstrap(df, "cid", lambda d: d["v"].sum(), n_boot=500, seed=SEED)
    assert boot.shape == (500,)
    assert np.all(boot == 0.0)
    # the same statistic detects splitting: resampling rows instead of clusters breaks it
    split = stats.cluster_bootstrap(df, "row_id", lambda d: d["v"].sum(), n_boot=500, seed=SEED)
    assert np.mean(split != 0.0) > 0.5
    # composition varies across replicates (it is really resampling)
    n_distinct = stats.cluster_bootstrap(df, "cid", lambda d: d["cid"].nunique(), n_boot=200, seed=SEED)
    assert n_distinct.max() <= 10 and n_distinct.min() < 10


def test_cluster_bootstrap_draw_col_marks_whole_clusters_in_order():
    df = _zero_sum_clusters()

    def check(d: pd.DataFrame) -> float:
        assert "_draw" in d.columns
        for _, g in d.groupby("_draw", sort=False):
            # one whole cluster per draw, its rows in their original frame order
            if (g["cid"].nunique() != 1 or len(g) != g["size"].iloc[0]
                    or sorted(g["j"]) != list(range(len(g))) or list(g["row_id"]) != sorted(g["row_id"])):
                return 0.0
        return float(d["_draw"].nunique())

    out = stats.cluster_bootstrap(df, "cid", check, n_boot=100, seed=SEED, draw_col="_draw")
    assert np.all(out == 10.0)  # 10 draws per replicate, each one complete cluster
    with pytest.raises(ValueError):
        stats.cluster_bootstrap(df, "cid", check, n_boot=5, seed=SEED, draw_col="v")  # name clash


def _stratified_frame() -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(SEED)
    for cid in range(20):
        stratum = "s1" if cid < 5 else "s2"
        size = int(rng.integers(1, 4))
        for j in range(size):
            rows.append({"pid": 100 + cid, "stratum": stratum, "first": int(j == 0), "y": float(rng.random() < 0.4)})
    return pd.DataFrame(rows)


def test_cluster_bootstrap_strata_keep_cluster_counts():
    df = _stratified_frame()

    def n_s1(d: pd.DataFrame) -> float:
        return float(d.loc[d["stratum"] == "s1", "first"].sum())  # number of clusters drawn in s1

    strat = stats.cluster_bootstrap(df, "pid", n_s1, n_boot=400, seed=SEED, strata_col="stratum")
    assert np.all(strat == 5.0)
    total = stats.cluster_bootstrap(df, "pid", lambda d: d["first"].sum(), n_boot=50, seed=SEED,
                                    strata_col="stratum")
    assert np.all(total == 20.0)
    unstrat = stats.cluster_bootstrap(df, "pid", n_s1, n_boot=400, seed=SEED)
    assert unstrat.std() > 0  # without strata the count per stratum varies


def test_cluster_bootstrap_rejects_clusters_spanning_strata():
    df = _stratified_frame()
    df.loc[df.index[0], "stratum"] = "s2"  # pid 100 now has rows in two strata (if it has >1 row)
    df = pd.concat([df, pd.DataFrame([{"pid": 100, "stratum": "s1", "first": 0, "y": 0.0}])], ignore_index=True)
    with pytest.raises(ValueError, match="nested"):
        stats.cluster_bootstrap(df, "pid", lambda d: 0.0, n_boot=5, seed=SEED, strata_col="stratum")


def test_cluster_bootstrap_deterministic_and_validated():
    df = _stratified_frame()
    fn = lambda d: d["y"].mean()  # noqa: E731
    a = stats.cluster_bootstrap(df, "pid", fn, n_boot=300, seed=7)
    b = stats.cluster_bootstrap(df, "pid", fn, n_boot=300, seed=7)
    c = stats.cluster_bootstrap(df, "pid", fn, n_boot=300, seed=8)
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    with pytest.raises(ValueError):
        stats.cluster_bootstrap(df, "pid", fn, n_boot=0, seed=1)
    with pytest.raises(ValueError):
        stats.cluster_bootstrap(df, "pid", fn, n_boot=10, seed=None)
    with pytest.raises(ValueError):
        stats.cluster_bootstrap(df.iloc[:0], "pid", fn, n_boot=10, seed=1)
    with pytest.raises(KeyError):
        stats.cluster_bootstrap(df, "nope", fn, n_boot=10, seed=1)
    with pytest.raises(ValueError):
        stats.cluster_bootstrap(df, "pid", lambda d: d["y"].to_numpy(), n_boot=3, seed=1)  # not a scalar
    bad = df.astype({"pid": float})
    bad.loc[bad.index[0], "pid"] = np.nan
    with pytest.raises(ValueError):
        stats.cluster_bootstrap(bad, "pid", fn, n_boot=3, seed=1)


def test_cluster_jackknife():
    df = pd.DataFrame({"cid": [3, 3, 1, 2, 2, 2], "x": [1.0, 2.0, 10.0, 4.0, 5.0, 6.0]})
    jack = stats.cluster_jackknife(df, "cid", _mean_x)
    # clusters in sorted order: leave out 1, then 2, then 3
    expected = [np.mean([1, 2, 4, 5, 6]), np.mean([1, 2, 10]), np.mean([10, 4, 5, 6])]
    np.testing.assert_allclose(jack, expected)
    with pytest.raises(ValueError):
        stats.cluster_jackknife(df[df["cid"] == 3], "cid", _mean_x)


# ----------------------------------------------------------------------------- BCa
def test_bca_unclustered_mean_agrees_with_scipy():
    rng = np.random.default_rng(SEED)
    x = rng.exponential(scale=0.5, size=120)  # skewed: exercises z0 and the acceleration
    df = _singletons(x)
    boot = stats.cluster_bootstrap(df, "cid", _mean_x, n_boot=9999, seed=SEED)
    jack = stats.cluster_jackknife(df, "cid", _mean_x)
    lo, hi, method = stats.bca_interval(_mean_x(df), boot, jack)
    ref = scipy.stats.bootstrap((x,), np.mean, n_resamples=9999, method="BCa",
                                rng=np.random.default_rng(SEED + 1))
    assert method == "bca"
    assert abs(lo - ref.confidence_interval.low) < 0.01
    assert abs(hi - ref.confidence_interval.high) < 0.01
    # given scipy's own bootstrap distribution, the interval matches scipy to rounding
    lo2, hi2, method2 = stats.bca_interval(float(np.mean(x)), ref.bootstrap_distribution, jack)
    assert method2 == "bca"
    assert lo2 == pytest.approx(ref.confidence_interval.low, rel=1e-9)
    assert hi2 == pytest.approx(ref.confidence_interval.high, rel=1e-9)


def test_bca_clustered_mean_matches_scipy_on_cluster_means():
    # equal cluster sizes: the row mean of a cluster resample is the mean of the drawn cluster means
    rng = np.random.default_rng(SEED)
    cluster_effect = rng.exponential(1.0, size=60)
    df = pd.DataFrame({"cid": np.repeat(np.arange(60), 2),
                       "x": np.repeat(cluster_effect, 2) + rng.normal(0, 0.3, size=120)})
    cmeans = df.groupby("cid")["x"].mean().to_numpy()
    ref = scipy.stats.bootstrap((cmeans,), np.mean, n_resamples=9999, method="BCa",
                                rng=np.random.default_rng(SEED))
    jack = stats.cluster_jackknife(df, "cid", _mean_x)
    lo, hi, method = stats.bca_interval(_mean_x(df), ref.bootstrap_distribution, jack)
    assert method == "bca"
    assert lo == pytest.approx(ref.confidence_interval.low, rel=1e-9)
    assert hi == pytest.approx(ref.confidence_interval.high, rel=1e-9)
    res = stats.clustered_ci(df, "cid", _mean_x, n_boot=9999, seed=SEED)
    assert abs(res["lo"] - ref.confidence_interval.low) < 0.02
    assert abs(res["hi"] - ref.confidence_interval.high) < 0.02


def test_bca_reduces_to_percentile_without_bias_or_skew():
    boot = np.linspace(-1.0, 1.0, 1001)
    lo, hi, method = stats.bca_interval(0.0, boot, np.array([-1.0, 0.0, 1.0]))
    assert method == "bca"  # z0 = 0 and a = 0
    plo, phi = np.quantile(boot, [0.025, 0.975])
    assert lo == pytest.approx(plo) and hi == pytest.approx(phi)


def test_bca_percentile_fallbacks():
    boot = np.linspace(0.0, 1.0, 101)
    jack = np.array([0.4, 0.5, 0.7])
    plo, phi = np.quantile(boot, [0.025, 0.975])
    # constant statistic: all replicates equal
    assert stats.bca_interval(2.0, np.full(100, 2.0), np.full(10, 2.0)) == (2.0, 2.0, "percentile")
    # z0 infinite (theta below or above every replicate)
    for theta in (-1.0, 2.0):
        lo, hi, method = stats.bca_interval(theta, boot, jack)
        assert method == "percentile" and lo == pytest.approx(plo) and hi == pytest.approx(phi)
    # acceleration denominator zero (all jackknife values equal) or undefined (NaN, too few)
    for bad_jack in (np.full(5, 0.3), np.array([0.2, np.nan, 0.5]), np.array([0.3])):
        assert stats.bca_interval(0.5, boot, bad_jack)[2] == "percentile"
    # 1 - a (z0 + z) <= 0: strong skew in the jackknife and theta near the top of the replicates
    boot2 = np.linspace(0.0, 1.0, 1000)
    jack2 = np.r_[-10.0, np.zeros(999)]  # acceleration about +0.166
    theta2 = 0.9995                        # 999 of 1000 replicates below: z0 about 3.09
    lo, hi, method = stats.bca_interval(theta2, boot2, jack2, alpha=0.001)
    assert method == "percentile"
    assert (lo, hi) == pytest.approx(tuple(np.quantile(boot2, [0.0005, 0.9995])))
    assert stats.bca_interval(theta2, boot2, jack2, alpha=0.05)[2] == "bca"


def test_bca_drops_nan_replicates_and_handles_nan_theta():
    rng = np.random.default_rng(SEED)
    boot = rng.normal(0.3, 0.1, size=2000)
    jack = rng.normal(0.3, 0.01, size=50)
    with_nan = np.r_[boot[:1000], [np.nan] * 37, boot[1000:]]
    assert stats.bca_interval(0.3, with_nan, jack) == stats.bca_interval(0.3, boot, jack)
    lo, hi, method = stats.bca_interval(float("nan"), boot, jack)
    assert math.isnan(lo) and math.isnan(hi) and method == "none"
    assert stats.bca_interval(0.3, np.full(10, np.nan), jack)[2] == "none"


# ----------------------------------------------------------------------------- CI inversion
def _boot_inputs(x: np.ndarray, n_boot: int = 1999):
    df = _singletons(x)
    boot = stats.cluster_bootstrap(df, "cid", _mean_x, n_boot=n_boot, seed=SEED)
    jack = stats.cluster_jackknife(df, "cid", _mean_x)
    return _mean_x(df), boot, jack


def test_ci_inversion_extremes():
    rng = np.random.default_rng(SEED)
    theta, boot, jack = _boot_inputs(rng.normal(5.0, 1.0, size=40))
    assert stats.ci_inversion_p(theta, boot, jack) == (0.001, "<=0.001")
    v = rng.normal(0.0, 1.0, size=20)
    theta, boot, jack = _boot_inputs(np.r_[v, -v])  # centred on the null
    assert stats.ci_inversion_p(theta, boot, jack) == (1.0, ">0.5")
    assert stats.ci_inversion_p(theta, boot, jack, alpha_grid=[0.01, 0.05, 0.1]) == (1.0, ">0.1")
    p, flag = stats.ci_inversion_p(float("nan"), boot, jack)
    assert math.isnan(p) and flag == "na"
    with pytest.raises(ValueError):
        stats.ci_inversion_p(theta, boot, jack, alpha_grid=[0.0, 0.05])


def test_ci_inversion_p_decreases_as_effect_grows():
    rng = np.random.default_rng(SEED)
    v = rng.normal(0.0, 1.0, size=30)
    v = v - v.mean()
    theta0, boot0, jack0 = _boot_inputs(v)
    # for a mean, shifting the data by s shifts the estimate, replicates and jackknife values by s
    theta_s, boot_s, jack_s = _boot_inputs(v + 0.3)
    np.testing.assert_allclose(boot_s, boot0 + 0.3, atol=1e-12)
    shifts = [0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 1.2]
    inputs = [(theta0 + s, boot0 + s, jack0 + s) for s in shifts]
    results = [stats.ci_inversion_p(*inp) for inp in inputs]
    ps = [p for p, _ in results]
    assert results[0] == (1.0, ">0.5")
    assert results[-1] == (0.001, "<=0.001")
    assert all(a >= b for a, b in zip(ps, ps[1:]))
    assert ps[2] > ps[3] > ps[4]  # strictly decreasing in the informative range
    for (theta, boot, jack), (p, flag) in zip(inputs, results):
        lo, hi, _ = stats.bca_interval(theta, boot, jack, alpha=0.05)
        assert (lo > 0 or hi < 0) == (p <= 0.05)  # consistent with the 95% interval
        if flag == "":
            grid = stats.ALPHA_GRID
            i = int(np.flatnonzero(grid == p)[0])
            lo, hi, _ = stats.bca_interval(theta, boot, jack, alpha=grid[i])
            assert lo > 0 or hi < 0
            lo, hi, _ = stats.bca_interval(theta, boot, jack, alpha=grid[i - 1])
            assert lo <= 0 <= hi


# ----------------------------------------------------------------------------- clustered_ci
DOCUMENTED_KEYS = {"estimate", "lo", "hi", "method", "p_value", "p_flag", "n_clusters", "n_rows", "n_boot"}


def test_clustered_ci_keys_and_consistency():
    rng = np.random.default_rng(SEED)
    df = pd.DataFrame({"pid": np.repeat(np.arange(50), 2), "x": rng.normal(0.4, 1.0, size=100)})
    res = stats.clustered_ci(df, "pid", _mean_x, n_boot=999, seed=SEED)
    assert DOCUMENTED_KEYS <= set(res)
    assert res["n_clusters"] == 50 and res["n_rows"] == 100 and res["n_boot"] == 999
    assert res["n_boot_valid"] == 999
    assert res["estimate"] == pytest.approx(df["x"].mean())
    boot = stats.cluster_bootstrap(df, "pid", _mean_x, n_boot=999, seed=SEED)
    jack = stats.cluster_jackknife(df, "pid", _mean_x)
    assert (res["lo"], res["hi"], res["method"]) == stats.bca_interval(res["estimate"], boot, jack)
    assert (res["p_value"], res["p_flag"]) == stats.ci_inversion_p(res["estimate"], boot, jack)
    assert res["lo"] < res["estimate"] < res["hi"]
    shifted = stats.clustered_ci(df, "pid", _mean_x, n_boot=999, seed=SEED, null=res["estimate"])
    assert shifted["p_value"] == 1.0  # the estimate itself is never excluded


def test_clustered_ci_constant_statistic_uses_percentile():
    df = pd.DataFrame({"pid": [1, 1, 2, 3], "x": [0.0, 1.0, 2.0, 3.0]})
    res = stats.clustered_ci(df, "pid", lambda d: 1.0, n_boot=200, seed=SEED)
    assert res["method"] == "percentile"
    assert res["lo"] == res["hi"] == res["estimate"] == 1.0


def test_clustered_ci_degenerate_inputs():
    def boom(d):
        raise AssertionError("stat_fn must not be called on an empty frame")

    empty = stats.clustered_ci(pd.DataFrame({"pid": [], "x": []}), "pid", boom, n_boot=10, seed=1)
    assert DOCUMENTED_KEYS <= set(empty)
    assert empty["n_rows"] == 0 and empty["n_clusters"] == 0 and empty["method"] == "none"
    assert math.isnan(empty["estimate"]) and math.isnan(empty["p_value"]) and empty["p_flag"] == "na"
    one = stats.clustered_ci(pd.DataFrame({"pid": [4, 4], "x": [1.0, 3.0]}), "pid", _mean_x, n_boot=10, seed=1)
    assert one["estimate"] == 2.0 and one["n_clusters"] == 1
    assert math.isnan(one["lo"]) and math.isnan(one["hi"]) and one["method"] == "none"


def test_clustered_ci_with_strata_h1_style():
    rng = np.random.default_rng(SEED)
    n_line, n_noline = 60, 40
    df = pd.DataFrame({
        "nih_patient_id": np.arange(n_line + n_noline),
        "stratum": ["line"] * n_line + ["noline"] * n_noline,
        "yes": np.r_[rng.random(n_line) < 0.55, rng.random(n_noline) < 0.25].astype(float),
    })

    def diff(d):
        y, line = d["yes"].to_numpy(), d["stratum"].to_numpy() == "line"
        return y[line].mean() - y[~line].mean()

    res = stats.clustered_ci(df, "nih_patient_id", diff, n_boot=1999, seed=SEED, strata_col="stratum")
    assert res["estimate"] == pytest.approx(diff(df))
    assert res["lo"] < res["estimate"] < res["hi"]
    assert res["n_boot_valid"] == 1999  # strata keep both groups non-empty in every replicate
    assert 0 < res["p_value"] <= 1


def test_clustered_ci_draw_col_supports_pair_pivots():
    # one A and one B row per patient; the pivot needs the draw id when a patient is drawn twice
    rng = np.random.default_rng(SEED)
    n = 40
    df = pd.DataFrame({
        "pid": np.repeat(np.arange(n), 2),
        "arm": ["A", "B"] * n,
        "yes": (rng.random(2 * n) < np.tile([0.8, 0.3], n)).astype(float),
    })

    def pivot_diff(d):
        wide = d.pivot(index="_draw", columns="arm", values="yes")
        return float((wide["A"] - wide["B"]).mean())

    def row_diff(d):
        return d.loc[d["arm"] == "A", "yes"].mean() - d.loc[d["arm"] == "B", "yes"].mean()

    b_pivot = stats.cluster_bootstrap(df, "pid", pivot_diff, n_boot=150, seed=SEED, draw_col="_draw")
    b_rows = stats.cluster_bootstrap(df, "pid", row_diff, n_boot=150, seed=SEED)
    np.testing.assert_allclose(b_pivot, b_rows, atol=1e-12)
    res = stats.clustered_ci(df, "pid", pivot_diff, n_boot=150, seed=SEED, draw_col="_draw")
    assert res["estimate"] == pytest.approx(row_diff(df))
    assert res["method"] in ("bca", "percentile")


def test_clustered_ci_counts_nan_replicates():
    df = _singletons(np.arange(30, dtype=float))

    def needs_first(d):
        return d["x"].mean() if (d["cid"] == 0).any() else float("nan")

    res = stats.clustered_ci(df, "cid", needs_first, n_boot=400, seed=SEED)
    assert 0 < res["n_boot_valid"] < 400
    assert np.isfinite(res["lo"]) and np.isfinite(res["hi"])


# ----------------------------------------------------------------------------- performance
def _perf_frame() -> pd.DataFrame:
    """400 rows in 240 clusters: 160 patients with an A/B pair (320 rows) and 80 U patients."""
    rng = np.random.default_rng(SEED)
    pid = np.r_[np.repeat(np.arange(160), 2), np.arange(160, 240)]
    arm = np.r_[np.tile(["A", "B"], 160), np.repeat("U", 80)]
    yes = (rng.random(400) < np.where(arm == "A", 0.8, 0.3)).astype(float)
    return pd.DataFrame({"nih_patient_id": pid, "arm": arm, "yes": yes, "z": rng.normal(size=400)})


@pytest.mark.slow
def test_cluster_bootstrap_performance_9999_resamples():
    df = _perf_frame()
    assert len(df) == 400 and df["nih_patient_id"].nunique() == 240

    def stat(d):
        # a typical metric (yes-rate on A minus yes-rate elsewhere), written on numpy columns
        y, is_a = d["yes"].to_numpy(), d["arm"].to_numpy() == "A"
        return y[is_a].mean() - y[~is_a].mean()

    t0 = time.perf_counter()
    boot = stats.cluster_bootstrap(df, "nih_patient_id", stat, n_boot=9999, seed=SEED)
    elapsed = time.perf_counter() - t0
    assert boot.shape == (9999,) and np.all(np.isfinite(boot))
    assert elapsed < 30.0, f"9,999 resamples took {elapsed:.1f} s"
    # the per-cluster layout is precomputed: replicate values match a direct reimplementation
    rng = np.random.default_rng(SEED)
    pids = np.sort(df["nih_patient_id"].unique())
    rows_of = {p: np.flatnonzero(df["nih_patient_id"].to_numpy() == p) for p in pids}
    for b in range(3):
        sel = pids[rng.integers(0, pids.size, size=pids.size)]
        rows = np.concatenate([rows_of[p] for p in sel])
        assert boot[b] == pytest.approx(stat(df.iloc[rows]), abs=1e-12)
