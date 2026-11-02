"""Statistics for Stage 1 (PROTOCOL.md sections 5 and 6).

- Wilson score intervals for rates (hit_A, FAR_B, FAR_U, yes-rate, abstention, coverage).
- Patient-clustered bootstrap: whole clusters (NIH patients, so A and B of a pair move
  together) resampled with replacement, optionally within strata; cluster jackknife; BCa
  interval with the documented percentile fallback; the bootstrap p-value by CI inversion on
  a fixed alpha grid (the same resamples reused), which feeds Holm within a family.
- Exact McNemar test, Holm step-down adjustment, Cohen's kappa, AUROC (Mann-Whitney, ties
  count one half), design effect from the one-way ANOVA ICC, and the normal-approximation
  power of a two-sided two-proportion test (the PROTOCOL power note).

Every random choice takes an explicit integer seed (callers pass
``constants.derive_seed("bootstrap")``); the same inputs and seed give identical output.
"""
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy import special
from scipy import stats as sps

#: Alpha grid for the CI-inversion p-value: 0.001..0.009 step 0.001, then 0.01..0.50 step 0.005.
ALPHA_GRID: np.ndarray = np.unique(
    np.round(np.concatenate([np.arange(1, 10), np.arange(10, 505, 5)]) / 1000.0, 4)
)

_NAN = float("nan")


# ----------------------------------------------------------------------------- helpers
def _check_alpha(alpha: float) -> float:
    a = float(alpha)
    if not 0.0 < a < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
    return a


def _as_count(value: Any, name: str) -> int:
    """A non-negative integer (accepts numpy integers and integral floats)."""
    if isinstance(value, bool) or value is None:
        raise TypeError(f"{name} must be a non-negative integer, got {value!r}")
    try:
        f = float(value)
    except (TypeError, ValueError):
        raise TypeError(f"{name} must be a non-negative integer, got {value!r}") from None
    if not math.isfinite(f) or f != int(f) or f < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    return int(f)


def _as_seed(seed: Any) -> int:
    if seed is None or isinstance(seed, bool):
        raise ValueError("an explicit integer seed is required (use constants.derive_seed)")
    return _as_count(seed, "seed")


def _scalar(value: Any) -> float:
    """Convert a stat_fn result to float (None and pd.NA become NaN)."""
    if value is None or value is pd.NA:
        return _NAN
    arr = np.asarray(value, dtype=np.float64)
    if arr.size != 1:
        raise ValueError(f"stat_fn must return a scalar, got an array of shape {arr.shape}")
    return float(arr.reshape(()))


def _codes(values: pd.Series, what: str) -> tuple[np.ndarray, pd.Index]:
    """Integer codes 0..G-1 over the sorted unique values (first-appearance order if unsortable)."""
    if values.isna().any():
        raise ValueError(f"{what} has missing values")
    try:
        codes, uniques = pd.factorize(values, sort=True)
    except TypeError:  # unorderable mix of types: first-appearance order is still deterministic
        codes, uniques = pd.factorize(values, sort=False)
    return np.asarray(codes, dtype=np.intp), pd.Index(uniques)


def _cluster_layout(
    df: pd.DataFrame, cluster_col: str, strata_col: str | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    """Precomputed row-index layout for cluster resampling.

    Returns ``codes`` (cluster code per row), ``order`` (row positions sorted by cluster, stable),
    ``starts`` and ``sizes`` (each cluster's block in ``order``) and ``groups`` (cluster codes per
    stratum, strata in sorted order; one group holding every cluster when unstratified).
    """
    if cluster_col not in df.columns:
        raise KeyError(f"cluster column {cluster_col!r} not in frame")
    codes, uniques = _codes(df[cluster_col], f"cluster column {cluster_col!r}")
    n_clusters = len(uniques)
    sizes = np.bincount(codes, minlength=n_clusters).astype(np.intp)
    order = np.argsort(codes, kind="stable").astype(np.intp)
    starts = np.zeros(n_clusters, dtype=np.intp)
    if n_clusters > 1:
        starts[1:] = np.cumsum(sizes)[:-1]
    if strata_col is None:
        groups = [np.arange(n_clusters, dtype=np.intp)]
    else:
        if strata_col not in df.columns:
            raise KeyError(f"strata column {strata_col!r} not in frame")
        scodes, suniques = _codes(df[strata_col], f"strata column {strata_col!r}")
        n_strata = len(suniques)
        lo = np.full(n_clusters, n_strata, dtype=np.intp)
        hi = np.full(n_clusters, -1, dtype=np.intp)
        np.minimum.at(lo, codes, scodes)
        np.maximum.at(hi, codes, scodes)
        bad = np.flatnonzero(lo != hi)
        if bad.size:
            raise ValueError(
                f"clusters must be nested within strata; {bad.size} cluster(s) span several strata, "
                f"e.g. {list(uniques[bad[:5]])}"
            )
        groups = [np.flatnonzero(lo == s).astype(np.intp) for s in range(n_strata)]
    return codes, order, starts, sizes, groups


def _rows_for(
    sel: np.ndarray, order: np.ndarray, starts: np.ndarray, sizes: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Row positions of the drawn clusters ``sel``, each cluster's rows kept as one block."""
    sz = sizes[sel]
    ends = np.cumsum(sz)
    pos = np.arange(ends[-1], dtype=np.intp) + np.repeat(starts[sel] - (ends - sz), sz)
    return order[pos], sz


# ----------------------------------------------------------------------------- intervals for rates
def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for k successes in n trials; (nan, nan) when n == 0.

    The bounds are exactly 0 when k == 0 and exactly 1 when k == n.
    """
    k = _as_count(k, "k")
    n = _as_count(n, "n")
    a = _check_alpha(alpha)
    if k > n:
        raise ValueError(f"k={k} exceeds n={n}")
    if n == 0:
        return (_NAN, _NAN)
    z = float(special.ndtri(1.0 - a / 2.0))
    z2 = z * z
    p = k / n
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / denom
    lo = 0.0 if k == 0 else max(0.0, centre - half)
    hi = 1.0 if k == n else min(1.0, centre + half)
    return (lo, hi)


def wilson_halfwidth(p: float, n: float, alpha: float = 0.05) -> float:
    """Half-width of the Wilson interval at observed proportion p and sample size n (NaN if n == 0).

    For example about 0.088 at p = 0.5 and n = 120 (the PROTOCOL power note's "about 0.09").
    """
    p = float(p)
    n = float(n)
    a = _check_alpha(alpha)
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1], got {p!r}")
    if not n >= 0.0:
        raise ValueError(f"n must be non-negative, got {n!r}")
    if n == 0.0:
        return _NAN
    z = float(special.ndtri(1.0 - a / 2.0))
    z2 = z * z
    return z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / (1.0 + z2 / n)


# ----------------------------------------------------------------------------- clustered bootstrap
def cluster_bootstrap(
    df: pd.DataFrame,
    cluster_col: str,
    stat_fn: Callable[[pd.DataFrame], float],
    n_boot: int = 9999,
    seed: int = 0,
    strata_col: str | None = None,
    *,
    draw_col: str | None = None,
) -> np.ndarray:
    """Bootstrap replicates of ``stat_fn`` with whole clusters resampled with replacement.

    Each replicate draws as many clusters as the frame has (within each stratum when
    ``strata_col`` is given, keeping the number of clusters per stratum; clusters must be
    nested within strata) and calls ``stat_fn`` on ``df.iloc[rows]``, where the rows of every
    drawn cluster appear together and in their original order. Row-index arrays per cluster
    are precomputed, so the library's own cost is one ``iloc`` per replicate (a few seconds for
    9,999 replicates of a 400-row frame); ``stat_fn`` usually dominates, so write it on numpy
    columns (``d["yes"].to_numpy()[mask].mean()``), which is several times faster than chains of
    pandas boolean indexing. The resampled frame keeps the original index labels (with repeats).

    ``draw_col`` (optional extension): when given, the resampled frame carries an extra integer
    column of that name numbering the draws 0..G-1, so a cluster drawn twice can be told apart
    (needed when ``stat_fn`` pivots or groups by a within-cluster key such as a pair id).

    Returns a float64 array of length ``n_boot`` (NaN where ``stat_fn`` returned NaN).
    """
    n_boot = _as_count(n_boot, "n_boot")
    if n_boot < 1:
        raise ValueError("n_boot must be at least 1")
    if len(df) == 0:
        raise ValueError("cannot bootstrap an empty frame")
    if draw_col is not None and draw_col in df.columns:
        raise ValueError(f"draw_col {draw_col!r} already exists in the frame")
    _, order, starts, sizes, groups = _cluster_layout(df, cluster_col, strata_col)
    rng = np.random.default_rng(_as_seed(seed))
    out = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        if len(groups) == 1:
            g = groups[0]
            sel = g[rng.integers(0, g.size, size=g.size)]
        else:
            sel = np.concatenate([g[rng.integers(0, g.size, size=g.size)] for g in groups])
        rows, sz = _rows_for(sel, order, starts, sizes)
        sub = df.iloc[rows]
        if draw_col is not None:
            sub[draw_col] = np.repeat(np.arange(sel.size), sz)
        out[b] = _scalar(stat_fn(sub))
    return out


def cluster_jackknife(
    df: pd.DataFrame,
    cluster_col: str,
    stat_fn: Callable[[pd.DataFrame], float],
    *,
    draw_col: str | None = None,
) -> np.ndarray:
    """Leave-one-cluster-out values of ``stat_fn`` (one per cluster, clusters in sorted order).

    Used for the BCa acceleration. ``draw_col`` (optional) adds an integer column that is unique
    per cluster, matching what ``cluster_bootstrap`` provides. Needs at least 2 clusters.
    """
    if draw_col is not None and draw_col in df.columns:
        raise ValueError(f"draw_col {draw_col!r} already exists in the frame")
    codes, _, _, sizes, _ = _cluster_layout(df, cluster_col, None)
    n_clusters = sizes.size
    if n_clusters < 2:
        raise ValueError("the cluster jackknife needs at least 2 clusters")
    out = np.empty(n_clusters, dtype=np.float64)
    for g in range(n_clusters):
        rows = np.flatnonzero(codes != g)
        sub = df.iloc[rows]
        if draw_col is not None:
            sub[draw_col] = codes[rows]
        out[g] = _scalar(stat_fn(sub))
    return out


def bca_interval(
    theta_hat: float, boot: np.ndarray, jack: np.ndarray, alpha: float = 0.05
) -> tuple[float, float, str]:
    """BCa interval (Efron and Tibshirani 1993, ch. 14) with a percentile fallback.

    z0 = Phi^-1((#{boot < theta_hat} + 0.5 #{boot == theta_hat}) / B); acceleration
    a = sum(d^3) / (6 (sum(d^2))^1.5) with d = mean(jack) - jack; the interval is the
    quantiles of ``boot`` (linear interpolation) at Phi(z0 + (z0 + z_q) / (1 - a (z0 + z_q)))
    for q = alpha/2 and 1 - alpha/2. Falls back to the percentile interval (quantiles
    alpha/2 and 1 - alpha/2), method "percentile", when all replicates are equal, when z0 is
    infinite, or when an acceleration denominator is non-positive or undefined (all jackknife
    values equal, NaN in ``jack``, or 1 - a (z0 + z_q) <= 0); method "bca" otherwise.

    NaN replicates are dropped. Returns (nan, nan, "none") when ``theta_hat`` is not finite or
    no replicate is left.
    """
    a_ = _check_alpha(alpha)
    theta = float(theta_hat)
    reps = np.asarray(boot, dtype=np.float64).ravel()
    reps = reps[~np.isnan(reps)]
    if not math.isfinite(theta) or reps.size == 0:
        return (_NAN, _NAN, "none")
    q = np.array([a_ / 2.0, 1.0 - a_ / 2.0])

    def percentile() -> tuple[float, float, str]:
        lo, hi = np.quantile(reps, q)
        return (float(lo), float(hi), "percentile")

    if np.all(reps == reps[0]):
        return percentile()
    p0 = (np.count_nonzero(reps < theta) + 0.5 * np.count_nonzero(reps == theta)) / reps.size
    z0 = float(special.ndtri(p0))
    if not math.isfinite(z0):
        return percentile()
    jk = np.asarray(jack, dtype=np.float64).ravel()
    if jk.size < 2:
        return percentile()
    d = jk.mean() - jk
    den = 6.0 * float(np.sum(d * d)) ** 1.5
    if not den > 0.0:
        return percentile()
    acc = float(np.sum(d**3)) / den
    w = z0 + special.ndtri(q)
    adj = 1.0 - acc * w
    if not np.all(adj > 0.0):
        return percentile()
    levels = special.ndtr(z0 + w / adj)
    lo, hi = np.quantile(reps, levels)
    return (float(lo), float(hi), "bca")


def ci_inversion_p(
    theta_hat: float,
    boot: np.ndarray,
    jack: np.ndarray,
    null: float = 0.0,
    alpha_grid: Sequence[float] | np.ndarray = ALPHA_GRID,
) -> tuple[float, str]:
    """Bootstrap p-value by CI inversion: the smallest alpha in the grid whose (1 - alpha)
    interval from ``bca_interval`` excludes ``null`` (strictly outside the interval).

    Returns (grid[0], "<=0.001") when the null is excluded already at the first grid point,
    (1.0, ">0.5") when it is never excluded, (alpha, "") otherwise (flags shown for the
    default grid), and (nan, "na") when no interval can be formed.
    """
    grid = np.unique(np.asarray(alpha_grid, dtype=np.float64).ravel())
    if grid.size == 0 or not (grid[0] > 0.0 and grid[-1] < 1.0):
        raise ValueError("alpha_grid must be non-empty with values in (0, 1)")
    nul = float(null)
    for i, a in enumerate(grid):
        lo, hi, method = bca_interval(theta_hat, boot, jack, alpha=float(a))
        if method == "none":
            return (_NAN, "na")
        if nul < lo or nul > hi:
            return (float(a), f"<={a:g}") if i == 0 else (float(a), "")
    return (1.0, f">{grid[-1]:g}")


def clustered_ci(
    df: pd.DataFrame,
    cluster_col: str,
    stat_fn: Callable[[pd.DataFrame], float],
    n_boot: int = 9999,
    seed: int = 0,
    alpha: float = 0.05,
    null: float = 0.0,
    strata_col: str | None = None,
    *,
    draw_col: str | None = None,
) -> dict:
    """Point estimate, clustered BCa interval and CI-inversion p-value in one call.

    Returns a dict with ``estimate``, ``lo``, ``hi``, ``method`` ("bca", "percentile" or
    "none"), ``p_value``, ``p_flag``, ``n_clusters``, ``n_rows``, ``n_boot`` (requested) and
    ``n_boot_valid`` (replicates that were not NaN). The interval and p-value are NaN (method
    "none", flag "na") for an empty frame, fewer than 2 clusters or a non-finite estimate;
    ``stat_fn`` is not called on an empty frame. ``draw_col`` is passed to the bootstrap and
    the jackknife and, on the full frame, numbers the clusters.
    """
    n_boot = _as_count(n_boot, "n_boot")
    result: dict[str, Any] = {
        "estimate": _NAN, "lo": _NAN, "hi": _NAN, "method": "none", "p_value": _NAN, "p_flag": "na",
        "n_clusters": 0, "n_rows": int(len(df)), "n_boot": n_boot, "n_boot_valid": 0,
    }
    if len(df) == 0:
        return result
    codes, _, _, sizes, _ = _cluster_layout(df, cluster_col, strata_col)
    result["n_clusters"] = int(sizes.size)
    if draw_col is not None and draw_col in df.columns:
        raise ValueError(f"draw_col {draw_col!r} already exists in the frame")
    full = df if draw_col is None else df.assign(**{draw_col: codes})
    estimate = _scalar(stat_fn(full))
    result["estimate"] = estimate
    if sizes.size < 2 or not math.isfinite(estimate):
        return result
    boot = cluster_bootstrap(df, cluster_col, stat_fn, n_boot=n_boot, seed=seed,
                             strata_col=strata_col, draw_col=draw_col)
    jack = cluster_jackknife(df, cluster_col, stat_fn, draw_col=draw_col)
    lo, hi, method = bca_interval(estimate, boot, jack, alpha=alpha)
    p_value, p_flag = ci_inversion_p(estimate, boot, jack, null=null)
    result.update(lo=lo, hi=hi, method=method, p_value=p_value, p_flag=p_flag,
                  n_boot_valid=int(np.count_nonzero(~np.isnan(boot))))
    return result


# ----------------------------------------------------------------------------- tests and summaries
def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar p = min(1, 2 * BinomCDF(min(b, c); b + c, 0.5)); 1.0 when b + c == 0.

    b and c are the two discordant counts (for H3: abstain under P2hi only, abstain under P1 only).
    """
    b = _as_count(b, "b")
    c = _as_count(c, "c")
    n = b + c
    if n == 0:
        return 1.0
    return float(min(1.0, 2.0 * sps.binom.cdf(min(b, c), n, 0.5)))


def holm(pvals: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values, in input order.

    adjusted p_(i) = max over j <= i of min(1, (m - j + 1) p_(j)) for the sorted p-values, so the
    adjusted values are monotone and capped at 1. NaN (or None / pd.NA) stays NaN and is
    excluded from m.
    """
    p = np.array([_NAN if _is_missing(v) else float(v) for v in pvals], dtype=np.float64)
    valid = ~np.isnan(p)
    if np.any((p[valid] < 0.0) | (p[valid] > 1.0)):
        raise ValueError("p-values must lie in [0, 1]")
    out = np.full(p.shape, _NAN)
    idx = np.flatnonzero(valid)
    m = idx.size
    if m:
        order = idx[np.argsort(p[idx], kind="stable")]
        adj = np.maximum.accumulate(np.minimum(1.0, (m - np.arange(m)) * p[order]))
        out[order] = adj
    return [float(v) for v in out]


def _is_missing(x: Any) -> bool:
    if x is None or x is pd.NA:
        return True
    return isinstance(x, (float, np.floating)) and bool(np.isnan(x))


def cohen_kappa(a: Sequence, b: Sequence) -> float:
    """Cohen's kappa between two ratings of the same items (unweighted, any hashable labels).

    kappa = (p_o - p_e) / (1 - p_e), computed in exact integer arithmetic; NaN when the expected
    agreement p_e is 1 (both ratings use one and the same single category) or there are no items.
    Raises ValueError for unequal lengths or missing ratings (drop "skip" responses first).
    """
    ra, rb = list(a), list(b)
    if len(ra) != len(rb):
        raise ValueError(f"ratings differ in length: {len(ra)} vs {len(rb)}")
    n = len(ra)
    if n == 0:
        return _NAN
    if any(_is_missing(x) for x in ra) or any(_is_missing(x) for x in rb):
        raise ValueError("ratings contain missing values")
    ca, cb = Counter(ra), Counter(rb)
    agree = sum(1 for x, y in zip(ra, rb) if x == y)
    pe_num = sum(count * cb.get(label, 0) for label, count in ca.items())
    if pe_num == n * n:
        return _NAN
    return (n * agree - pe_num) / (n * n - pe_num)


def auroc(pos_scores: Sequence[float], neg_scores: Sequence[float]) -> float:
    """AUROC = P(pos > neg) + 0.5 P(pos == neg), from the Mann-Whitney U with average ranks.

    NaN when either group is empty; raises ValueError for NaN scores (filter them first).
    """
    pos = np.asarray(pos_scores, dtype=np.float64).ravel()
    neg = np.asarray(neg_scores, dtype=np.float64).ravel()
    if pos.size == 0 or neg.size == 0:
        return _NAN
    if np.isnan(pos).any() or np.isnan(neg).any():
        raise ValueError("scores contain NaN")
    ranks = sps.rankdata(np.concatenate([pos, neg]))
    u = float(ranks[: pos.size].sum()) - pos.size * (pos.size + 1) / 2.0
    return u / (pos.size * neg.size)


def design_effect(y: Sequence[float], clusters: Sequence) -> dict:
    """Design effect of clustering for an outcome y (for example a 0/1 yes indicator).

    ICC from the one-way ANOVA estimator for unequal cluster sizes:
    icc = (MSB - MSW) / (MSB + (n0 - 1) MSW), n0 = (N - sum(n_i^2) / N) / (k - 1); then
    deff = 1 + (mean_cluster_size - 1) * icc with mean_cluster_size = N / k. Returns
    ``icc``, ``mean_cluster_size``, ``deff`` and ``n_clusters``. When every cluster is a
    singleton, icc is NaN and deff is 1.0; with one cluster or a constant outcome, icc and
    deff are NaN. The raw (possibly negative) ICC is reported.
    """
    yv = np.asarray(y, dtype=np.float64).ravel()
    cl = clusters if isinstance(clusters, pd.Series) else pd.Series(list(clusters))
    if len(cl) != yv.size:
        raise ValueError(f"y and clusters differ in length: {yv.size} vs {len(cl)}")
    if np.isnan(yv).any():
        raise ValueError("y contains NaN")
    res: dict[str, Any] = {"icc": _NAN, "mean_cluster_size": _NAN, "deff": _NAN, "n_clusters": 0}
    n_obs = yv.size
    if n_obs == 0:
        return res
    codes, uniques = _codes(cl.reset_index(drop=True), "clusters")
    k = len(uniques)
    res["n_clusters"] = k
    mbar = n_obs / k
    res["mean_cluster_size"] = mbar
    if n_obs == k:
        res["deff"] = 1.0
        return res
    if k < 2:
        return res
    sizes = np.bincount(codes, minlength=k).astype(np.float64)
    means = np.bincount(codes, weights=yv, minlength=k) / sizes
    grand = float(yv.mean())
    msb = float(np.sum(sizes * (means - grand) ** 2)) / (k - 1)
    msw = float(np.sum((yv - means[codes]) ** 2)) / (n_obs - k)
    n0 = (n_obs - float(np.sum(sizes**2)) / n_obs) / (k - 1)
    den = msb + (n0 - 1.0) * msw
    if not den > 0.0:
        return res
    icc = (msb - msw) / den
    res["icc"] = icc
    res["deff"] = 1.0 + (mbar - 1.0) * icc
    return res


def power_two_proportions(p1: float, p2: float, n1: float, n2: float, alpha: float = 0.05) -> float:
    """Power of the two-sided two-proportion z-test (normal approximation).

    Critical value from the pooled variance under H0, spread from the unpooled variance under
    H1: power = Phi((|p1 - p2| - z SE0) / SE1) + Phi((-|p1 - p2| - z SE0) / SE1). Gives 0.611 for
    0.50 vs 0.70 at 60/60, 0.816 for 0.50 vs 0.75 at 60/60 and 0.828 for 0.50 vs 0.70 at 100/100.
    """
    p1, p2, n1, n2 = float(p1), float(p2), float(n1), float(n2)
    a = _check_alpha(alpha)
    if not (0.0 <= p1 <= 1.0 and 0.0 <= p2 <= 1.0):
        raise ValueError("p1 and p2 must lie in [0, 1]")
    if not (n1 > 0.0 and n2 > 0.0):
        raise ValueError("n1 and n2 must be positive")
    z = float(special.ndtri(1.0 - a / 2.0))
    pbar = (n1 * p1 + n2 * p2) / (n1 + n2)
    se0 = math.sqrt(pbar * (1.0 - pbar) * (1.0 / n1 + 1.0 / n2))
    se1 = math.sqrt(p1 * (1.0 - p1) / n1 + p2 * (1.0 - p2) / n2)
    diff = abs(p1 - p2)
    if se1 == 0.0:
        return 1.0 if diff > z * se0 else 0.0
    return float(special.ndtr((diff - z * se0) / se1) + special.ndtr((-diff - z * se0) / se1))
