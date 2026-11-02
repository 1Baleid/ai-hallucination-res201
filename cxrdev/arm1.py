"""Arm 1: detector-gated answers and their controls (PROTOCOL section 7; H5 in section 6).

Template: CCD's expert-informed stage, applied to the cached first-token logit z. The expert s is
the MedSigLIP zero-shot ETT probability (S9, softmax over the CCD prompt pair):

    z' = z + beta * clip(log(s / (1 - s)), -gamma, gamma),  gamma = 5;  answer yes iff z' > 0.

Rows, all on the same arrays of one model under P0:

* greedy-text (the parsed greedy answer; unparsable and truncated outputs excluded as in
  ``metrics``), beta = 0 (yes iff z > 0, the baseline), fused beta = 1 (primary, fixed a priori),
  the dev-selected beta (from ``constants.BETA_GRID`` by dev-pool balanced accuracy, ties closest
  to 1), detector alone (s > tau, tau at Youden on the dev pool);
* shuffled expert: s permuted across the same A and B images, the null distribution of the gain
  and the one-sided p = (1 + #{null >= observed}) / (1 + n_perm) (H5 primary inference);
* the constant-bias curve z + b and the bias b* whose A-and-B yes-rate matches the fused rule's;
* disagreement-abstain (abstain when sign(z) differs from the detector's call; coverage and risk);
* PLUS: the oracle label (balanced accuracy 1.0 by construction, reference only) and the clipped
  oracle (s = the label, so z is moved by exactly +/- gamma), with the share of |z| > gamma.

Balanced accuracy (BA) is (P(yes | A) + P(no | B)) / 2 on A and B; U is scored with the same rule
(FAR_U). The yes-rate of a row is over A, B and U. Intervals: Wilson for rates and coverage,
patient-clustered BCa bootstrap for BA and for every difference (the H5 interval is
descriptive). The dev pool is never used for a test statistic. Labels of record: RANZCR CLiP
labels (43 trained labellers); ground truth is never a detector output.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cxrdev import constants as C
from cxrdev import metrics as M
from cxrdev import stats
from cxrdev.fileio import require_columns

__all__ = [
    "DETECTOR_COLUMNS", "balanced_accuracy", "constant_bias_curve", "detector_summary", "disagreement_abstain",
    "expert_logodds", "fuse", "join_detector", "matched_bias", "oracle_rows", "read_detector", "run_arm1",
    "select_beta_dev", "shuffled_expert_test", "youden_tau",
]

#: S9 columns (``pod/run_medsiglip.py``).
DETECTOR_COLUMNS: tuple[str, ...] = ("item_id", "nih_filename", "s", "logit_with", "logit_without", "model_id", "revision")
_NAN = float("nan")
_TIE = 12  # decimals used to compare balanced accuracies and yes-rates for ties


# ----------------------------------------------------------------------------- array helpers
def _vec(x: Any, name: str) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    return a


def _labels(y: Any, n: int | None = None) -> np.ndarray:
    a = np.asarray(y).reshape(-1)
    if a.dtype.kind == "b":
        a = a.astype(np.int64)
    if a.size and not np.all(np.isin(a, (0, 1))):
        raise ValueError("labels must be 0/1")
    a = a.astype(np.int64)
    if n is not None and a.size != n:
        raise ValueError(f"labels have length {a.size}, expected {n}")
    return a


def _finite(name: str, a: np.ndarray) -> None:
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains non-finite values (filter them first)")


def _ba_counts(yes: np.ndarray, pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    """Balanced accuracy of each row of a boolean decision matrix (rows = rules, columns = items)."""
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    hits = yes[:, pos].sum(axis=1)
    tn = (~yes[:, neg]).sum(axis=1)
    return 0.5 * (hits / n_pos + tn / n_neg)


# ----------------------------------------------------------------------------- the fusion rule
def expert_logodds(s: Any, eps: float = 1e-6) -> np.ndarray:
    """log(s / (1 - s)) with s clipped to [eps, 1 - eps] (NaN stays NaN)."""
    a = np.clip(_vec(s, "s"), eps, 1.0 - eps)
    return np.log(a) - np.log1p(-a)


def fuse(z: Any, s: Any, beta: float, gamma: float = C.GAMMA) -> np.ndarray:
    """Fused log-odds z + beta * clip(logodds(s), -gamma, gamma); answer yes iff the result is > 0."""
    zz = _vec(z, "z")
    ss = _vec(s, "s")
    if zz.shape != ss.shape:
        raise ValueError(f"z and s differ in length: {zz.size} vs {ss.size}")
    if float(beta) == 0.0:
        return zz.copy()
    return zz + float(beta) * np.clip(expert_logodds(ss), -float(gamma), float(gamma))


def balanced_accuracy(yes: Any, y: Any, no: Any | None = None) -> float:
    """(P(yes | y = 1) + P(no | y = 0)) / 2; ``no`` defaults to ``~yes`` (pass it when abstentions exist).

    NaN when either class is empty.
    """
    ya = np.asarray(yes, dtype=bool).reshape(-1)
    lab = _labels(y, ya.size)
    na = ~ya if no is None else np.asarray(no, dtype=bool).reshape(-1)
    if na.size != ya.size:
        raise ValueError("yes and no differ in length")
    pos, neg = lab == 1, lab == 0
    if not pos.any() or not neg.any():
        return _NAN
    return float(0.5 * (ya[pos].mean() + na[neg].mean()))


def youden_tau(s_dev: Any, y_dev: Any) -> float:
    """Threshold tau maximising TPR - FPR for the rule "positive iff s > tau" (Youden's J).

    Candidates are the midpoints between consecutive distinct scores (a single distinct score is
    returned as is). Ties in J go to the candidate closest to 0.5, then to the smaller one.
    Raises ValueError for non-finite scores or a missing class.
    """
    s = _vec(s_dev, "s_dev")
    y = _labels(y_dev, s.size)
    _finite("s_dev", s)
    pos, neg = np.sort(s[y == 1]), np.sort(s[y == 0])
    if pos.size == 0 or neg.size == 0:
        raise ValueError("youden_tau needs both classes in the dev pool")
    u = np.unique(s)
    if u.size == 1:
        return float(u[0])
    cand = 0.5 * (u[:-1] + u[1:])
    tpr = (pos.size - np.searchsorted(pos, cand, side="right")) / pos.size
    fpr = (neg.size - np.searchsorted(neg, cand, side="right")) / neg.size
    j = np.round(tpr - fpr, _TIE)
    best = np.flatnonzero(j == j.max())
    order = np.lexsort((cand[best], np.round(np.abs(cand[best] - 0.5), _TIE)))
    return float(cand[best[order[0]]])


def select_beta_dev(
    z_dev: Any, s_dev: Any, y_dev: Any, grid: Sequence[float] = C.BETA_GRID, gamma: float = C.GAMMA
) -> float:
    """The beta of ``grid`` with the highest dev-pool balanced accuracy of the fused rule (ties -> closest to 1)."""
    z = _vec(z_dev, "z_dev")
    s = _vec(s_dev, "s_dev")
    y = _labels(y_dev, z.size)
    _finite("z_dev", z)
    _finite("s_dev", s)
    if not (np.any(y == 1) and np.any(y == 0)):
        raise ValueError("select_beta_dev needs both classes in the dev pool")
    betas = [float(b) for b in grid]
    if not betas:
        raise ValueError("grid must not be empty")
    scores = [round(balanced_accuracy(fuse(z, s, b, gamma) > 0, y), _TIE) for b in betas]
    best = max(scores)
    tied = [b for b, sc in zip(betas, scores) if sc == best]
    return float(sorted(tied, key=lambda b: (abs(b - 1.0), b))[0])


# ----------------------------------------------------------------------------- shuffled expert (H5)
def shuffled_expert_test(
    z: Any,
    s: Any,
    y: Any,
    beta: float = C.BETA_PRIMARY,
    gamma: float = C.GAMMA,
    n_perm: int = C.N_PERM,
    seed: int | None = None,
) -> dict[str, Any]:
    """Shuffled-expert permutation test of the balanced-accuracy gain (one-sided, H5 primary).

    observed gain = BA(fused) - BA(z > 0) on the given items (A and B); the null permutes s across
    the same items ``n_perm`` times (seed ``derive_seed("permutation")`` by default) and
    p = (1 + #{null >= observed}) / (1 + n_perm). Gains are compared with a 1e-12 tolerance, so a
    permutation that reproduces the observed decisions counts as ">=". Raises ValueError for
    non-finite inputs or a missing class.
    """
    seed = C.derive_seed("permutation") if seed is None else int(seed)
    zz, ss = _vec(z, "z"), _vec(s, "s")
    yy = _labels(y, zz.size)
    if ss.size != zz.size:
        raise ValueError("z and s differ in length")
    _finite("z", zz)
    _finite("s", ss)
    n_perm = int(n_perm)
    if n_perm < 1:
        raise ValueError("n_perm must be at least 1")
    pos, neg = yy == 1, yy == 0
    if not pos.any() or not neg.any():
        raise ValueError("the permutation test needs both classes")
    lo = np.clip(expert_logodds(ss), -float(gamma), float(gamma))
    ba0 = float(_ba_counts((zz > 0)[None, :], pos, neg)[0])
    ba_f = float(_ba_counts((zz + float(beta) * lo > 0)[None, :], pos, neg)[0])
    observed = ba_f - ba0
    rng = np.random.default_rng(seed)
    n = zz.size
    null = np.empty(n_perm, dtype=np.float64)
    chunk = max(1, min(n_perm, 2_000_000 // max(n, 1)))
    base = np.arange(n)
    done = 0
    while done < n_perm:
        m = min(chunk, n_perm - done)
        perm = rng.permuted(np.tile(base, (m, 1)), axis=1)
        dec = zz[None, :] + float(beta) * lo[perm] > 0
        null[done:done + m] = _ba_counts(dec, pos, neg) - ba0
        done += m
    p = (1 + int(np.count_nonzero(null >= observed - 1e-12))) / (1 + n_perm)
    q = np.quantile(null, [0.025, 0.5, 0.975])
    return {
        "observed_gain": observed, "p_value": float(p), "n_perm": n_perm, "null_mean": float(null.mean()),
        "null_sd": float(null.std(ddof=1)) if n_perm > 1 else _NAN, "null_q025": float(q[0]),
        "null_q500": float(q[1]), "null_q975": float(q[2]), "ba_fused": ba_f, "ba_beta0": ba0, "beta": float(beta),
        "gamma": float(gamma), "n": int(n), "n_pos": int(pos.sum()), "n_neg": int(neg.sum()), "seed": seed,
        "sided": "one-sided (greater)",
    }


# ----------------------------------------------------------------------------- constant bias
def _default_grid(b_grid: Any | None) -> np.ndarray:
    return np.linspace(-10.0, 10.0, 401) if b_grid is None else _vec(b_grid, "b_grid")


def constant_bias_curve(z: Any, y: Any, b_grid: Any | None = None) -> pd.DataFrame:
    """Yes-rate and balanced accuracy of the rule "yes iff z + b > 0" for each b (default -10..10, step 0.05)."""
    zz = _vec(z, "z")
    yy = _labels(y, zz.size)
    _finite("z", zz)
    b = _default_grid(b_grid)
    dec = zz[None, :] + b[:, None] > 0
    pos, neg = yy == 1, yy == 0
    ba = _ba_counts(dec, pos, neg) if (pos.any() and neg.any()) else np.full(b.size, _NAN)
    yes_rate = dec.mean(axis=1) if zz.size else np.full(b.size, _NAN)
    return pd.DataFrame({"b": b, "yes_rate": yes_rate, "ba": ba})


def matched_bias(z: Any, y: Any, target_yes_rate: float, b_grid: Any | None = None) -> dict[str, Any]:
    """The grid bias b* whose yes-rate is closest to ``target_yes_rate`` (ties -> smallest |b|, then smallest b).

    Returns ``b``, ``yes_rate``, ``ba``, ``target_yes_rate`` and ``abs_diff``.
    """
    curve = constant_bias_curve(z, y, b_grid)
    if curve.empty:
        raise ValueError("b_grid must not be empty")
    diff = np.round(np.abs(curve["yes_rate"].to_numpy() - float(target_yes_rate)), _TIE)
    b = curve["b"].to_numpy()
    order = np.lexsort((b, np.round(np.abs(b), _TIE), diff))
    i = int(order[0])
    return {"b": float(b[i]), "yes_rate": float(curve["yes_rate"].iloc[i]), "ba": float(curve["ba"].iloc[i]),
            "target_yes_rate": float(target_yes_rate), "abs_diff": float(diff[i])}


# ----------------------------------------------------------------------------- disagreement-abstain and oracle
def disagreement_abstain(z: Any, s: Any, y: Any, tau: float, alpha: float = C.ALPHA) -> dict[str, Any]:
    """Abstain when sign(z) (yes iff z > 0) differs from the detector's call (yes iff s > tau).

    Returns ``n``, ``n_answered``, ``coverage`` (answered / n) and ``risk`` (errors among answered)
    as Wilson rate dicts, and ``tau``.
    """
    zz, ss = _vec(z, "z"), _vec(s, "s")
    yy = _labels(y, zz.size)
    if ss.size != zz.size:
        raise ValueError("z and s differ in length")
    model_yes = zz > 0
    answered = model_yes == (ss > float(tau))
    k = int(answered.sum())
    errors = int(np.count_nonzero(model_yes[answered] != (yy[answered] == 1)))
    return {"n": int(zz.size), "n_answered": k, "coverage": M.rate(k, zz.size, alpha), "risk": M.rate(errors, k, alpha),
            "tau": float(tau)}


def oracle_rows(z: Any, y: Any, gamma: float = C.GAMMA) -> dict[str, Any]:
    """PLUS oracle references: the oracle label (BA 1.0 by construction) and the clipped oracle.

    The clipped oracle is the fusion rule with beta = 1 and s = the label, so z moves by exactly
    +/- gamma; ``share_abs_z_gt_gamma`` is the share of items whose answer no expert can flip.
    """
    zz = _vec(z, "z")
    yy = _labels(y, zz.size)
    clipped = fuse(zz, yy.astype(np.float64), 1.0, gamma) > 0
    return {
        "oracle_label_ba": balanced_accuracy(yy == 1, yy), "clipped_oracle_ba": balanced_accuracy(clipped, yy),
        "share_abs_z_gt_gamma": float(np.mean(np.abs(zz) > float(gamma))) if zz.size else _NAN,
        "n": int(zz.size), "gamma": float(gamma),
    }


# ----------------------------------------------------------------------------- detector scores (S9)
def read_detector(path: str | Path) -> pd.DataFrame:
    """Read ``detector_scores.csv`` (S9) with text ids and float scores."""
    df = pd.read_csv(path, dtype={"item_id": str, "nih_filename": str, "model_id": str, "revision": str, "dtype": str},
                     keep_default_na=False, na_values={"s": [""], "logit_with": [""], "logit_without": [""]})
    require_columns(df, DETECTOR_COLUMNS, str(path))
    for c in ("s", "logit_with", "logit_without"):
        df[c] = pd.to_numeric(df[c], errors="raise").astype(np.float64)
    return df


def join_detector(df: pd.DataFrame, detector: pd.DataFrame) -> pd.DataFrame:
    """``df`` with the detector score ``s`` joined on ``item_id`` (NaN where an item has no score).

    Raises ValueError when ``df`` already has an ``s`` column, when an item has two scores, or
    when a score lies outside [0, 1].
    """
    require_columns(detector, ("item_id", "s"), "detector scores (S9)")
    if "s" in df.columns:
        raise ValueError("the frame already has a column 's'")
    det = detector.loc[:, ["item_id", "s"]].copy()
    det["item_id"] = det["item_id"].astype(str)
    det["s"] = pd.to_numeric(det["s"], errors="raise").astype(np.float64)
    if bool(det["item_id"].duplicated().any()):
        raise ValueError("detector scores: an item_id appears more than once")
    sv = det["s"].to_numpy()
    if bool(np.any(np.isfinite(sv) & ((sv < 0.0) | (sv > 1.0)))):
        raise ValueError("detector scores: s must lie in [0, 1]")
    out = df.copy()
    out["item_id"] = out["item_id"].astype(str)
    return out.merge(det, on="item_id", how="left", validate="many_to_one", sort=False)


def detector_summary(
    items: pd.DataFrame,
    detector: pd.DataFrame,
    exclusions: pd.DataFrame | Iterable[str] | None = None,
    n_boot: int = C.N_BOOT,
    seed: int | None = None,
    alpha: float = C.ALPHA,
) -> dict[str, Any]:
    """The detector as an automatic verifier: dev-pool AUROC (clustered BCa), tau, agreement per arm.

    Agreement = the call "s > tau" equals the RANZCR label, per test arm (retained items with a
    score), with Wilson intervals; tau is Youden's threshold on the dev pool. The detector never
    excludes, flags or selects an item.
    """
    seed = C.derive_seed("bootstrap") if seed is None else int(seed)
    require_columns(items, ("item_id", "arm", "ett", "nih_patient_id"), "items")
    it = M.apply_exclusions(items, exclusions)
    joined = join_detector(it.loc[:, ["item_id", "arm", "ett", "nih_patient_id"]], detector)
    s = joined["s"].to_numpy(dtype=np.float64)
    fin = np.isfinite(s)
    arm = joined["arm"].astype(str).to_numpy()
    y = joined["ett"].to_numpy(dtype=np.int64)
    dev = (arm == "DEV") & fin
    out: dict[str, Any] = {"n_scored": int(fin.sum()), "n_missing": int((~fin).sum()),
                           "model_id": sorted(set(detector["model_id"].astype(str))) if "model_id" in detector else [],
                           "revision": sorted(set(detector["revision"].astype(str))) if "revision" in detector else []}
    if dev.any() and np.any(y[dev] == 1) and np.any(y[dev] == 0):
        fr = pd.DataFrame({"cl": joined["nih_patient_id"].to_numpy(dtype=np.int64)[dev], "a": y[dev] == 1, "z": s[dev]})
        auc = M.clustered_estimate(fr, M.auroc_stat, n_boot, seed, alpha, strata=False)
        auc.pop("p_value", None)
        auc.pop("p_flag", None)
        out["dev_auroc"] = {**auc, "n": int(dev.sum()), "n_pos": int(np.sum(y[dev] == 1)),
                            "n_neg": int(np.sum(y[dev] == 0))}
        tau = youden_tau(s[dev], y[dev])
        out["tau"] = tau
        agree: dict[str, Any] = {}
        for a in M.TEST_ARMS:
            m = (arm == a) & fin
            k = int(np.count_nonzero((s[m] > tau) == (y[m] == 1)))
            agree[a] = M.rate(k, int(m.sum()), alpha)
        out["agreement"] = agree
    else:
        out["dev_auroc"] = None
        out["tau"] = None
        out["agreement"] = None
        out["skipped_reason"] = "no dev-pool scores of both classes"
    return out


# ----------------------------------------------------------------------------- run_arm1
def _ba_stat_factory(yes_col: str, no_col: str | None = None, mask_col: str | None = None) -> Callable[[pd.DataFrame], float]:
    """A fast stat_fn: balanced accuracy of a decision column (optionally restricted to a mask column)."""

    def stat(d: pd.DataFrame) -> float:
        a = d["a"].to_numpy(dtype=bool)
        yes = d[yes_col].to_numpy(dtype=bool)
        no = ~yes if no_col is None else d[no_col].to_numpy(dtype=bool)
        if mask_col is not None:
            m = d[mask_col].to_numpy(dtype=bool)
            a, yes, no = a[m], yes[m], no[m]
        na = int(a.sum())
        nb = a.size - na
        if na == 0 or nb == 0:
            return _NAN
        return 0.5 * (float(yes[a].sum()) / na + float(no[~a].sum()) / nb)

    return stat


def _delta(f1: Callable[[pd.DataFrame], float], f2: Callable[[pd.DataFrame], float]) -> Callable[[pd.DataFrame], float]:
    return lambda d: f1(d) - f2(d)


def _row_metrics(yes: np.ndarray, no: np.ndarray, answered: np.ndarray, arm: np.ndarray, alpha: float) -> dict[str, Any]:
    """hit_A, FAR_B, FAR_U, yes-rate (A, B and U) and the BA point estimate of a rule on its answered items."""
    is_a, is_b, is_u = (arm == "A") & answered, (arm == "B") & answered, (arm == "U") & answered
    hit = M.rate(int(np.count_nonzero(yes & is_a)), int(is_a.sum()), alpha)
    far_b = M.rate(int(np.count_nonzero(yes & is_b)), int(is_b.sum()), alpha)
    spec_k = int(np.count_nonzero(no & is_b))
    ba = 0.5 * (hit["estimate"] + (spec_k / int(is_b.sum()))) if is_a.any() and is_b.any() else _NAN
    return {
        "hit_A": hit, "FAR_B": far_b,
        "FAR_U": M.rate(int(np.count_nonzero(yes & is_u)), int(is_u.sum()), alpha),
        "yes_rate": M.rate(int(np.count_nonzero(yes & answered)), int(answered.sum()), alpha),
        "ba_point": ba, "n": int(answered.sum()), "n_A": int(is_a.sum()), "n_B": int(is_b.sum()), "n_U": int(is_u.sum()),
    }


def run_arm1(
    test_df: pd.DataFrame,
    dev_df: pd.DataFrame | None,
    model_key: str,
    n_perm: int = C.N_PERM,
    n_boot: int = C.N_BOOT,
    seed: int | None = None,
    *,
    boot_seed: int | None = None,
    exclusions: pd.DataFrame | Iterable[str] | None = None,
    prompt_id: str = "P0",
    gamma: float = C.GAMMA,
    beta_primary: float = C.BETA_PRIMARY,
    beta_grid: Sequence[float] = C.BETA_GRID,
    b_grid: Any | None = None,
    alpha: float = C.ALPHA,
    with_oracle: bool = True,
) -> dict[str, Any]:
    """Every Arm 1 row of one model under P0 (PROTOCOL section 7).

    ``test_df`` and ``dev_df`` are scoring frames with the detector score joined (``join_detector``);
    they may hold several models, prompts and arms (the same frame can be passed twice): the test
    rows are this model's A, B and U items under ``prompt_id`` after exclusions, the dev rows its
    DEV items under P0. Items without a finite z or s are dropped and counted. ``seed`` drives the
    permutations (default ``derive_seed("permutation")``), ``boot_seed`` the clustered bootstrap
    (default ``derive_seed("bootstrap")``).

    Returns a JSON-serialisable dict: ``rows`` (greedy_text, beta0, fused_beta1, dev_beta,
    detector_alone, shuffled_null, matched_bias, disagree_abstain, oracle_label, clipped_oracle;
    each with ``ba`` {estimate, lo, hi, method, n}, ``delta_ba`` vs beta = 0 on the same items,
    ``hit_A``, ``FAR_B``, ``FAR_U``, ``yes_rate`` and ``n``), ``gain`` (fused minus beta = 0, the H5
    estimate with its descriptive clustered BCa interval), ``shuffled_expert``, ``matched_bias``,
    ``disagreement_abstain``, ``oracle``, ``agreement_greedy_beta0`` (share of valid greedy answers
    that equal the beta = 0 decision; an abstention counts as disagreement), ``tau``, ``beta_dev``,
    ``dev`` and the counts. Rows that need the dev pool are skipped (listed in ``skipped``) when it
    lacks either class; ``skipped_reason`` is set when A or B is empty.
    """
    seed = C.derive_seed("permutation") if seed is None else int(seed)
    boot_seed = C.derive_seed("bootstrap") if boot_seed is None else int(boot_seed)
    need = ("model_key", "prompt_id", "condition", "arm", "item_id", "z", "s", "y", "parsed", "valid", "nih_patient_id")
    if "s" not in test_df.columns:
        raise ValueError("test_df has no detector score column 's'; join it with arm1.join_detector first")
    require_columns(test_df, need, "test_df")
    t = M.rows_for(test_df, model_key, prompt_id, M.TEST_ARMS, exclusions, condition="image")
    z_all = t["z"].to_numpy(dtype=np.float64)
    s_all = t["s"].to_numpy(dtype=np.float64)
    fin_z, fin_s = np.isfinite(z_all), np.isfinite(s_all)
    keep = fin_z & fin_s
    result: dict[str, Any] = {
        "model_key": model_key, "prompt_id": prompt_id, "gamma": float(gamma), "beta_primary": float(beta_primary),
        "beta_grid": [float(b) for b in beta_grid], "n_retained": int(len(t)), "n_dropped_no_z": int((~fin_z).sum()),
        "n_dropped_no_score": int((fin_z & ~fin_s).sum()), "n_perm": int(n_perm), "n_boot": int(n_boot), "seed": seed,
        "boot_seed": boot_seed, "skipped": [], "skipped_reason": "",
    }
    t = t.loc[keep]
    arm = t["arm"].astype(str).to_numpy()
    z = t["z"].to_numpy(dtype=np.float64)
    s = t["s"].to_numpy(dtype=np.float64)
    y = t["y"].to_numpy(dtype=np.int64)
    patient = t["nih_patient_id"].to_numpy(dtype=np.int64)
    parsed = t["parsed"].astype(str).to_numpy()
    valid = t["valid"].to_numpy(dtype=bool)
    is_a, is_b, is_u = arm == "A", arm == "B", arm == "U"
    result.update(n_A=int(is_a.sum()), n_B=int(is_b.sum()), n_U=int(is_u.sum()), n=int(len(t)))
    if not is_a.any() or not is_b.any():
        result["skipped_reason"] = "no A or no B item with a finite z and a detector score"
        return result

    # ---- dev pool: tau, beta_dev
    tau: float | None = None
    beta_dev: float | None = None
    dev_info: dict[str, Any] | None = None
    if dev_df is not None and "s" in dev_df.columns:
        d = dev_df.loc[(dev_df["model_key"].astype(str) == model_key) & (dev_df["prompt_id"].astype(str) == "P0")
                       & (dev_df["condition"].astype(str) == "image") & (dev_df["arm"].astype(str) == "DEV")]
        zd, sd, yd = d["z"].to_numpy(dtype=np.float64), d["s"].to_numpy(dtype=np.float64), d["y"].to_numpy(dtype=np.int64)
        fd = np.isfinite(zd) & np.isfinite(sd)
        zd, sd, yd = zd[fd], sd[fd], yd[fd]
        if zd.size and np.any(yd == 1) and np.any(yd == 0):
            tau = youden_tau(sd, yd)
            beta_dev = select_beta_dev(zd, sd, yd, beta_grid, gamma)
            dev_info = {
                "n": int(zd.size), "n_pos": int(np.sum(yd == 1)), "n_neg": int(np.sum(yd == 0)),
                "n_dropped": int((~fd).sum()), "auroc_s": stats.auroc(sd[yd == 1], sd[yd == 0]),
                "auroc_z": stats.auroc(zd[yd == 1], zd[yd == 0]), "tau": tau, "beta_dev": beta_dev,
                "ba_by_beta": {str(float(b)): balanced_accuracy(fuse(zd, sd, b, gamma) > 0, yd) for b in beta_grid},
                "ba_beta0": balanced_accuracy(zd > 0, yd),
            }
    if dev_info is None:
        result["skipped"].append({"row": "dev_beta, detector_alone, disagree_abstain",
                                  "reason": "no dev-pool P0 rows of both classes with z and a detector score"})
    result.update(tau=tau, beta_dev=beta_dev, dev=dev_info)

    # ---- decisions on every test item
    ab = is_a | is_b
    rules: dict[str, np.ndarray] = {"beta0": z > 0, "fused": fuse(z, s, beta_primary, gamma) > 0}
    if beta_dev is not None:
        rules["dev_beta"] = fuse(z, s, beta_dev, gamma) > 0
    if tau is not None:
        rules["detector"] = s > tau
    target = float(rules["fused"][ab].mean())
    mb = matched_bias(z[ab], y[ab], target, b_grid)
    rules["matched"] = z + mb["b"] > 0
    if with_oracle:
        rules["oracle"] = y == 1
        rules["clipped"] = fuse(z, y.astype(np.float64), 1.0, gamma) > 0
    g_yes = valid & (parsed == "yes")
    g_no = valid & (parsed == "no")
    agree_k = int(np.count_nonzero((g_yes & rules["beta0"]) | (g_no & ~rules["beta0"])))
    result["agreement_greedy_beta0"] = M.rate(agree_k, int(valid.sum()), alpha)

    frame = pd.DataFrame({"cl": patient[ab], "a": is_a[ab], "gv": valid[ab], "gy": g_yes[ab], "gn": g_no[ab],
                          **{f"r_{k}": v[ab] for k, v in rules.items()}})
    disagree_ok = tau is not None
    if disagree_ok:
        answered_d = rules["beta0"] == rules["detector"]
        frame["ans_d"] = answered_d[ab]

    def boot(stat: Callable[[pd.DataFrame], float]) -> dict[str, Any]:
        res = M.clustered_estimate(frame, stat, n_boot, boot_seed, alpha, strata=False)
        return {k: res[k] for k in ("estimate", "lo", "hi", "method", "n_clusters")} | (
            {"p_value": res["p_value"], "p_flag": res["p_flag"]} if "p_value" in res else {})

    f_b0 = _ba_stat_factory("r_beta0")
    rows: list[dict[str, Any]] = []
    everyone = np.ones(len(t), dtype=bool)

    def logit_row(row_id: str, label: str, rule_key: str, param: float | None, param_name: str | None, note: str) -> dict:
        dec = rules[rule_key]
        met = _row_metrics(dec, ~dec, everyone, arm, alpha)
        f = _ba_stat_factory(f"r_{rule_key}")
        ba = {**boot(f), "n": met["n_A"] + met["n_B"]}
        delta = None if rule_key == "beta0" else {**boot(_delta(f, f_b0)), "n": met["n_A"] + met["n_B"]}
        return {"row": row_id, "label": label, "param": param, "param_name": param_name, "ba": ba, "delta_ba": delta,
                **{k: met[k] for k in ("hit_A", "FAR_B", "FAR_U", "yes_rate", "n", "n_A", "n_B", "n_U")},
                "coverage": None, "risk": None, "p_value": None, "note": note}

    # greedy text (valid outputs only; beta0 restricted to the same items for the delta)
    met = _row_metrics(g_yes, g_no, valid, arm, alpha)
    f_g = _ba_stat_factory("gy", "gn", "gv")
    rows.append({"row": "greedy_text", "label": "Greedy text answer", "param": None, "param_name": None,
                 "ba": {**boot(f_g), "n": met["n_A"] + met["n_B"]},
                 "delta_ba": {**boot(_delta(f_g, _ba_stat_factory("r_beta0", None, "gv"))), "n": met["n_A"] + met["n_B"]},
                 **{k: met[k] for k in ("hit_A", "FAR_B", "FAR_U", "yes_rate", "n", "n_A", "n_B", "n_U")},
                 "coverage": None, "risk": None, "p_value": None,
                 "note": "valid outputs only; delta vs beta = 0 on the same items"})
    rows.append(logit_row("beta0", "Logit rule, beta = 0 (baseline)", "beta0", 0.0, "beta", "yes iff z > 0"))
    perm = shuffled_expert_test(z[ab], s[ab], y[ab], beta_primary, gamma, n_perm, seed)
    fused = logit_row("fused_beta1", f"Fused, beta = {beta_primary:g} (primary)", "fused", float(beta_primary), "beta",
                      "H5: one-sided shuffled-expert permutation p; interval descriptive")
    fused["p_value"] = perm["p_value"]
    rows.append(fused)
    result["gain"] = {**fused["delta_ba"], "role": "descriptive"}
    result["shuffled_expert"] = perm
    if beta_dev is not None:
        rows.append(logit_row("dev_beta", f"Fused, dev-selected beta = {beta_dev:g}", "dev_beta", float(beta_dev), "beta",
                              f"beta chosen on the dev pool from {tuple(float(b) for b in beta_grid)}"))
    if tau is not None:
        rows.append(logit_row("detector_alone", "Detector alone (s > tau)", "detector", float(tau), "tau",
                              "tau at Youden on the dev pool"))
    rows.append({"row": "shuffled_null", "label": "Shuffled expert (null of the gain)", "param": float(beta_primary),
                 "param_name": "beta", "ba": None, "delta_ba": {"estimate": perm["null_mean"], "lo": perm["null_q025"],
                                                              "hi": perm["null_q975"], "method": "permutation quantiles",
                                                              "n": perm["n"]},
                 "hit_A": None, "FAR_B": None, "FAR_U": None, "yes_rate": None, "n": perm["n"], "n_A": perm["n_pos"],
                 "n_B": perm["n_neg"], "n_U": 0, "coverage": None, "risk": None, "p_value": perm["p_value"],
                 "note": f"{perm['n_perm']} permutations of s across the A and B images; mean and 95% range of null gains"})
    mrow = logit_row("matched_bias", f"Constant bias b* = {mb['b']:g} (yes-rate matched)", "matched", float(mb["b"]), "b",
                     "b* matches the fused rule's yes-rate on A and B")
    f_m = _ba_stat_factory("r_matched")
    mrow["fused_minus_matched"] = {**boot(_delta(_ba_stat_factory("r_fused"), f_m)), "n": mrow["n_A"] + mrow["n_B"]}
    rows.append(mrow)
    result["matched_bias"] = mb
    if disagree_ok:
        ans = rules["beta0"] == rules["detector"]
        dmet = _row_metrics(rules["beta0"] & ans, ~rules["beta0"] & ans, ans, arm, alpha)
        da = disagreement_abstain(z, s, y, tau, alpha)
        f_d = _ba_stat_factory("r_beta0", None, "ans_d")
        rows.append({"row": "disagree_abstain", "label": "Disagreement-abstain", "param": float(tau), "param_name": "tau",
                     "ba": {**boot(f_d), "n": dmet["n_A"] + dmet["n_B"]}, "delta_ba": None,
                     **{k: dmet[k] for k in ("hit_A", "FAR_B", "FAR_U", "yes_rate", "n", "n_A", "n_B", "n_U")},
                     "coverage": da["coverage"], "risk": da["risk"], "p_value": None,
                     "note": "abstain when sign(z) differs from s > tau; metrics over answered items"})
        result["disagreement_abstain"] = da
    else:
        result["disagreement_abstain"] = None
    if with_oracle:
        orc = oracle_rows(z[ab], y[ab], gamma)
        orc["share_abs_z_gt_gamma_all"] = float(np.mean(np.abs(z) > float(gamma)))
        result["oracle"] = orc
        om = _row_metrics(rules["oracle"], ~rules["oracle"], everyone, arm, alpha)
        rows.append({"row": "oracle_label", "label": "Oracle label (reference)", "param": None, "param_name": None,
                     "ba": {"estimate": orc["oracle_label_ba"], "lo": _NAN, "hi": _NAN, "method": "none",
                            "n": om["n_A"] + om["n_B"]}, "delta_ba": None,
                     **{k: om[k] for k in ("hit_A", "FAR_B", "FAR_U", "yes_rate", "n", "n_A", "n_B", "n_U")},
                     "coverage": None, "risk": None, "p_value": None, "note": "answer = RANZCR label; BA 1.0 by construction"})
        crow = logit_row("clipped_oracle", f"Clipped oracle (z +/- {gamma:g})", "clipped", float(gamma), "gamma",
                         f"share of A and B items with |z| > gamma: {orc['share_abs_z_gt_gamma']:.3f}")
        rows.append(crow)
    else:
        result["oracle"] = None
    result["rows"] = rows
    return result
