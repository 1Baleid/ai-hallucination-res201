"""Primary tests H1, H3 and H5 (and PLUS H2) with Holm within each family (PROTOCOL section 6).

One primary test per hypothesis and model; a family is the per-model primaries of one hypothesis,
and Holm's step-down adjustment is applied within it (``holm_family``). Bootstrap p-values come
from CI inversion of the patient-clustered BCa interval (``stats.clustered_ci``, cluster = NIH
patient id); cells with fewer than ``constants.MIN_CELL_N`` items get an interval and no test.

* H1 co-occurrence prior: yes-rate under P0 on U with at least one other labelled line minus U
  with none (or, under the registered substitute, two or more other lines minus exactly one),
  two-sided, resampling within strata so each replicate keeps the stratum sizes. The same
  registered test under every rung (the U arm stays 60 + 60 under every rung; PROTOCOL v0.4).
* H3 penalty effect: exact McNemar on the abstain indicator, P2hi vs P1, over B and U items valid
  under both prompts; whenever any NIH patient contributes two pairs (the "two_per_patient" rung,
  or the "n80" rung built from it; PROTOCOL v0.4) the clustered BCa interval of the paired
  difference is primary (CI-inversion p) and McNemar on one random B per patient is a check.
* H5 Arm 1: the one-sided shuffled-expert permutation p of the balanced-accuracy gain (fused,
  beta = 1, minus the beta = 0 logit rule); its clustered BCa interval is descriptive. The numbers
  come from ``arm1.run_arm1``.
* PLUS H2: CheXOne minus Qwen2.5-VL-3B paired differences in the hallucination-free rate
  (1 - FAR on B and U) and in balanced accuracy on A and B, two-sided.

Every function takes the ``metrics.scoring_frame`` of the parsed results; RANZCR CLiP labels
(43 trained labellers) are the labels of record.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from cxrdev import constants as C
from cxrdev import metrics as M
from cxrdev import stats

__all__ = ["h1", "h1_secondary", "h2", "h3", "h3_secondary", "h5", "holm_family", "max_pairs_per_patient",
           "resolve_h1_contrast"]

_NAN = float("nan")


def _seed(seed: int | None) -> int:
    return C.derive_seed("bootstrap") if seed is None else int(seed)


def _diff_stat(d: pd.DataFrame) -> float:
    """Mean of ``v`` where ``e`` minus mean of ``v`` where not ``e`` (NaN when a group is empty)."""
    e = d["e"].to_numpy(dtype=bool)
    v = d["v"].to_numpy(dtype=np.float64)
    ne = int(e.sum())
    if ne == 0 or ne == e.size:
        return _NAN
    return float(v[e].mean() - v[~e].mean())


def _untested(base: Mapping[str, Any], reason: str) -> dict[str, Any]:
    return {**base, "estimate": _NAN, "lo": _NAN, "hi": _NAN, "method": "none", "p_value": _NAN, "p_flag": "na",
            "tested": False, "reason": reason, "n": 0}


# ----------------------------------------------------------------------------- H1
def resolve_h1_contrast(
    strata_present: Iterable[str], contrast: Mapping[str, Any] | Sequence[str] | None = None
) -> tuple[str, str, bool] | None:
    """(exposed, reference, substitute) of the registered H1 contrast.

    ``contrast`` is the selection report's ``u.h1_contrast`` ({"exposed": .., "reference": ..}) or an
    (exposed, reference) pair; when given it is authoritative (substitute = (ge2, eq1)), and None is
    returned if either of its strata is missing from ``strata_present``. Without it the contrast is
    inferred from the strata (``metrics.h1_contrast``: ge2 vs eq1 when both exist, else line vs
    noline). Any other U stratum (the descriptive no-line stratum under the substitute) is never part
    of the contrast.
    """
    present = {str(s) for s in strata_present}
    if contrast is None:
        return M.h1_contrast(present)
    if isinstance(contrast, Mapping):
        exposed, reference = str(contrast["exposed"]), str(contrast["reference"])
    else:
        exposed, reference = (str(x) for x in contrast)
    if (exposed, reference) not in (("line", "noline"), ("ge2", "eq1")):
        raise ValueError(f"unknown H1 contrast {exposed!r} vs {reference!r}")
    if not {exposed, reference} <= present:
        return None
    return exposed, reference, (exposed, reference) == ("ge2", "eq1")


def h1(
    df: pd.DataFrame,
    model_key: str,
    n_boot: int = C.N_BOOT,
    seed: int | None = None,
    *,
    exclusions: pd.DataFrame | Iterable[str] | None = None,
    prompt_id: str = "P0",
    rung: str | None = None,
    alpha: float = C.ALPHA,
    contrast: Mapping[str, Any] | Sequence[str] | None = None,
) -> dict[str, Any]:
    """H1: yes-rate(line or ge2) minus yes-rate(noline or eq1) on U under P0, two-sided.

    The contrast is the registered one: ``contrast`` from the selection report (``u.h1_contrast``)
    when given, else inferred from the U strata of the item list (``resolve_h1_contrast``: ge2 vs eq1
    when those strata exist, else line vs noline). Under the substitute the registered contrast is
    U(ge2) vs U(eq1); the remaining no-line U items are descriptive and are never pooled into either
    side (their count is reported as ``n_descriptive``). Only valid
    outputs count (unparsable and truncated excluded). The interval is the patient-clustered BCa
    bootstrap with ``strata_col`` = stratum (stratum sizes fixed in each resample); the p-value is
    its CI inversion. The test is the same under every rung, "n80" included (the U arm stays
    60 + 60 under every rung; PROTOCOL v0.4); ``rung`` is recorded only. No test (p = NaN,
    ``tested`` False) when a stratum has fewer than ``constants.MIN_CELL_N`` valid items;
    ``p_value_descriptive`` keeps the CI-inversion p in that case.
    """
    seed = _seed(seed)
    base: dict[str, Any] = {"hypothesis": "H1", "family": "H1", "model_key": model_key, "prompt_id": prompt_id,
                            "sided": "two-sided", "n_boot": int(n_boot), "seed": seed, "rung": rung}
    u_strata = df.loc[df["arm"].astype(str) == "U", "stratum"].astype(str).unique()
    resolved = resolve_h1_contrast(u_strata, contrast)
    if resolved is None:
        return _untested(base, "the U strata of the H1 contrast are missing")
    exposed, reference, substitute = resolved
    base.update(exposed=exposed, reference=reference, substitute=bool(substitute),
                contrast=f"U {exposed} minus U {reference}",
                contrast_source="selection report" if contrast is not None else "item strata")
    sub = M.rows_for(df, model_key, prompt_id, ("U",), exclusions, condition="image")
    in_contrast = sub["stratum"].astype(str).isin([exposed, reference])
    descriptive = sub.loc[~in_contrast]
    base.update(n_descriptive=int(descriptive["item_id"].nunique()),
                descriptive_strata=sorted(set(descriptive["stratum"].astype(str))))
    sub = sub.loc[in_contrast & sub["valid"].astype(bool)]
    e = (sub["stratum"].astype(str) == exposed).to_numpy(dtype=bool)
    yes = (sub["parsed"].astype(str) == "yes").to_numpy(dtype=bool)
    frame = pd.DataFrame({"cl": sub["nih_patient_id"].to_numpy(dtype=np.int64), "st": sub["stratum"].astype(str).to_numpy(),
                          "e": e, "v": yes.astype(np.float64)})
    n_e, n_r = int(e.sum()), int((~e).sum())
    rate_e = M.rate(int((yes & e).sum()), n_e, alpha)
    rate_r = M.rate(int((yes & ~e).sum()), n_r, alpha)
    if n_e == 0 or n_r == 0:
        return {**_untested(base, "a stratum of the H1 contrast has no valid item"), "rate_exposed": rate_e,
                "rate_reference": rate_r, "n_exposed": n_e, "n_reference": n_r}
    res = M.clustered_estimate(frame, _diff_stat, n_boot, seed, alpha, strata=True)
    reasons = []
    if min(n_e, n_r) < C.MIN_CELL_N:
        reasons.append(f"a stratum has fewer than {C.MIN_CELL_N} valid items (interval only, no test)")
    tested = not reasons and math.isfinite(res.get("p_value", _NAN))
    return {
        **base, "estimate": res["estimate"], "lo": res["lo"], "hi": res["hi"], "method": res["method"],
        "p_value": res.get("p_value", _NAN) if tested else _NAN, "p_flag": res.get("p_flag", "na") if tested else "na",
        "p_value_descriptive": res.get("p_value", _NAN), "tested": bool(tested), "reason": "; ".join(reasons),
        "n": int(len(frame)), "n_clusters": res["n_clusters"], "n_exposed": n_e, "n_reference": n_r,
        "rate_exposed": rate_e, "rate_reference": rate_r,
    }


def h1_secondary(
    df: pd.DataFrame,
    model_key: str,
    n_boot: int = C.N_BOOT,
    seed: int | None = None,
    *,
    exclusions: pd.DataFrame | Iterable[str] | None = None,
    summary: Mapping[str, Any] | None = None,
    blind: pd.DataFrame | None = None,
    alpha: float = C.ALPHA,
) -> dict[str, Any]:
    """H1 secondary results: FAR_B minus FAR_U (clustered BCa, descriptive), persistence, B strata, blind P(yes).

    ``summary`` is the P0 ``metrics.summarise`` result of the model (for persistence and the B
    strata); ``blind`` the model's ``metrics.blind_rows``.
    """
    seed = _seed(seed)
    sub = M.rows_for(df, model_key, "P0", ("B", "U"), exclusions, condition="image")
    sub = sub.loc[sub["valid"].astype(bool)]
    e = (sub["arm"].astype(str) == "B").to_numpy(dtype=bool)
    frame = pd.DataFrame({"cl": sub["nih_patient_id"].to_numpy(dtype=np.int64), "st": sub["set"].astype(str).to_numpy(),
                          "e": e, "v": (sub["parsed"].astype(str) == "yes").to_numpy(dtype=np.float64)})
    if e.any() and (~e).any():
        diff = M.clustered_estimate(frame, _diff_stat, n_boot, seed, alpha, strata=True)
    else:
        diff = {"estimate": _NAN, "lo": _NAN, "hi": _NAN, "method": "none", "n_clusters": 0}
    diff = {**diff, "n_B": int(e.sum()), "n_U": int((~e).sum()), "role": "descriptive"}
    out: dict[str, Any] = {"model_key": model_key, "far_b_minus_far_u": diff}
    if summary is not None:
        for key in ("persistence", "pair_correct", "FAR_B_line", "FAR_B_noline", "FAR_B", "FAR_U"):
            if key in summary:
                out[key] = summary[key]
    if blind is not None and len(blind):
        out["blind"] = [
            {"prompt_id": str(r.prompt_id), "condition": str(r.condition), "p_yes_norm": float(r.p_yes_norm),
             "greedy": str(r.greedy)}
            for r in blind.itertuples(index=False)
        ]
    return out


# ----------------------------------------------------------------------------- H3
def h3(
    df: pd.DataFrame,
    model_key: str,
    rung: str = "standard",
    *,
    exclusions: pd.DataFrame | Iterable[str] | None = None,
    n_boot: int = C.N_BOOT,
    seed: int | None = None,
    alpha: float = C.ALPHA,
    prompts: tuple[str, str] = ("P1", "P2hi"),
) -> dict[str, Any]:
    """H3: abstention under P2hi vs P1 on B and U items valid under both prompts.

    b = abstain under P2hi only, c = abstain under P1 only, estimate = (b - c) / n (the paired
    difference in abstention rates). One pair per patient: p = ``stats.mcnemar_exact(b, c)``
    (primary) and a descriptive clustered BCa interval of the paired difference. Whenever any NIH
    patient contributes two pairs (counted from the A/B pair ids per ``nih_patient_id`` in ``df``;
    the "two_per_patient" rung, or the "n80" rung built from it; PROTOCOL v0.4) or ``rung`` is
    "two_per_patient": the clustered BCa interval (strata = design set, cluster = NIH patient) is
    primary with its CI-inversion p, and McNemar on one random B per patient (seed ``seed``) plus
    every U item is reported as ``mcnemar_check``. ``n_dropped`` counts retained B and U items
    that lack a valid output under either prompt (with a breakdown). No test when
    n < ``constants.MIN_CELL_N``.
    """
    seed = _seed(seed)
    p_ref, p_pen = prompts
    max_pairs = max_pairs_per_patient(df)
    clustered = rung == "two_per_patient" or max_pairs >= 2
    base: dict[str, Any] = {"hypothesis": "H3", "family": "H3", "model_key": model_key, "prompts": [p_ref, p_pen],
                            "sided": "two-sided", "rung": rung, "n_boot": int(n_boot), "seed": seed,
                            "max_pairs_per_patient": max_pairs, "clustered_primary": bool(clustered)}
    pick = ((df["model_key"].astype(str) == model_key) & (df["condition"].astype(str) == "image")
            & df["arm"].astype(str).isin(["B", "U"]) & df["prompt_id"].astype(str).isin([p_ref, p_pen]))
    sub = M.apply_exclusions(df.loc[pick], exclusions)
    candidates = sorted(set(sub["item_id"].astype(str)))
    info: dict[str, dict[str, tuple[bool, bool]]] = {p_ref: {}, p_pen: {}}
    meta: dict[str, tuple[int, str, str]] = {}
    for r in sub.itertuples(index=False):
        info[str(r.prompt_id)][str(r.item_id)] = (bool(r.valid), str(r.parsed) == "abstain")
        meta[str(r.item_id)] = (int(r.nih_patient_id), str(r.set), str(r.arm))
    both = [i for i in candidates if info[p_ref].get(i, (False, False))[0] and info[p_pen].get(i, (False, False))[0]]
    dropped = {
        f"missing_{p_ref}": sum(1 for i in candidates if i not in info[p_ref]),
        f"missing_{p_pen}": sum(1 for i in candidates if i not in info[p_pen]),
        f"invalid_{p_ref}": sum(1 for i in candidates if i in info[p_ref] and not info[p_ref][i][0]),
        f"invalid_{p_pen}": sum(1 for i in candidates if i in info[p_pen] and not info[p_pen][i][0]),
    }
    base.update(n_candidates=len(candidates), n_dropped=len(candidates) - len(both), dropped=dropped)
    if not both:
        return _untested(base, "no B or U item is valid under both prompts")
    a1 = np.array([info[p_ref][i][1] for i in both], dtype=bool)
    a2 = np.array([info[p_pen][i][1] for i in both], dtype=bool)
    b = int(np.count_nonzero(a2 & ~a1))
    c = int(np.count_nonzero(~a2 & a1))
    n = len(both)
    frame = pd.DataFrame({"cl": [meta[i][0] for i in both], "st": [meta[i][1] for i in both],
                          "v": a2.astype(np.float64) - a1.astype(np.float64)})
    ci = M.clustered_estimate(frame, M.mean_stat, n_boot, seed, alpha, strata=True)
    out: dict[str, Any] = {
        **base, "estimate": (b - c) / n, "lo": ci["lo"], "hi": ci["hi"], "method": ci["method"], "b": b, "c": c, "n": n,
        "n_clusters": ci["n_clusters"],
        "abstention_ref": M.rate(int(a1.sum()), n, alpha), "abstention_pen": M.rate(int(a2.sum()), n, alpha),
        "mcnemar_p": stats.mcnemar_exact(b, c),
    }
    if clustered:
        rng = np.random.default_rng(seed)
        by_patient: dict[int, list[str]] = {}
        for i in sorted(both, key=lambda k: (meta[k][0], k)):
            if meta[i][2] == "B":
                by_patient.setdefault(meta[i][0], []).append(i)
        keep = {i for i in both if meta[i][2] == "U"}
        for pid in sorted(by_patient):
            ids = by_patient[pid]
            keep.add(ids[int(rng.integers(0, len(ids)))] if len(ids) > 1 else ids[0])
        mask = np.array([i in keep for i in both], dtype=bool)
        bb = int(np.count_nonzero(a2 & ~a1 & mask))
        cc = int(np.count_nonzero(~a2 & a1 & mask))
        out["mcnemar_check"] = {"b": bb, "c": cc, "n": int(mask.sum()), "p_value": stats.mcnemar_exact(bb, cc),
                                "rule": "one random B per patient plus every U item"}
        out["p_method"] = ("clustered BCa CI inversion (a patient contributes two pairs)" if max_pairs >= 2
                           else "clustered BCa CI inversion (two_per_patient rung)")
        out["ci_role"] = "primary"
        p, flag = ci.get("p_value", _NAN), ci.get("p_flag", "na")
    else:
        out["p_method"] = "exact McNemar"
        out["ci_role"] = "descriptive"
        out["mcnemar_check"] = None
        p, flag = out["mcnemar_p"], ""
    tested = n >= C.MIN_CELL_N and math.isfinite(p)
    out.update(p_value=p if tested else _NAN, p_flag=flag if tested else "na", p_value_descriptive=p, tested=bool(tested),
               reason="" if tested else f"fewer than {C.MIN_CELL_N} items valid under both prompts (interval only, no test)")
    return out


def max_pairs_per_patient(df: pd.DataFrame) -> int:
    """Largest number of distinct A/B pair ids of one NIH patient in ``df`` (0 without pairs)."""
    if not {"arm", "pair_id", "nih_patient_id"} <= set(df.columns):
        return 0
    pairs = df.loc[df["arm"].astype(str).isin(["A", "B"]), ["nih_patient_id", "pair_id"]]
    pairs = pairs.loc[pairs["pair_id"].astype(str) != ""]
    if pairs.empty:
        return 0
    return int(pairs.drop_duplicates().groupby("nih_patient_id")["pair_id"].nunique().max())


def h3_secondary(summaries: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """H3 secondary results of one model: ER_0.9 by prompt beside P0 accuracy, and FAR_B by prompt.

    ``summaries`` maps prompt id -> ``metrics.summarise`` result (primary rows).
    """
    out: dict[str, Any] = {"ER_0.9": {}, "accuracy": {}, "FAR_B": {}, "abstention": {}}
    for pid in M.ordered_prompts(summaries):
        s = summaries[pid]
        for key in out:
            if key in s:
                out[key][pid] = s[key]
    out["P0_accuracy"] = summaries["P0"]["accuracy"] if "P0" in summaries else None
    return out


# ----------------------------------------------------------------------------- H5
def h5(arm1_result: Mapping[str, Any] | None, model_key: str | None = None) -> dict[str, Any]:
    """H5 from an ``arm1.run_arm1`` result: the one-sided shuffled-expert permutation p (primary).

    estimate = BA(fused, beta = 1) - BA(beta = 0) on A and B; p = (1 + #{null gains >= observed})
    / (1 + n_perm); the patient-clustered BCa interval of the gain is descriptive. No test when
    Arm 1 was skipped or when A or B has fewer than ``constants.MIN_CELL_N`` items.
    """
    key = model_key if model_key is not None else (arm1_result or {}).get("model_key")
    base: dict[str, Any] = {"hypothesis": "H5", "family": "H5", "model_key": key, "sided": "one-sided (greater)",
                            "p_method": "shuffled-expert permutation", "ci_role": "descriptive"}
    if arm1_result is None:
        return _untested(base, "Arm 1 was not run")
    if arm1_result.get("skipped_reason"):
        return _untested(base, str(arm1_result["skipped_reason"]))
    gain = arm1_result["gain"]
    perm = arm1_result["shuffled_expert"]
    n_a, n_b = int(arm1_result["n_A"]), int(arm1_result["n_B"])
    tested = min(n_a, n_b) >= C.MIN_CELL_N and math.isfinite(perm["p_value"])
    return {
        **base, "estimate": float(perm["observed_gain"]), "lo": gain["lo"], "hi": gain["hi"], "method": gain["method"],
        "p_value": float(perm["p_value"]) if tested else _NAN, "p_flag": "" if tested else "na",
        "p_value_descriptive": float(perm["p_value"]), "tested": bool(tested),
        "reason": "" if tested else f"A or B has fewer than {C.MIN_CELL_N} items (interval only, no test)",
        "n": n_a + n_b, "n_A": n_a, "n_B": n_b, "n_perm": int(perm["n_perm"]), "null_mean": perm["null_mean"],
        "null_q025": perm["null_q025"], "null_q975": perm["null_q975"], "ba_fused": perm["ba_fused"],
        "ba_beta0": perm["ba_beta0"], "beta": perm["beta"], "gamma": perm["gamma"],
    }


# ----------------------------------------------------------------------------- Holm
def holm_family(results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Copies of a family's per-model results with ``p_holm`` (Holm step-down) and ``family_m``.

    Untested results (NaN p) keep NaN and are excluded from m.
    """
    out = [dict(r) for r in results]
    pvals = [r.get("p_value", _NAN) if r.get("tested", True) else _NAN for r in out]
    adj = stats.holm(pvals)
    m = sum(1 for p in pvals if p is not None and not (isinstance(p, float) and math.isnan(p)))
    for r, a in zip(out, adj):
        r["p_holm"] = float(a)
        r["family_m"] = int(m)
    return out


# ----------------------------------------------------------------------------- PLUS H2
def _ba_pair_stat(d: pd.DataFrame) -> float:
    a = d["a"].to_numpy(dtype=bool)
    na = int(a.sum())
    nb = a.size - na
    if na == 0 or nb == 0:
        return _NAN

    def ba(yes: str, no: str) -> float:
        return 0.5 * (d[yes].to_numpy(dtype=bool)[a].mean() + d[no].to_numpy(dtype=bool)[~a].mean())

    return float(ba("yes_a", "no_a") - ba("yes_b", "no_b"))


def h2(
    df: pd.DataFrame,
    model_a: str,
    model_b: str,
    n_boot: int = C.N_BOOT,
    seed: int | None = None,
    *,
    prompt_id: str = "P0",
    exclusions: pd.DataFrame | Iterable[str] | None = None,
    alpha: float = C.ALPHA,
) -> dict[str, Any]:
    """PLUS H2: ``model_a`` minus ``model_b`` (CheXOne minus Qwen2.5-VL-3B), paired, two-sided.

    Primary: the hallucination-free rate 1 - FAR on B and U (share of valid answers that are not
    "yes") over items valid under both models; ``ba_diff``: balanced accuracy on A and B over items
    valid under both. Both use patient-clustered BCa intervals with CI-inversion p.
    """
    seed = _seed(seed)
    base: dict[str, Any] = {"hypothesis": "H2", "family": "H2", "model_a": model_a, "model_b": model_b,
                            "prompt_id": prompt_id, "sided": "two-sided", "n_boot": int(n_boot), "seed": seed}

    def answers(model: str) -> pd.DataFrame:
        s = M.rows_for(df, model, prompt_id, M.TEST_ARMS, exclusions, condition="image")
        s = s.loc[s["valid"].astype(bool)]
        return s.loc[:, ["item_id", "arm", "set", "nih_patient_id", "parsed"]].set_index("item_id")

    ra, rb = answers(model_a), answers(model_b)
    common = ra.join(rb[["parsed"]].rename(columns={"parsed": "parsed_b"}), how="inner")
    if common.empty:
        return _untested(base, "no item is valid under both models")
    arm = common["arm"].astype(str).to_numpy()
    pa = common["parsed"].astype(str).to_numpy()
    pb = common["parsed_b"].astype(str).to_numpy()
    cl = common["nih_patient_id"].to_numpy(dtype=np.int64)
    st = common["set"].astype(str).to_numpy()
    neg = (arm == "B") | (arm == "U")
    hf = pd.DataFrame({"cl": cl[neg], "st": st[neg], "v": (pa[neg] != "yes").astype(float) - (pb[neg] != "yes").astype(float)})
    res = M.clustered_estimate(hf, M.mean_stat, n_boot, seed, alpha, strata=True) if len(hf) else {
        "estimate": _NAN, "lo": _NAN, "hi": _NAN, "method": "none", "p_value": _NAN, "p_flag": "na", "n_clusters": 0}
    ab = (arm == "A") | (arm == "B")
    baf = pd.DataFrame({"cl": cl[ab], "a": arm[ab] == "A", "yes_a": pa[ab] == "yes", "no_a": pa[ab] == "no",
                        "yes_b": pb[ab] == "yes", "no_b": pb[ab] == "no"})
    ba = M.clustered_estimate(baf, _ba_pair_stat, n_boot, seed, alpha, strata=False) if len(baf) else {
        "estimate": _NAN, "lo": _NAN, "hi": _NAN, "method": "none", "p_value": _NAN, "p_flag": "na", "n_clusters": 0}
    n = int(len(hf))
    tested = n >= C.MIN_CELL_N and math.isfinite(res.get("p_value", _NAN))
    return {
        **base, "quantity": "hallucination-free rate (1 - FAR) on B and U", "estimate": res["estimate"], "lo": res["lo"],
        "hi": res["hi"], "method": res["method"], "p_value": res.get("p_value", _NAN) if tested else _NAN,
        "p_flag": res.get("p_flag", "na") if tested else "na", "tested": bool(tested),
        "reason": "" if tested else f"fewer than {C.MIN_CELL_N} paired items (interval only, no test)",
        "n": n, "n_clusters": res["n_clusters"],
        "hfr_a": M.rate(int(np.count_nonzero(pa[neg] != "yes")), n, alpha),
        "hfr_b": M.rate(int(np.count_nonzero(pb[neg] != "yes")), n, alpha),
        "ba_diff": {**ba, "n": int(len(baf))},
    }
