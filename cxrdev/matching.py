"""RANZCR-to-NIH image matching by image content (kernel-side).

The only linkage in the study is between two public de-identified releases of the same radiographs,
matched by image content; no attempt is made to identify any person (PROTOCOL sections 1 and 9).

Rule (PROTOCOL section 1, PIPELINE_SPEC section 3):

1. Candidates: the NIH images whose original size ``(orig_w, orig_h)`` equals the JPEG size
   ``(jpg_w, jpg_h)`` when the dimension key is on and that bucket is non-empty (``key_used`` "dim");
   otherwise every NIH image (``key_used`` "global").
2. Distances: Hamming distance between 256-bit perceptual hashes; d1 / d2 = best / second best.
3. Cut: ``MatchConfig.cut`` or, when None, ``auto_cut`` (Otsu valley of the best-distance histogram,
   capped at a quarter of the bits, 64 of 256).
4. Margin route: accept the best candidate when d1 <= cut and d2 - d1 >= margin_bits (a single
   candidate has no runner-up and passes the margin test) and, when an NCC function is given,
   its 128-px NCC is >= ncc_min.
5. NCC tie-break: otherwise, when an NCC function is given and d1 <= cut, score the top-k candidates
   (k = max(2, topk_ncc), by Hamming distance) by NCC and accept the best when ncc_best >= ncc_min and
   ncc_best - ncc_second >= ncc_margin.
6. Otherwise "ambiguous" (d1 <= cut) or "unmatched" (d1 > cut, or no candidate).
7. One-to-one: when several UIDs claim one NIH file, the smallest d1 (then the highest ncc1) keeps it
   and the others become "duplicate_conflict" (an exact tie leaves no winner: all become conflicts).
8. Patient consistency: matched rows that disagree with their RANZCR patient's majority NIH patient
   become "patient_conflict" (a tie for the majority flags every matched row of that patient).

Recorded distances: ``d1`` is the Hamming distance of the selected candidate (for unmatched and
ambiguous rows, of the closest candidate) and ``d2`` the smallest distance among the other
candidates (-1 if there is none). In the margin route the selected candidate is the closest one,
so d1 / d2 are the best and second-best distances; in the NCC tie-break the NCC winner may be the
second-closest candidate, in which case d2 < d1. ``ncc1`` is the NCC of the selected (or best-NCC)
candidate and ``ncc2`` the runner-up NCC of the tie-break (NaN when not computed).

Runs inside the private Kaggle kernels and locally (numpy, pandas; Python 3.10 syntax). Private
helpers carry the ``_mt_`` prefix.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Callable, Mapping

import numpy as np
import pandas as pd

from cxrdev.imghash import hamming, packed_from_hex

MATCH_COLUMNS = (
    "uid", "ranzcr_patient_id", "jpg_w", "jpg_h", "nih_filename", "nih_patient_id", "d1", "d2",
    "ncc1", "ncc2", "n_candidates", "key_used", "route", "status",
)
MATCH_STATUSES = ("matched", "ambiguous", "unmatched", "duplicate_conflict", "patient_conflict")
MATCH_ROUTES = ("margin", "ncc_tiebreak", "none")
MATCH_KEYS = ("dim", "global")


@dataclass
class MatchConfig:
    """Matching parameters (defaults as registered in PIPELINE_SPEC section 3)."""

    margin_bits: int = 16
    cut: int | None = None
    ncc_min: float = 0.95
    ncc_margin: float = 0.02
    topk_ncc: int = 5
    use_dimension_key: bool = True


# --------------------------------------------------------------------------- helpers
def _mt_require(df: pd.DataFrame, columns, name: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{name} lacks columns {missing}")


def _mt_patient_from_name(name: str) -> int:
    head = str(name).split("_", 1)[0]
    return int(head) if head.isdigit() and len(head) == 8 else -1


def _mt_jsonable(obj):
    """Convert numpy scalars/arrays and NaN to plain JSON-compatible Python objects."""
    if isinstance(obj, dict):
        return {str(k): _mt_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_mt_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_mt_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if not math.isfinite(f) else f
    return obj


def _mt_nih_table(nih_hashes: pd.DataFrame, nih_meta: pd.DataFrame) -> pd.DataFrame:
    """NIH hashes joined to metadata, deduplicated by filename and sorted by filename."""
    h = pd.DataFrame({
        "nih_filename": nih_hashes["nih_filename"].astype(str).str.strip().to_numpy(),
        "ph": nih_hashes["ph"].astype(str).str.strip().str.lower().to_numpy(),
    })
    h = h[(h["ph"] != "") & (h["ph"] != "nan") & (h["nih_filename"] != "")]
    h = h.drop_duplicates("nih_filename", keep="first").sort_values("nih_filename", kind="stable")
    h = h.reset_index(drop=True)
    if nih_meta is not None and len(nih_meta):
        _mt_require(nih_meta, ["nih_filename", "nih_patient_id", "orig_w", "orig_h"], "nih_meta")
        meta = pd.DataFrame({
            "nih_filename": nih_meta["nih_filename"].astype(str).str.strip().to_numpy(),
            "nih_patient_id": pd.to_numeric(nih_meta["nih_patient_id"], errors="coerce").to_numpy(),
            "orig_w": pd.to_numeric(nih_meta["orig_w"], errors="coerce").to_numpy(),
            "orig_h": pd.to_numeric(nih_meta["orig_h"], errors="coerce").to_numpy(),
        }).drop_duplicates("nih_filename", keep="first")
        t = h.merge(meta, on="nih_filename", how="left")
    else:
        t = h.assign(nih_patient_id=np.nan, orig_w=np.nan, orig_h=np.nan)
    from_name = np.array([_mt_patient_from_name(n) for n in t["nih_filename"]], dtype=np.int64)
    pid = pd.to_numeric(t["nih_patient_id"], errors="coerce").to_numpy(dtype=np.float64)
    t["nih_patient_id"] = np.where(np.isfinite(pid), np.nan_to_num(pid, nan=-1.0), from_name).astype(np.int64)
    for c in ("orig_w", "orig_h"):
        v = pd.to_numeric(t[c], errors="coerce").to_numpy(dtype=np.float64)
        t[c] = np.where(np.isfinite(v), np.nan_to_num(v, nan=-1.0), -1.0).astype(np.int64)
    return t.reset_index(drop=True)


def candidate_index(nih_meta: pd.DataFrame) -> dict[tuple[int, int], np.ndarray]:
    """Map original size (orig_w, orig_h) -> positional row indices of ``nih_meta`` (sorted).

    Rows with a missing or non-positive size are left out (they remain global candidates).
    """
    _mt_require(nih_meta, ["orig_w", "orig_h"], "nih_meta")
    w = pd.to_numeric(nih_meta["orig_w"], errors="coerce").to_numpy(dtype=np.float64)
    h = pd.to_numeric(nih_meta["orig_h"], errors="coerce").to_numpy(dtype=np.float64)
    ok = np.isfinite(w) & np.isfinite(h) & (w > 0) & (h > 0)
    pos = np.flatnonzero(ok)
    groups: dict[tuple[int, int], list[int]] = {}
    for i, ww, hh in zip(pos.tolist(), w[ok].astype(np.int64).tolist(), h[ok].astype(np.int64).tolist()):
        groups.setdefault((int(ww), int(hh)), []).append(i)
    return {k: np.asarray(v, dtype=np.int64) for k, v in sorted(groups.items())}


def _mt_otsu(hist: np.ndarray) -> int | None:
    """Otsu threshold t on an integer histogram (class 0 = values <= t); None if undefined."""
    counts = np.asarray(hist, dtype=np.int64)
    levels = np.arange(counts.size, dtype=np.int64)
    c0 = np.cumsum(counts)
    total = int(c0[-1])
    s0 = np.cumsum(counts * levels)
    s_total = int(s0[-1])
    c1 = total - c0
    valid = (c0 > 0) & (c1 > 0)
    if not valid.any():
        return None
    mu0 = np.where(valid, s0 / np.maximum(c0, 1), 0.0)
    mu1 = np.where(valid, (s_total - s0) / np.maximum(c1, 1), 0.0)
    between = np.where(valid, c0.astype(np.float64) * c1.astype(np.float64) * (mu1 - mu0) ** 2, -1.0)
    return int(np.argmax(between))


def auto_cut(best_distances: np.ndarray, n_bits: int = 256) -> int:
    """Absolute Hamming cut from the empirical best-distance distribution (Otsu valley, capped).

    1. Otsu's threshold splits the histogram of best distances into a low (match) and a high (null)
       class; 2. within the open interval between the two class means, a position is a valley point
       when the 5-bin moving average there is at most half of the highest smoothed count on each
       side of it (a real dip, not a shoulder); 3. the cut is the middle of the longest run of the
       deepest valley points; 4. the cut never exceeds ``n_bits // 4`` (64 of 256 bits), which is
       also the result when the distribution has no valley (for example when every image matches).
    """
    cap = int(n_bits) // 4
    d = np.asarray(best_distances, dtype=np.float64).reshape(-1)
    d = d[np.isfinite(d) & (d >= 0)]
    if d.size < 2:
        return cap
    d = np.clip(np.round(d), 0, n_bits).astype(np.int64)
    if d.min() == d.max():
        return cap
    hist = np.bincount(d, minlength=n_bits + 1)[: n_bits + 1]
    t = _mt_otsu(hist)
    if t is None:
        return cap
    levels = np.arange(hist.size, dtype=np.float64)
    w0, w1 = hist[: t + 1].sum(), hist[t + 1:].sum()
    if w0 == 0 or w1 == 0:
        return cap
    mu0 = float((levels[: t + 1] * hist[: t + 1]).sum() / w0)
    mu1 = float((levels[t + 1:] * hist[t + 1:]).sum() / w1)
    smooth = np.convolve(hist.astype(np.float64), np.ones(5) / 5.0, mode="same")
    lo, hi = int(math.floor(mu0)) + 1, int(math.ceil(mu1)) - 1
    valley = []
    for v in range(max(lo, 0), min(hi, n_bits) + 1):
        left_peak = float(smooth[: v + 1].max())
        right_peak = float(smooth[v:].max())
        if smooth[v] <= 0.5 * min(left_peak, right_peak) and min(left_peak, right_peak) > 0:
            valley.append(v)
    if not valley:
        return cap
    floor_value = min(float(smooth[v]) for v in valley)
    floor = [v for v in valley if smooth[v] <= floor_value + 1e-12]
    runs: list[list[int]] = [[floor[0]]]
    for v in floor[1:]:
        if v == runs[-1][-1] + 1:
            runs[-1].append(v)
        else:
            runs.append([v])
    longest = max(runs, key=len)
    return int(min(longest[(len(longest) - 1) // 2], cap))


def _mt_rank(q: np.ndarray, db: np.ndarray, keys: list[tuple[int, int]], index: dict, use_key: bool,
             k: int, block: int = 512):
    """Top-k candidates per query: (idx [n x k], dist [n x k], n_candidates [n], key_used [n])."""
    n, n_db = q.shape[0], db.shape[0]
    top_idx = np.full((n, k), -1, dtype=np.int64)
    top_d = np.full((n, k), -1, dtype=np.int64)
    n_cand = np.zeros(n, dtype=np.int64)
    key_used = np.array(["global"] * n, dtype=object)
    groups: dict[object, list[int]] = {}
    for i, key in enumerate(keys):
        bucket = index.get(key) if use_key else None
        if bucket is not None and len(bucket) > 0:
            groups.setdefault(key, []).append(i)
            key_used[i] = "dim"
        else:
            groups.setdefault("__global__", []).append(i)
    all_idx = np.arange(n_db, dtype=np.int64)
    for gkey in sorted(groups, key=lambda g: (g == "__global__", str(g))):
        rows = np.asarray(groups[gkey], dtype=np.int64)
        cand = all_idx if gkey == "__global__" else index[gkey]
        m = int(len(cand))
        if m == 0:
            continue
        kk = min(k, m)
        dbc = db[cand]
        step = max(1, min(int(block), (1 << 24) // m))
        for b0 in range(0, len(rows), step):
            r = rows[b0:b0 + step]
            dist = hamming(q[r], dbc).astype(np.int64)
            composite = dist * (n_db + 1) + cand[None, :]
            if m > kk:
                part = np.argpartition(composite, kk - 1, axis=1)[:, :kk]
            else:
                part = np.tile(np.arange(m), (len(r), 1))
            pc = np.take_along_axis(composite, part, axis=1)
            order = np.argsort(pc, axis=1, kind="stable")
            sel = np.take_along_axis(part, order, axis=1)
            top_idx[r, :kk] = cand[sel]
            top_d[r, :kk] = np.take_along_axis(dist, sel, axis=1)
            n_cand[r] = m
    return top_idx, top_d, n_cand, key_used


# --------------------------------------------------------------------------- main entry point
def match_all(ranzcr_hashes: pd.DataFrame, nih_hashes: pd.DataFrame, nih_meta: pd.DataFrame,
              config: MatchConfig | None = None,
              ncc_fn: Callable[[str, str], float] | None = None) -> tuple[pd.DataFrame, dict]:
    """Match every RANZCR image to at most one NIH image; returns (S3 match table, stats dict).

    ``ranzcr_hashes``: uid, ranzcr_patient_id, jpg_w, jpg_h, ph (hex). ``nih_hashes``: nih_filename,
    ph (hex). ``nih_meta``: S2 (orig_w, orig_h, nih_patient_id are used). ``ncc_fn(uid, nih_filename)``
    returns the 128-px NCC of the pair (NaN when an image cannot be read). Row order follows
    ``ranzcr_hashes``. See the module docstring for the rule and the recorded columns.
    """
    cfg = config if config is not None else MatchConfig()
    if cfg.topk_ncc < 1:
        raise ValueError("topk_ncc must be at least 1")
    _mt_require(ranzcr_hashes, ["uid", "ranzcr_patient_id", "jpg_w", "jpg_h", "ph"], "ranzcr_hashes")
    _mt_require(nih_hashes, ["nih_filename", "ph"], "nih_hashes")
    uids = ranzcr_hashes["uid"].astype(str).str.strip().tolist()
    if len(set(uids)) != len(uids):
        raise ValueError("ranzcr_hashes has duplicated uid values")
    patients = ranzcr_hashes["ranzcr_patient_id"].astype(str).str.strip().tolist()
    jw = pd.to_numeric(ranzcr_hashes["jpg_w"], errors="coerce").fillna(-1).astype(np.int64).tolist()
    jh = pd.to_numeric(ranzcr_hashes["jpg_h"], errors="coerce").fillna(-1).astype(np.int64).tolist()
    r_hex = ranzcr_hashes["ph"].astype(str).str.strip().str.lower().tolist()
    nih = _mt_nih_table(nih_hashes, nih_meta)
    n_bits = 4 * len(r_hex[0]) if r_hex else (4 * len(nih["ph"].iloc[0]) if len(nih) else 256)
    if len(nih) and 4 * len(nih["ph"].iloc[0]) != n_bits:
        raise ValueError("RANZCR and NIH hashes have different lengths")
    q = packed_from_hex(r_hex) if r_hex else np.zeros((0, n_bits // 8), dtype=np.uint8)
    db = packed_from_hex(nih["ph"].tolist()) if len(nih) else np.zeros((0, n_bits // 8), dtype=np.uint8)
    index = candidate_index(nih) if len(nih) else {}
    k = max(2, int(cfg.topk_ncc))
    keys = list(zip(jw, jh))
    top_idx, top_d, n_cand, key_used = _mt_rank(q, db, keys, index, bool(cfg.use_dimension_key), k)

    best = top_d[:, 0] if len(top_d) else np.zeros(0, dtype=np.int64)
    best = best[n_cand > 0]
    cut_auto = cfg.cut is None
    cut = auto_cut(best, n_bits=n_bits) if cut_auto else int(cfg.cut)

    files = nih["nih_filename"].tolist()
    pids = nih["nih_patient_id"].to_numpy(dtype=np.int64)
    rec = {c: [] for c in MATCH_COLUMNS}
    top2_same: dict[str, bool] = {}
    n_ncc_calls = 0
    for i, uid in enumerate(uids):
        n = int(n_cand[i])
        kk = min(k, n)
        cidx = top_idx[i, :kk]
        cd = top_d[i, :kk]
        d1 = int(cd[0]) if n >= 1 else -1
        d2 = int(cd[1]) if n >= 2 else -1
        if n >= 2:
            top2_same[uid] = bool(pids[cidx[0]] >= 0 and pids[cidx[0]] == pids[cidx[1]])
        status, route, sel = "unmatched", "none", -1
        ncc1 = ncc2 = float("nan")
        rec_d1, rec_d2 = d1, d2
        if n >= 1 and d1 <= cut:
            margin_ok = n == 1 or (d2 - d1) >= int(cfg.margin_bits)
            scores: dict[int, float] = {}
            if margin_ok:
                if ncc_fn is None:
                    status, route, sel = "matched", "margin", 0
                else:
                    s0 = float(ncc_fn(uid, files[cidx[0]]))
                    n_ncc_calls += 1
                    scores[0] = s0
                    ncc1 = s0
                    if math.isfinite(s0) and s0 >= float(cfg.ncc_min):
                        status, route, sel = "matched", "margin", 0
            if status != "matched" and ncc_fn is not None:
                n_top = min(k, n)  # top-k by Hamming distance, never fewer than two when two exist
                for j in range(n_top):
                    if j not in scores:
                        scores[j] = float(ncc_fn(uid, files[cidx[j]]))
                        n_ncc_calls += 1
                vals = np.array([scores[j] if math.isfinite(scores[j]) else -np.inf for j in range(n_top)])
                order = np.argsort(-vals, kind="stable")
                b = int(order[0])
                best_ncc = float(vals[b])
                second = float(vals[order[1]]) if n_top > 1 else -np.inf
                ncc1 = best_ncc if math.isfinite(best_ncc) else float("nan")
                ncc2 = second if math.isfinite(second) else float("nan")
                if best_ncc >= float(cfg.ncc_min) and best_ncc - second >= float(cfg.ncc_margin):
                    status, route, sel = "matched", "ncc_tiebreak", b
            if status != "matched":
                status = "ambiguous"
            elif sel > 0:
                rec_d1 = int(cd[sel])
                rec_d2 = int(cd[0])
        rec["uid"].append(uid)
        rec["ranzcr_patient_id"].append(patients[i])
        rec["jpg_w"].append(int(jw[i]))
        rec["jpg_h"].append(int(jh[i]))
        rec["nih_filename"].append(files[cidx[sel]] if status == "matched" else "")
        rec["nih_patient_id"].append(int(pids[cidx[sel]]) if status == "matched" else -1)
        rec["d1"].append(rec_d1)
        rec["d2"].append(rec_d2)
        rec["ncc1"].append(ncc1)
        rec["ncc2"].append(ncc2)
        rec["n_candidates"].append(n)
        rec["key_used"].append(str(key_used[i]))
        rec["route"].append(route)
        rec["status"].append(status)

    table = pd.DataFrame(rec, columns=list(MATCH_COLUMNS))
    for c in ("jpg_w", "jpg_h", "nih_patient_id", "d1", "d2", "n_candidates"):
        table[c] = table[c].astype("int64")
    for c in ("ncc1", "ncc2"):
        table[c] = table[c].astype("float64")
    table = patient_consistency(resolve_one_to_one(table))
    stats = calibration_report(table, cut=cut, n_bits=n_bits, top2_same_patient=top2_same)
    stats.update(_mt_jsonable({
        "cut_rule": "auto_otsu_valley_capped" if cut_auto else "fixed",
        "config": asdict(cfg),
        "n_nih_hashes": len(nih),
        "n_nih_without_meta_size": int((nih["orig_w"] < 0).sum()) if len(nih) else 0,
        "n_dimension_buckets": len(index),
        "n_ncc_calls": n_ncc_calls,
    }))
    return table, stats


def resolve_one_to_one(table: pd.DataFrame) -> pd.DataFrame:
    """Enforce one RANZCR image per NIH file among matched rows (see rule 7 in the module docstring)."""
    t = table.copy()
    matched = t["status"] == "matched"
    if not matched.any():
        return t
    claims = t.loc[matched, ["nih_filename", "d1", "ncc1"]]
    counts = claims["nih_filename"].value_counts()
    dup_files = set(counts[counts > 1].index.tolist())
    if not dup_files:
        return t
    losers: list = []
    sub = claims[claims["nih_filename"].isin(dup_files)]
    for _, rows in sub.groupby("nih_filename", sort=True):
        d1 = rows["d1"].astype("int64")
        ncc = rows["ncc1"].astype("float64").fillna(-np.inf)
        closest = rows.index[(d1 == d1.min()).to_numpy()]
        top = ncc.loc[closest]
        winners = closest[(top == top.max()).to_numpy()]
        keep = set(winners.tolist()) if len(winners) == 1 else set()
        losers.extend(ix for ix in rows.index.tolist() if ix not in keep)
    if losers:
        t.loc[losers, "status"] = "duplicate_conflict"
    return t


def patient_consistency(table: pd.DataFrame) -> pd.DataFrame:
    """Flag matched rows that disagree with the patient's majority NIH patient (rule 8)."""
    t = table.copy()
    matched = t["status"] == "matched"
    if not matched.any():
        return t
    sub = t.loc[matched, ["ranzcr_patient_id", "nih_patient_id"]]
    n_nih = sub.groupby("ranzcr_patient_id")["nih_patient_id"].nunique()
    bad_patients = set(n_nih[n_nih > 1].index.tolist())
    if not bad_patients:
        return t
    flagged: list = []
    for _, rows in sub[sub["ranzcr_patient_id"].isin(bad_patients)].groupby("ranzcr_patient_id", sort=True):
        vc = rows["nih_patient_id"].value_counts()
        majority = vc[vc == vc.max()].index.tolist()
        if len(majority) == 1:
            flagged.extend(rows.index[(rows["nih_patient_id"] != majority[0]).to_numpy()].tolist())
        else:
            flagged.extend(rows.index.tolist())
    t.loc[flagged, "status"] = "patient_conflict"
    return t


def _mt_quantiles(values: np.ndarray) -> dict:
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0}
    q = np.quantile(v, [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0])
    return {"n": int(v.size), "min": q[0], "p05": q[1], "p25": q[2], "median": q[3], "p75": q[4],
            "p95": q[5], "max": q[6], "mean": float(v.mean())}


def calibration_report(table: pd.DataFrame, cut: int | None = None, n_bits: int = 256,
                       top2_same_patient: Mapping[str, bool] | None = None) -> dict:
    """Calibration and yield summary of an S3 table (JSON-serialisable dict).

    Keys: n_ranzcr, n_ranzcr_patients, status_counts, route_counts, key_used_counts, match_rate,
    matched_patients, ambiguous_count, ambiguous_patients, ambiguous_share_within_patient (share of
    ambiguous UIDs whose two closest NIH candidates are films of one NIH patient; needs
    ``top2_same_patient`` from ``match_all``, else None), ambiguous_uids_in_matched_patients_share
    (share of ambiguous UIDs whose RANZCR patient also has a matched UID), cut, n_bits, d1_hist
    (counts per distance 0..n_bits), margin_hist ({"lo", "counts"} of d2 - d1 per integer value),
    d1 quantiles per status, ncc1 quantiles of matched rows, nih_patients_with_multiple_ranzcr_patients.
    """
    _mt_require(table, MATCH_COLUMNS, "match table")
    n = len(table)
    status = table["status"].astype(str)
    d1 = table["d1"].to_numpy(dtype=np.int64)
    d2 = table["d2"].to_numpy(dtype=np.int64)
    ok1 = d1 >= 0
    hist = np.bincount(np.clip(d1[ok1], 0, n_bits), minlength=n_bits + 1)[: n_bits + 1]
    both = ok1 & (d2 >= 0)
    margins = (d2 - d1)[both]
    if margins.size:
        lo = int(margins.min())
        margin_hist = {"lo": lo, "counts": np.bincount(margins - lo).tolist()}
    else:
        margin_hist = {"lo": 0, "counts": []}
    matched = (status == "matched").to_numpy()
    ambiguous = (status == "ambiguous").to_numpy()
    patients = table["ranzcr_patient_id"].astype(str).to_numpy()
    matched_patients = set(patients[matched].tolist())
    amb_uids = table.loc[ambiguous, "uid"].astype(str).tolist()
    if top2_same_patient is not None and amb_uids:
        flags = [bool(top2_same_patient[u]) for u in amb_uids if u in top2_same_patient]
        within = (sum(flags) / len(flags)) if flags else None
    else:
        within = None
    amb_in_matched = float(np.mean([p in matched_patients for p in patients[ambiguous]])) if ambiguous.any() else None
    m_rows = table.loc[matched, ["ranzcr_patient_id", "nih_patient_id"]]
    multi = int((m_rows.groupby("nih_patient_id")["ranzcr_patient_id"].nunique() > 1).sum()) if len(m_rows) else 0
    report = {
        "n_ranzcr": n,
        "n_ranzcr_patients": int(len(set(patients.tolist()))),
        "status_counts": {s: int((status == s).sum()) for s in MATCH_STATUSES},
        "route_counts": {r: int((table["route"].astype(str) == r).sum()) for r in MATCH_ROUTES},
        "key_used_counts": {k_: int((table["key_used"].astype(str) == k_).sum()) for k_ in MATCH_KEYS},
        "match_rate": (float(matched.sum()) / n) if n else None,
        "matched_patients": len(matched_patients),
        "ambiguous_count": int(ambiguous.sum()),
        "ambiguous_patients": int(len(set(patients[ambiguous].tolist()))),
        "ambiguous_share_within_patient": within,
        "ambiguous_uids_in_matched_patients_share": amb_in_matched,
        "cut": None if cut is None else int(cut),
        "n_bits": int(n_bits),
        "d1_hist": hist.tolist(),
        "margin_hist": margin_hist,
        "d1_by_status": {s: _mt_quantiles(d1[(status == s).to_numpy() & ok1]) for s in MATCH_STATUSES},
        "ncc1_matched": _mt_quantiles(table.loc[matched, "ncc1"].to_numpy(dtype=np.float64)),
        "nih_patients_with_multiple_ranzcr_patients": multi,
    }
    return _mt_jsonable(report)
