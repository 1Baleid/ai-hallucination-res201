"""Join the blind reading to the item list: exclusions (S12) and the verification summary.

The reading (``verified_items.csv``, S11) was made blind to arm and to model output under opaque
read ids; this script is the first place the read ids meet the arms (through ``reading_key.csv``).
Exclusion rule per read item (PROTOCOL section 2, v0.4), first match wins:
"skip" response -> skip; ambiguous flag -> flag_ambiguous; tracheostomy flag ->
flag_tracheostomy; out-of-field flag -> flag_out_of_field; only then, a present/absent response
that contradicts the RANZCR CLiP label (43 trained labellers) -> label_disagreement. Every skip and
flag exclusion is applied before the label check, so a flagged or skipped image is never also
counted as a disagreement (RANZCR CLiP labels tracheostomy tubes as "ETT normal", so an A image
whose only airway tube is a tracheostomy is read "absent" with the tracheostomy flag and is excluded
as flag_tracheostomy, not as a disagreement). The agreement with the labels, Cohen's kappa against
them and the disagreement list are computed over the images that reach the label check (answered,
not flagged); ``agreement_all_read`` keeps the agreement over every answered image for reference. A
disagreement is first treated as a possible mapping error: the disagreeing items are listed in a
local file for the patient re-match, and an item whose disagreement persists stays excluded as label
noise.

Retest clock (PROTOCOL section 2): the retest must start at least 48 h after the first read was
saved. The check uses ``saved_utc`` of ``verified_items.meta.json`` (the reading tool's sidecar) and
the earliest response timestamp of the retest (falling back to the retest meta's ``saved_utc``);
the result goes to ``retest.gap_hours_after_read`` / ``retest.gap_ok`` and a too-early or
uncheckable retest adds a line to ``warnings`` (a warning, not an error).

Outputs:
* ``--out`` exclusions.csv (S12: item_id, reason), local (``items/``);
* ``--disagreements`` (default ``<out dir>/verification_disagreements.csv``), local: the items
  whose reading contradicts the label, for the mapping re-check;
* ``--summary`` verification_summary.json: aggregates only (counts, agreement with the labels
  overall and per arm with Wilson intervals, exclusions by reason and arm, the retest Cohen's
  kappa, the retest clock check, ``warnings``, file hashes and the meta-file checks). No item id,
  filename or label of a single image. Undefined values (for example a kappa with no variation) are
  written as JSON null.

Usage:
  python scripts/verify_join.py --items items/items.csv --key items/reading_key.csv
      --verified items/verified_items.csv [--retest items/retest.csv]
      --out items/exclusions.csv --summary results/verification_summary.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from cxrdev import fileio, stats  # noqa: E402
from cxrdev.metrics import EXCLUSION_REASONS, rate  # noqa: E402

VERIFIED_COLUMNS: tuple[str, ...] = (
    "read_id", "nih_filename", "response", "flag_ambiguous", "flag_tracheostomy", "flag_out_of_field", "note",
    "seconds", "order_index", "timestamp_utc",
)
RESPONSES: tuple[str, ...] = ("present", "absent", "skip")
FLAG_REASONS: tuple[str, ...] = ("flag_ambiguous", "flag_tracheostomy", "flag_out_of_field")
DISAGREEMENT_COLUMNS: tuple[str, ...] = ("read_id", "item_id", "arm", "nih_filename", "response", "ranzcr_ett")
RETEST_MIN_HOURS = 48.0


def read_verified(path: str | Path) -> pd.DataFrame:
    """Read an S11 file (verified_items.csv or retest.csv) with text ids and 0/1 flags; validates responses."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    fileio.require_columns(df, VERIFIED_COLUMNS, str(path))
    bad = sorted(set(df["response"]) - set(RESPONSES))
    if bad:
        raise ValueError(f"{path}: unknown response value(s) {bad}")
    for c in FLAG_REASONS:
        vals = df[c].str.strip()
        if not vals.isin(["0", "1"]).all():
            raise ValueError(f"{path}: column {c} must be 0/1")
        df[c] = vals.astype(int)
    if bool(df["read_id"].duplicated().any()):
        raise ValueError(f"{path}: duplicated read_id")
    return df


def _parse_utc(text: str) -> datetime | None:
    try:
        t = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    except ValueError:
        return None
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def meta_check(path: str | Path) -> dict[str, Any]:
    """SHA-256 of an S11 file and the check against its sidecar ``<stem>.meta.json`` (or ``<name>.meta.json``).

    Returns ``sha256``, ``meta_found``, ``meta_sha256_ok`` (None without a meta file) and the meta
    fields ``saved_utc``, ``seed``, ``n``, ``mode``, ``tool_version`` when present.
    """
    p = Path(path)
    out: dict[str, Any] = {"sha256": fileio.sha256_file(p), "meta_found": False, "meta_sha256_ok": None}
    for cand in (p.with_suffix(".meta.json"), p.with_name(p.name + ".meta.json")):
        if cand.is_file():
            meta = json.loads(cand.read_text(encoding="utf-8"))
            out["meta_found"] = True
            out["meta_sha256_ok"] = bool(meta.get("sha256") == out["sha256"])
            for k in ("saved_utc", "seed", "n", "mode", "tool_version"):
                out[k] = meta.get(k)
            break
    return out


def _primary_reason(r: Mapping[str, Any], ett: int) -> str | None:
    """The exclusion reason of one read item: skip, then the flags, and only then the label check."""
    if r["response"] == "skip":
        return "skip"
    for flag in FLAG_REASONS:
        if int(r[flag]) == 1:
            return flag
    if (r["response"] == "present") != (int(ett) == 1):
        return "label_disagreement"
    return None


def _json_safe(obj: Any) -> Any:
    """Plain JSON with NaN and infinities replaced by None (written as null)."""
    if isinstance(obj, Mapping):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def join_verification(
    items: pd.DataFrame,
    key: pd.DataFrame,
    verified: pd.DataFrame,
    retest: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    """Exclusions (S12), the aggregate summary and the local disagreement list.

    Raises ValueError when a read id is not in the key, a keyed item is not in the item list, or a
    filename differs between the reading and the item list.
    """
    fileio.require_columns(items, ("item_id", "arm", "ett", "nih_filename"), "items")
    fileio.require_columns(key, ("read_id", "item_id"), "reading key")
    k = key.loc[:, ["read_id", "item_id"]].astype(str)
    it = items.loc[:, ["item_id", "arm", "ett", "nih_filename"]].copy()
    it["item_id"] = it["item_id"].astype(str)
    v = verified.copy()
    v["read_id"] = v["read_id"].astype(str)
    unknown = sorted(set(v["read_id"]) - set(k["read_id"]))
    if unknown:
        raise ValueError(f"{len(unknown)} read id(s) of the reading are not in the reading key")
    j = v.merge(k, on="read_id", how="left").merge(it, on="item_id", how="left", suffixes=("", "_item"))
    if bool(j["arm"].isna().any()):
        raise ValueError("a keyed item is not in the item list")
    if bool((j["nih_filename"].astype(str) != j["nih_filename_item"].astype(str)).any()):
        raise ValueError("an NIH filename of the reading differs from the item list (wrong key or list)")
    j["ett"] = j["ett"].astype(int)
    j["reason"] = [_primary_reason(r, e) for r, e in zip(j.to_dict("records"), j["ett"])]
    excl = j.loc[j["reason"].notna(), ["item_id", "reason"]].sort_values(["reason", "item_id"]).reset_index(drop=True)
    answered = j.loc[j["response"] != "skip"]
    flagged = answered[list(FLAG_REASONS)].sum(axis=1) > 0
    # the label check sees only answered, unflagged images (flags and skips are applied first)
    compared = answered.loc[~flagged]
    agree = (compared["response"] == "present") == (compared["ett"] == 1)
    disagreements = compared.loc[~agree].assign(ranzcr_ett=lambda d: d["ett"].astype(int)).loc[:, list(DISAGREEMENT_COLUMNS)]
    if int((~agree).sum()) != int((j["reason"] == "label_disagreement").sum()):  # pragma: no cover - invariant
        raise RuntimeError("internal error: disagreements and label_disagreement exclusions differ")
    agree_all = (answered["response"] == "present") == (answered["ett"] == 1)

    summary: dict[str, Any] = {
        "n_read": int(len(v)), "n_key": int(len(k)), "n_key_not_read": int(len(set(k["read_id"]) - set(v["read_id"]))),
        "responses": {r: int((v["response"] == r).sum()) for r in RESPONSES},
        "flags": {f: int(v[f].sum()) for f in FLAG_REASONS},
        "agreement": rate(int(agree.sum()), int(len(compared))),
        "agreement_basis": "answered images without a flag (skip and flag exclusions applied first)",
        "agreement_all_read": rate(int(agree_all.sum()), int(len(answered))),
        "n_disagreements": int((~agree).sum()),
        "n_flagged_answered_contrary_to_label": int((~agree_all & flagged).sum()),
        "by_arm": {},
        "exclusions": {
            "n_items": int(len(excl)),
            "by_reason": {r: int((excl["reason"] == r).sum()) for r in EXCLUSION_REASONS},
            "by_arm": {},
        },
        "blinding": "read under opaque ids in one random order, blind to arm and to model output; joined afterwards",
        "warnings": [],
    }
    summary["agreement_unflagged"] = summary["agreement"]  # kept for older readers of the summary
    a_lab = (compared["ett"] == 1).astype(int).tolist()
    a_read = (compared["response"] == "present").astype(int).tolist()
    summary["kappa_vs_ranzcr"] = float(stats.cohen_kappa(a_read, a_lab)) if len(compared) else float("nan")
    for arm in ("A", "B", "U"):
        ja = j.loc[j["arm"] == arm]
        ca = compared.loc[compared["arm"] == arm]
        ag = (ca["response"] == "present") == (ca["ett"] == 1)
        summary["by_arm"][arm] = {
            "n_read": int(len(ja)), **{r: int((ja["response"] == r).sum()) for r in RESPONSES},
            "flagged": int((ja[list(FLAG_REASONS)].sum(axis=1) > 0).sum()),
            "agreement": rate(int(ag.sum()), int(len(ca))), "n_disagreements": int((~ag).sum()),
            "by_reason": {r: int((ja["reason"] == r).sum()) for r in EXCLUSION_REASONS},
        }
        summary["exclusions"]["by_arm"][arm] = int(j.loc[(j["arm"] == arm) & j["reason"].notna()].shape[0])

    summary["retest"] = None
    if retest is not None:
        rt = retest.copy()
        rt["read_id"] = rt["read_id"].astype(str)
        both = v.merge(rt, on="read_id", suffixes=("_1", "_2"))
        usable = both.loc[(both["response_1"] != "skip") & (both["response_2"] != "skip")]
        r1 = usable["response_1"].tolist()
        r2 = usable["response_2"].tolist()
        first_ts = [ts for ts in (_parse_utc(x) for x in rt["timestamp_utc"]) if ts is not None]
        summary["retest"] = {
            "n": int(len(rt)), "n_matched": int(len(both)), "n_compared": int(len(usable)),
            "kappa": float(stats.cohen_kappa(r1, r2)) if r1 else float("nan"),
            "agreement": rate(sum(a == b for a, b in zip(r1, r2)), len(r1)),
            "first_response_utc": min(first_ts).strftime("%Y-%m-%dT%H:%M:%SZ") if first_ts else None,
        }
    return excl, summary, disagreements.reset_index(drop=True)


def run(
    items_path: str | Path,
    key_path: str | Path,
    verified_path: str | Path,
    out: str | Path,
    summary_path: str | Path,
    retest_path: str | Path | None = None,
    disagreements_path: str | Path | None = None,
) -> dict[str, Any]:
    """Read the inputs, write exclusions, the disagreement list and the summary; return the summary."""
    from cxrdev.items import read_items  # the S5 reader with its types

    items = read_items(items_path)
    key = pd.read_csv(key_path, dtype=str, keep_default_na=False)
    verified = read_verified(verified_path)
    retest = read_verified(retest_path) if retest_path else None
    excl, summary, dis = join_verification(items, key, verified, retest)
    summary["verified_file"] = meta_check(verified_path)
    warnings: list[str] = summary.setdefault("warnings", [])
    if not summary["verified_file"]["meta_found"]:
        warnings.append("no meta file next to the reading: its SHA-256 and save time could not be checked")
    elif not summary["verified_file"]["meta_sha256_ok"]:
        warnings.append("the reading's SHA-256 does not match its meta file")
    if retest_path:
        summary["retest_file"] = meta_check(retest_path)
        summary["retest"].update(retest_clock(summary["verified_file"], summary["retest_file"],
                                              summary["retest"].get("first_response_utc")))
        if summary["retest"]["gap_ok"] is False:
            warnings.append(f"the retest started {summary['retest']['gap_hours_after_read']:.2f} h after the first read "
                            f"was saved, earlier than the registered {RETEST_MIN_HOURS:.0f} h (PROTOCOL section 2)")
        elif summary["retest"]["gap_ok"] is None:
            warnings.append(f"the 48-h retest clock could not be checked: {summary['retest']['gap_source']}")
    out = Path(out)
    summary["exclusions_sha256"] = fileio.write_csv(excl, out)
    dpath = Path(disagreements_path) if disagreements_path else out.with_name("verification_disagreements.csv")
    fileio.write_csv(dis, dpath)
    sp = Path(summary_path)
    sp.parent.mkdir(parents=True, exist_ok=True)
    summary = _json_safe(summary)
    sp.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8",
                  newline="\n")
    return summary


def retest_clock(verified_meta: Mapping[str, Any], retest_meta: Mapping[str, Any] | None,
                 first_response_utc: str | None) -> dict[str, Any]:
    """The 48-h retest check: hours from the first read's save to the retest start, and whether >= 48.

    The first read's save time is ``saved_utc`` of its meta file (``verified_items.meta.json``); the
    retest start is its earliest response timestamp, else the retest meta's ``saved_utc`` (the end
    of the retest; used only when no response has a timestamp, and named in ``gap_source``).
    Returns ``read_saved_utc``, ``retest_start_utc``, ``gap_hours_after_read`` (None when either time
    is missing), ``gap_ok`` (True / False / None), ``gap_source`` and ``min_hours``.
    """
    saved_text = verified_meta.get("saved_utc") if verified_meta.get("meta_found") else None
    saved = _parse_utc(saved_text or "")
    start = _parse_utc(first_response_utc or "")
    source = "first retest response timestamp"
    if start is None and retest_meta and retest_meta.get("meta_found"):
        start = _parse_utc(retest_meta.get("saved_utc") or "")
        source = "retest meta saved_utc (end of the retest)"
    out: dict[str, Any] = {"min_hours": RETEST_MIN_HOURS,
                           "read_saved_utc": saved.strftime("%Y-%m-%dT%H:%M:%SZ") if saved else None,
                           "retest_start_utc": start.strftime("%Y-%m-%dT%H:%M:%SZ") if start else None}
    if saved is None:
        return {**out, "gap_hours_after_read": None, "gap_ok": None,
                "gap_source": "no saved_utc in the first read's meta file (verified_items.meta.json)"}
    if start is None:
        return {**out, "gap_hours_after_read": None, "gap_ok": None, "gap_source": "no retest start time"}
    gap = (start - saved).total_seconds() / 3600.0
    return {**out, "gap_hours_after_read": round(gap, 2), "gap_ok": bool(gap >= RETEST_MIN_HOURS),
            "gap_source": f"read meta saved_utc to {source}"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", default="items/items.csv")
    ap.add_argument("--key", default="items/reading_key.csv")
    ap.add_argument("--verified", default="items/verified_items.csv")
    ap.add_argument("--retest", default=None)
    ap.add_argument("--out", default="items/exclusions.csv")
    ap.add_argument("--summary", default="results/verification_summary.json")
    ap.add_argument("--disagreements", default=None, help="local list for the mapping re-check (default beside --out)")
    a = ap.parse_args(argv)
    s = run(a.items, a.key, a.verified, a.out, a.summary, a.retest, a.disagreements)
    ex = s["exclusions"]
    est = s["agreement"]["estimate"]
    print(f"read {s['n_read']} images; agreement with RANZCR CLiP labels "
          f"{'n/a' if est is None else format(est, '.3f')} (n={s['agreement']['n']}, unflagged answers); "
          f"{s['n_disagreements']} disagreement(s) listed for the mapping re-check")
    print(f"excluded {ex['n_items']} item(s): {ex['by_reason']} by arm {ex['by_arm']}")
    if s.get("retest"):
        r = s["retest"]
        kappa = "n/a" if r.get("kappa") is None else f"{r['kappa']:.3f}"
        print(f"retest kappa {kappa} (n={r['n_compared']}); started {r.get('gap_hours_after_read')} h after the read "
              f"was saved; 48-h rule met: {r.get('gap_ok')}")
    for w in s.get("warnings", []):
        print(f"WARNING: {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
