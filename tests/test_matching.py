"""Tests for cxrdev.matching on a synthetic test bed (no real data).

The bed: about 300 synthetic "NIH" films (256 px, smooth chest-like structure) from single-film
patients and multi-film patients, sharing a small set of original sizes so that each dimension
bucket holds many films; "RANZCR" JPEGs made by upscaling a subset non-uniformly to the original
size and JPEG-encoding at quality 95; and "RANZCR" images whose source is absent from the NIH set.
Sibling films come in three kinds: a textured twin (same film plus a fine stripe texture of period
6 px: close hash, clearly lower NCC, so the NCC tie-break decides), a near twin (sub-pixel shift,
light noise: close hash and NCC, so the UID stays ambiguous), and a follow-up film (larger shift,
gain, an extra blob).
"""
from __future__ import annotations

import io
import math

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from scipy.ndimage import shift as nd_shift

from cxrdev.imghash import bits_to_hex, ncc, phash_bits, thumb
from cxrdev.matching import (
    MATCH_COLUMNS,
    MatchConfig,
    auto_cut,
    calibration_report,
    candidate_index,
    match_all,
    patient_consistency,
    resolve_one_to_one,
)

SIDE = 256
GRID = 96
SIZES = [(300, 280), (320, 300), (280, 320), (310, 310), (340, 290), (290, 340)]
OFF_KEY_SIZE = (333, 301)  # a JPEG size that no NIH original size has


# --------------------------------------------------------------------------- synthetic images
def _patient(rng: np.random.Generator) -> dict:
    return {
        "cx": 0.5 + rng.normal(0, 0.03), "cy": 0.52 + rng.normal(0, 0.03),
        "bw": 0.42 + rng.normal(0, 0.03), "bh": 0.47 + rng.normal(0, 0.03),
        "lungs": [(0.17 + rng.normal(0, 0.025), -0.05 + rng.normal(0, 0.03), 0.11 + rng.normal(0, 0.015),
                   0.27 + rng.normal(0, 0.03), rng.normal(0, 0.15), 60 + rng.normal(0, 10)) for _ in range(2)],
        "rib_f": rng.uniform(8, 14), "rib_p": rng.uniform(0, 2 * np.pi), "rib_t": rng.normal(0, 0.4),
        "rib_a": rng.uniform(8, 18),
        "blobs": [(rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9), rng.uniform(0.03, 0.14), rng.normal(0, 45))
                  for _ in range(int(rng.integers(6, 13)))],
        "gx": rng.normal(0, 30), "gy": rng.normal(0, 30), "base": rng.uniform(30, 70),
    }


def _render(p: dict, rng: np.random.Generator, shift=(0.0, 0.0), gain=1.0, offset=0.0, noise=0.0,
            extra=0) -> np.ndarray:
    """Chest-like film: body, two lungs with ribs, patient-specific blobs, smooth gradient."""
    yy, xx = np.mgrid[0:GRID, 0:GRID] / GRID
    img = np.full((GRID, GRID), p["base"])
    img += 110 * ((((xx - p["cx"]) / p["bw"]) ** 2 + ((yy - p["cy"]) / p["bh"]) ** 2) < 1)
    lungs = np.zeros_like(img, dtype=bool)
    for sgn, (dx, dy, a, b, ang, depth) in zip((-1, 1), p["lungs"]):
        lx, ly = p["cx"] + sgn * dx, p["cy"] + dy
        u = (xx - lx) * np.cos(ang) + (yy - ly) * np.sin(ang)
        v = -(xx - lx) * np.sin(ang) + (yy - ly) * np.cos(ang)
        m = (u / a) ** 2 + (v / b) ** 2 < 1
        lungs |= m
        img -= depth * m
    img += p["rib_a"] * np.sin(2 * np.pi * p["rib_f"] * (yy + p["rib_t"] * (xx - p["cx"]) ** 2) + p["rib_p"]) * lungs
    for bx, by, s, amp in p["blobs"]:
        img += amp * np.exp(-((xx - bx) ** 2 + (yy - by) ** 2) / (2 * s * s))
    for _ in range(extra):
        bx, by, s, amp = rng.uniform(0.2, 0.8), rng.uniform(0.2, 0.8), rng.uniform(0.02, 0.05), rng.normal(0, 40)
        img += amp * np.exp(-((xx - bx) ** 2 + (yy - by) ** 2) / (2 * s * s))
    img += p["gx"] * (xx - 0.5) + p["gy"] * (yy - 0.5)
    big = np.asarray(Image.fromarray(img.astype(np.float32), mode="F").resize((SIDE, SIDE), Image.Resampling.BICUBIC),
                     dtype=np.float64)
    if shift != (0.0, 0.0):
        big = nd_shift(big, shift, mode="nearest", order=1)
    big = gain * big + offset
    if noise:
        big = big + rng.normal(0, noise, big.shape)
    return np.clip(np.round(big), 0, 255).astype(np.uint8)


def _jpeg(arr: np.ndarray, size: tuple[int, int], quality: int = 95, noise: float = 0.0,
          rng: np.random.Generator | None = None) -> np.ndarray:
    """Upscale non-uniformly to ``size`` (w, h), JPEG-encode, decode (the 'RANZCR' copy)."""
    big = np.asarray(Image.fromarray(arr).resize(size, Image.Resampling.BICUBIC), dtype=np.float64)
    if noise and rng is not None:
        big = np.clip(big + rng.normal(0, noise, big.shape), 0, 255)
    buf = io.BytesIO()
    Image.fromarray(big.astype(np.uint8)).save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    with Image.open(buf) as im:
        return np.array(im.convert("L"))


class Bed:
    """Container for the synthetic test bed."""

    def __init__(self) -> None:
        self.nih_rows: list[dict] = []
        self.nih_pixels: dict[str, np.ndarray] = {}
        self.ranzcr_rows: list[dict] = []
        self.ranzcr_pixels: dict[str, np.ndarray] = {}
        self.truth: dict[str, str] = {}      # uid -> source NIH filename ("" when absent)
        self.kind: dict[str, str] = {}       # uid -> single | sibling | absent | duplicate_* | conflict_* | offkey
        self._pid = 0

    def add_nih(self, pixels: np.ndarray, size: tuple[int, int], pid: int, follow: int) -> str:
        name = f"{pid:08d}_{follow:03d}.png"
        self.nih_rows.append({"nih_filename": name, "nih_patient_id": pid, "followup": follow, "age": 50,
                              "sex": "M", "view": "PA", "orig_w": size[0], "orig_h": size[1],
                              "finding_labels": "No Finding"})
        self.nih_pixels[name] = pixels
        return name

    def add_ranzcr(self, uid: str, rp: str, pixels: np.ndarray, truth: str, kind: str) -> None:
        h, w = pixels.shape
        self.ranzcr_rows.append({"uid": uid, "ranzcr_patient_id": rp, "jpg_w": w, "jpg_h": h})
        self.ranzcr_pixels[uid] = pixels
        self.truth[uid] = truth
        self.kind[uid] = kind

    def frames(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        meta = pd.DataFrame(self.nih_rows)
        nih_hashes = pd.DataFrame({"nih_filename": list(self.nih_pixels),
                                   "ph": [bits_to_hex(phash_bits(a)) for a in self.nih_pixels.values()]})
        rz = pd.DataFrame(self.ranzcr_rows)
        rz["ph"] = [bits_to_hex(phash_bits(self.ranzcr_pixels[u])) for u in rz["uid"]]
        return rz, nih_hashes, meta

    def ncc_fn(self):
        cache_r: dict[str, np.ndarray] = {}
        cache_n: dict[str, np.ndarray] = {}
        calls = []

        def fn(uid: str, name: str) -> float:
            calls.append((uid, name))
            if uid not in cache_r:
                cache_r[uid] = thumb(self.ranzcr_pixels[uid], 128)
            if name not in cache_n:
                cache_n[name] = thumb(self.nih_pixels[name], 128)
            return ncc(cache_r[uid], cache_n[name])

        fn.calls = calls
        return fn


def _build_bed(seed: int = 20261102) -> Bed:
    rng = np.random.default_rng(seed)
    bed = Bed()
    uid_n = [0]

    def uid() -> str:
        uid_n[0] += 1
        return f"test-uid-{uid_n[0]:04d}"

    # 200 single-film patients (60 of them get a RANZCR copy)
    singles = []
    for i in range(200):
        pid = 1000 + i
        p = _patient(rng)
        size = SIZES[i % len(SIZES)]
        name = bed.add_nih(_render(p, rng), size, pid, 0)
        singles.append((pid, name, size))
    for pid, name, size in singles[:60]:
        u = uid()
        bed.add_ranzcr(u, f"RP{pid}", _jpeg(bed.nih_pixels[name], size), name, "single")

    # 45 multi-film patients with siblings in the SAME bucket (15 of each sibling kind)
    for i in range(45):
        pid = 3000 + i
        p = _patient(rng)
        size = SIZES[i % len(SIZES)]
        src_px = _render(p, rng)
        kind = ("textured", "near", "followup")[i % 3]
        if kind == "textured":
            yy, xx = np.mgrid[0:SIDE, 0:SIDE]
            ang, phase = rng.uniform(0, np.pi), rng.uniform(0, 2 * np.pi)
            stripes = 30.0 * np.sin(2 * np.pi * (xx * np.cos(ang) + yy * np.sin(ang)) / 6.0 + phase)
            sib_px = np.clip(np.round(src_px + stripes), 0, 255).astype(np.uint8)
        elif kind == "near":
            sib_px = _render(p, rng, shift=(rng.uniform(0.3, 0.8), rng.uniform(0.3, 0.8)), noise=2.0)
        else:
            sib_px = _render(p, rng, shift=(rng.uniform(1.5, 3.5) * rng.choice([-1, 1]),
                                            rng.uniform(1.5, 3.5) * rng.choice([-1, 1])),
                             gain=rng.uniform(0.9, 1.1), offset=rng.uniform(-8, 8), noise=3.0, extra=1)
        src = bed.add_nih(src_px, size, pid, 0)
        bed.add_nih(sib_px, size, pid, 1)
        u = uid()
        bed.add_ranzcr(u, f"RP{pid}", _jpeg(src_px, size), src, f"sibling_{kind}")

    # 30 RANZCR images whose source is absent from the NIH set
    for i in range(30):
        p = _patient(rng)
        px = _render(p, rng)
        bed.add_ranzcr(uid(), f"RPA{i}", _jpeg(px, SIZES[i % len(SIZES)]), "", "absent")

    # duplicate claim: two RANZCR copies of one single-film NIH image (one clean, one degraded)
    pid, name, size = singles[100]
    bed.add_ranzcr(uid(), f"RP{pid}", _jpeg(bed.nih_pixels[name], size), name, "duplicate_clean")
    bed.add_ranzcr(uid(), f"RP{pid}", _jpeg(bed.nih_pixels[name], size, quality=60, noise=8.0, rng=rng), name,
                   "duplicate_degraded")

    # patient conflict: RANZCR patient RPX = two films of NIH patient 5000 (different buckets) plus one
    # film of the single-film NIH patient singles[101]
    p = _patient(rng)
    a0 = bed.add_nih(_render(p, rng), SIZES[0], 5000, 0)
    a1 = bed.add_nih(_render(p, rng, shift=(3.0, -2.0), gain=1.05, noise=3.0, extra=2), SIZES[1], 5000, 1)
    bed.add_ranzcr(uid(), "RPX", _jpeg(bed.nih_pixels[a0], SIZES[0]), a0, "conflict_majority")
    bed.add_ranzcr(uid(), "RPX", _jpeg(bed.nih_pixels[a1], SIZES[1]), a1, "conflict_majority")
    pid_b, name_b, size_b = singles[101]
    bed.add_ranzcr(uid(), "RPX", _jpeg(bed.nih_pixels[name_b], size_b), name_b, "conflict_minority")

    # empty bucket: a JPEG whose size matches no NIH original size (source is singles[102])
    pid_c, name_c, _ = singles[102]
    bed.add_ranzcr(uid(), f"RP{pid_c}", _jpeg(bed.nih_pixels[name_c], OFF_KEY_SIZE), name_c, "offkey")
    return bed


@pytest.fixture(scope="module")
def bed() -> Bed:
    return _build_bed()


@pytest.fixture(scope="module")
def frames(bed: Bed):
    return bed.frames()


@pytest.fixture(scope="module")
def result(bed: Bed, frames):
    rz, nh, meta = frames
    return match_all(rz, nh, meta, MatchConfig(), ncc_fn=bed.ncc_fn())


def _no_wrong_matches(table: pd.DataFrame, bed: Bed) -> None:
    for row in table.itertuples(index=False):
        if row.status == "matched":
            assert row.nih_filename == bed.truth[row.uid], (row.uid, bed.kind[row.uid], row)


# --------------------------------------------------------------------------- bed sanity
def test_bed_has_shared_buckets_and_size(bed: Bed, frames) -> None:
    rz, nh, meta = frames
    assert 290 <= len(meta) <= 320
    idx = candidate_index(meta)
    assert all(len(v) >= 30 for k, v in idx.items())
    assert OFF_KEY_SIZE not in idx
    assert len(rz) >= 130


# --------------------------------------------------------------------------- full rule with NCC
def test_schema_and_types(result) -> None:
    table, stats = result
    assert tuple(table.columns) == MATCH_COLUMNS
    assert table["uid"].is_unique
    assert set(table["status"]) <= {"matched", "ambiguous", "unmatched", "duplicate_conflict", "patient_conflict"}
    assert set(table["route"]) <= {"margin", "ncc_tiebreak", "none"}
    assert set(table["key_used"]) <= {"dim", "global"}
    for c in ("jpg_w", "jpg_h", "nih_patient_id", "d1", "d2", "n_candidates"):
        assert table[c].dtype == np.int64, c
    unmatched = table[table["status"].isin(["unmatched", "ambiguous"])]
    assert (unmatched["nih_filename"] == "").all() and (unmatched["nih_patient_id"] == -1).all()


def test_no_wrong_matches_with_ncc(result, bed: Bed) -> None:
    _no_wrong_matches(result[0], bed)


def test_every_non_sibling_present_image_matched(result, bed: Bed) -> None:
    t = result[0].set_index("uid")
    for uid, kind in bed.kind.items():
        if kind in ("single", "duplicate_clean", "conflict_majority", "offkey"):
            assert t.loc[uid, "status"] == "matched", (uid, kind, t.loc[uid].to_dict())
            assert t.loc[uid, "nih_filename"] == bed.truth[uid]


def test_siblings_matched_by_tiebreak_or_ambiguous(result, bed: Bed) -> None:
    t = result[0].set_index("uid")
    sib = [u for u, k in bed.kind.items() if k.startswith("sibling_")]
    assert len(sib) == 45
    statuses = t.loc[sib, "status"]
    assert set(statuses) <= {"matched", "ambiguous"}
    for u in sib:
        if t.loc[u, "status"] == "matched":
            assert t.loc[u, "nih_filename"] == bed.truth[u]
    textured = [u for u in sib if bed.kind[u] == "sibling_textured"]
    near = [u for u in sib if bed.kind[u] == "sibling_near"]
    tie = t.loc[textured]
    assert (tie["route"] == "ncc_tiebreak").sum() >= 10, tie[["d1", "d2", "ncc1", "ncc2", "route", "status"]]
    assert (t.loc[near, "status"] == "ambiguous").sum() >= 5
    # a tie-break match records the NCC winner and the runner-up
    tb = t[t["route"] == "ncc_tiebreak"]
    assert (tb["ncc1"] >= 0.95).all() and ((tb["ncc1"] - tb["ncc2"]) >= 0.02).all()
    # ambiguity among siblings is within one NIH patient
    assert result[1]["ambiguous_share_within_patient"] == pytest.approx(1.0)


def test_absent_sources_unmatched(result, bed: Bed) -> None:
    t = result[0].set_index("uid")
    absent = [u for u, k in bed.kind.items() if k == "absent"]
    assert (t.loc[absent, "status"] == "unmatched").all(), t.loc[absent, ["d1", "d2", "status"]]
    assert (t.loc[absent, "d1"] > result[1]["cut"]).all()


def test_duplicate_claim_resolved(result, bed: Bed) -> None:
    t = result[0].set_index("uid")
    clean = [u for u, k in bed.kind.items() if k == "duplicate_clean"][0]
    degraded = [u for u, k in bed.kind.items() if k == "duplicate_degraded"][0]
    assert t.loc[clean, "status"] == "matched"
    assert t.loc[degraded, "status"] == "duplicate_conflict"
    assert t.loc[degraded, "nih_filename"] == t.loc[clean, "nih_filename"]
    matched = result[0][result[0]["status"] == "matched"]
    assert matched["nih_filename"].is_unique


def test_patient_conflict_flagged(result, bed: Bed) -> None:
    t = result[0].set_index("uid")
    major = [u for u, k in bed.kind.items() if k == "conflict_majority"]
    minor = [u for u, k in bed.kind.items() if k == "conflict_minority"]
    assert (t.loc[major, "status"] == "matched").all()
    assert (t.loc[major, "nih_patient_id"] == 5000).all()
    assert t.loc[minor[0], "status"] == "patient_conflict"
    matched = result[0][result[0]["status"] == "matched"]
    assert (matched.groupby("ranzcr_patient_id")["nih_patient_id"].nunique() == 1).all()


def test_empty_bucket_uses_global(result, bed: Bed, frames) -> None:
    t = result[0].set_index("uid")
    off = [u for u, k in bed.kind.items() if k == "offkey"][0]
    assert t.loc[off, "key_used"] == "global"
    assert t.loc[off, "n_candidates"] == len(frames[2])
    assert t.loc[off, "status"] == "matched"
    others = result[0][result[0]["uid"] != off]
    assert (others["key_used"] == "dim").all()


def test_stats_report(result) -> None:
    table, stats = result
    assert stats["n_ranzcr"] == len(table)
    assert sum(stats["status_counts"].values()) == len(table)
    assert stats["cut_rule"] == "auto_otsu_valley_capped"
    assert 5 <= stats["cut"] <= 64
    assert len(stats["d1_hist"]) == 257 and sum(stats["d1_hist"]) == len(table)
    assert stats["match_rate"] == pytest.approx((table["status"] == "matched").mean())
    assert stats["config"]["margin_bits"] == 16
    assert stats["n_ncc_calls"] > 0
    import json
    json.dumps(stats)  # serialisable


def test_deterministic(bed: Bed, frames, result) -> None:
    rz, nh, meta = frames
    again, stats = match_all(rz, nh, meta, MatchConfig(), ncc_fn=bed.ncc_fn())
    pd.testing.assert_frame_equal(again, result[0])
    # row order of the NIH inputs does not matter
    shuffled, _ = match_all(rz, nh.sample(frac=1.0, random_state=3), meta.sample(frac=1.0, random_state=4),
                            MatchConfig(), ncc_fn=bed.ncc_fn())
    pd.testing.assert_frame_equal(shuffled, result[0])


# --------------------------------------------------------------------------- variants
def test_global_fallback_when_key_off(bed: Bed, frames) -> None:
    rz, nh, meta = frames
    table, stats = match_all(rz, nh, meta, MatchConfig(use_dimension_key=False), ncc_fn=bed.ncc_fn())
    assert (table["key_used"] == "global").all()
    assert (table["n_candidates"] == len(meta)).all()
    _no_wrong_matches(table, bed)
    t = table.set_index("uid")
    for uid, kind in bed.kind.items():
        if kind in ("single", "duplicate_clean", "offkey"):
            assert t.loc[uid, "status"] == "matched", (uid, kind)
        if kind == "absent":
            assert t.loc[uid, "status"] == "unmatched"


def test_without_ncc_margin_only(bed: Bed, frames) -> None:
    rz, nh, meta = frames
    table, stats = match_all(rz, nh, meta, MatchConfig(), ncc_fn=None)
    _no_wrong_matches(table, bed)
    assert (table["route"] != "ncc_tiebreak").all()
    assert table["ncc1"].isna().all() and table["ncc2"].isna().all()
    t = table.set_index("uid")
    singles = [u for u, k in bed.kind.items() if k == "single"]
    assert (t.loc[singles, "status"] == "matched").all()
    near = [u for u, k in bed.kind.items() if k == "sibling_near"]
    assert (t.loc[near, "status"] == "ambiguous").sum() >= 10


def test_fixed_cut_and_wide_margin(bed: Bed, frames) -> None:
    rz, nh, meta = frames
    table, stats = match_all(rz, nh, meta, MatchConfig(cut=0, margin_bits=200), ncc_fn=None)
    assert stats["cut"] == 0 and stats["cut_rule"] == "fixed"
    # nothing can pass a 200-bit margin unless it is the only candidate
    assert (table.loc[table["status"] == "matched", "n_candidates"] == 1).all()
    assert set(table["status"]) <= {"ambiguous", "unmatched", "matched"}
    _no_wrong_matches(table, bed)


def test_topk_one_still_compares_two_candidates(bed: Bed, frames) -> None:
    rz, nh, meta = frames
    table, _ = match_all(rz, nh, meta, MatchConfig(topk_ncc=1), ncc_fn=bed.ncc_fn())
    _no_wrong_matches(table, bed)
    tb = table[table["route"] == "ncc_tiebreak"]
    assert len(tb) >= 10 and tb["ncc2"].notna().all()
    with pytest.raises(ValueError):
        match_all(rz, nh, meta, MatchConfig(topk_ncc=0))


def test_ncc_failure_blocks_margin_route(bed: Bed, frames) -> None:
    rz, nh, meta = frames
    table, _ = match_all(rz, nh, meta, MatchConfig(), ncc_fn=lambda u, n: float("nan"))
    assert (table["status"] != "matched").all()
    assert set(table["status"]) <= {"ambiguous", "unmatched"}


def test_single_candidate_bucket(bed: Bed) -> None:
    rng = np.random.default_rng(5)
    src = _render(_patient(rng), rng)
    meta = pd.DataFrame([{"nih_filename": "00000001_000.png", "nih_patient_id": 1, "orig_w": 300, "orig_h": 280}])
    nh = pd.DataFrame({"nih_filename": ["00000001_000.png"], "ph": [bits_to_hex(phash_bits(src))]})
    jp = _jpeg(src, (300, 280))
    rz = pd.DataFrame([{"uid": "u1", "ranzcr_patient_id": "r1", "jpg_w": 300, "jpg_h": 280,
                        "ph": bits_to_hex(phash_bits(jp))}])
    table, stats = match_all(rz, nh, meta, MatchConfig(cut=20), ncc_fn=lambda u, n: ncc(thumb(jp), thumb(src)))
    row = table.iloc[0]
    assert row["status"] == "matched" and row["route"] == "margin" and row["d2"] == -1 and row["n_candidates"] == 1


def test_empty_inputs() -> None:
    meta = pd.DataFrame(columns=["nih_filename", "nih_patient_id", "orig_w", "orig_h"])
    nh = pd.DataFrame(columns=["nih_filename", "ph"])
    rz = pd.DataFrame([{"uid": "u1", "ranzcr_patient_id": "r1", "jpg_w": 10, "jpg_h": 10, "ph": "00" * 32}])
    table, stats = match_all(rz, nh, meta, MatchConfig())
    assert table.loc[0, "status"] == "unmatched" and table.loc[0, "n_candidates"] == 0
    assert table.loc[0, "d1"] == -1 and table.loc[0, "d2"] == -1
    empty_rz = rz.iloc[0:0]
    t2, s2 = match_all(empty_rz, pd.DataFrame({"nih_filename": ["00000001_000.png"], "ph": ["ab" * 32]}),
                       meta, MatchConfig())
    assert len(t2) == 0 and tuple(t2.columns) == MATCH_COLUMNS and s2["n_ranzcr"] == 0


def test_input_validation() -> None:
    meta = pd.DataFrame([{"nih_filename": "00000001_000.png", "nih_patient_id": 1, "orig_w": 1, "orig_h": 1}])
    nh = pd.DataFrame({"nih_filename": ["00000001_000.png"], "ph": ["ab" * 32]})
    rz = pd.DataFrame([{"uid": "u1", "ranzcr_patient_id": "r1", "jpg_w": 1, "jpg_h": 1, "ph": "ab" * 32}])
    with pytest.raises(ValueError, match="lacks columns"):
        match_all(rz.drop(columns=["jpg_w"]), nh, meta, MatchConfig())
    with pytest.raises(ValueError, match="duplicated uid"):
        match_all(pd.concat([rz, rz]), nh, meta, MatchConfig())
    with pytest.raises(ValueError, match="different lengths"):
        match_all(rz.assign(ph="ab" * 8), nh, meta, MatchConfig())


def test_nih_without_metadata_is_global_candidate() -> None:
    rng = np.random.default_rng(9)
    a = _render(_patient(rng), rng)
    b = _render(_patient(rng), rng)
    meta = pd.DataFrame([{"nih_filename": "00000002_000.png", "nih_patient_id": 2, "orig_w": 300, "orig_h": 280}])
    nh = pd.DataFrame({"nih_filename": ["00000001_000.png", "00000002_000.png"],
                       "ph": [bits_to_hex(phash_bits(a)), bits_to_hex(phash_bits(b))]})
    rz = pd.DataFrame([{"uid": "u1", "ranzcr_patient_id": "r1", "jpg_w": 999, "jpg_h": 999,
                        "ph": bits_to_hex(phash_bits(_jpeg(a, (999, 999))))}])
    table, _ = match_all(rz, nh, meta, MatchConfig(cut=20))
    assert table.loc[0, "key_used"] == "global" and table.loc[0, "status"] == "matched"
    assert table.loc[0, "nih_filename"] == "00000001_000.png"
    assert table.loc[0, "nih_patient_id"] == 1  # recovered from the filename


# --------------------------------------------------------------------------- unit tests of the steps
def _table(rows: list[dict]) -> pd.DataFrame:
    base = {"jpg_w": 1, "jpg_h": 1, "d2": 50, "ncc2": math.nan, "n_candidates": 5, "key_used": "dim",
            "route": "margin", "status": "matched"}
    t = pd.DataFrame([{**base, **r} for r in rows])
    return t.loc[:, list(MATCH_COLUMNS)]


def test_resolve_one_to_one_rules() -> None:
    t = _table([
        {"uid": "a", "ranzcr_patient_id": "p", "nih_filename": "f1", "nih_patient_id": 1, "d1": 3, "ncc1": 0.97},
        {"uid": "b", "ranzcr_patient_id": "p", "nih_filename": "f1", "nih_patient_id": 1, "d1": 2, "ncc1": 0.96},
        {"uid": "c", "ranzcr_patient_id": "q", "nih_filename": "f2", "nih_patient_id": 2, "d1": 2, "ncc1": 0.96},
        {"uid": "d", "ranzcr_patient_id": "q", "nih_filename": "f2", "nih_patient_id": 2, "d1": 2, "ncc1": 0.99},
        {"uid": "e", "ranzcr_patient_id": "r", "nih_filename": "f3", "nih_patient_id": 3, "d1": 1, "ncc1": 0.98},
        {"uid": "f", "ranzcr_patient_id": "r", "nih_filename": "f3", "nih_patient_id": 3, "d1": 1, "ncc1": 0.98},
        {"uid": "g", "ranzcr_patient_id": "s", "nih_filename": "f4", "nih_patient_id": 4, "d1": 1, "ncc1": math.nan},
        {"uid": "h", "ranzcr_patient_id": "s", "nih_filename": "f4", "nih_patient_id": 4, "d1": 1, "ncc1": 0.5},
        {"uid": "i", "ranzcr_patient_id": "t", "nih_filename": "f5", "nih_patient_id": 5, "d1": 0, "ncc1": 0.99,
         "status": "ambiguous"},
        {"uid": "j", "ranzcr_patient_id": "t", "nih_filename": "f5", "nih_patient_id": 5, "d1": 4, "ncc1": 0.99},
    ])
    out = resolve_one_to_one(t).set_index("uid")["status"].to_dict()
    assert out["b"] == "matched" and out["a"] == "duplicate_conflict"          # smaller d1 wins
    assert out["d"] == "matched" and out["c"] == "duplicate_conflict"          # then higher ncc1
    assert out["e"] == out["f"] == "duplicate_conflict"                         # exact tie: no winner
    assert out["h"] == "matched" and out["g"] == "duplicate_conflict"          # NaN ncc ranks last
    assert out["i"] == "ambiguous" and out["j"] == "matched"                    # only matched rows claim
    assert (t["status"] == "matched").sum() == 9                                # input not modified


def test_patient_consistency_rules() -> None:
    t = _table([
        {"uid": "a", "ranzcr_patient_id": "p", "nih_filename": "f1", "nih_patient_id": 1, "d1": 1, "ncc1": 0.99},
        {"uid": "b", "ranzcr_patient_id": "p", "nih_filename": "f2", "nih_patient_id": 1, "d1": 1, "ncc1": 0.99},
        {"uid": "c", "ranzcr_patient_id": "p", "nih_filename": "f3", "nih_patient_id": 2, "d1": 1, "ncc1": 0.99},
        {"uid": "d", "ranzcr_patient_id": "q", "nih_filename": "f4", "nih_patient_id": 3, "d1": 1, "ncc1": 0.99},
        {"uid": "e", "ranzcr_patient_id": "q", "nih_filename": "f5", "nih_patient_id": 4, "d1": 1, "ncc1": 0.99},
        {"uid": "f", "ranzcr_patient_id": "r", "nih_filename": "f6", "nih_patient_id": 5, "d1": 1, "ncc1": 0.99},
        {"uid": "g", "ranzcr_patient_id": "r", "nih_filename": "", "nih_patient_id": -1, "d1": 40, "ncc1": math.nan,
         "status": "unmatched", "route": "none"},
    ])
    out = patient_consistency(t).set_index("uid")["status"].to_dict()
    assert out["a"] == out["b"] == "matched" and out["c"] == "patient_conflict"
    assert out["d"] == out["e"] == "patient_conflict"   # 1-1 tie: no majority
    assert out["f"] == "matched" and out["g"] == "unmatched"


def test_auto_cut_cases() -> None:
    rng = np.random.default_rng(1)
    bimodal = np.r_[rng.integers(0, 16, 1000), rng.integers(80, 131, 250)]
    c = auto_cut(bimodal)
    assert 16 <= c < 80 and c <= 64
    assert auto_cut(np.r_[rng.integers(0, 10, 500), rng.integers(30, 60, 500)]) in range(10, 31)
    assert auto_cut(rng.integers(0, 21, 1000)) == 64                     # unimodal: cap
    assert auto_cut(rng.integers(80, 131, 100)) == 64                    # no matches at all: cap
    assert auto_cut(np.array([])) == 64 and auto_cut(np.array([5])) == 64 and auto_cut(np.full(10, 7)) == 64
    assert auto_cut(np.r_[np.zeros(100), np.full(100, 200)], n_bits=256) == 64   # valley beyond the cap
    assert auto_cut(np.r_[np.zeros(100), np.full(100, 40)], n_bits=64) <= 16      # cap scales with bits
    assert auto_cut(np.r_[np.zeros(50), np.full(50, 30), [-1, np.nan]]) == 15     # invalid values ignored


def test_candidate_index() -> None:
    meta = pd.DataFrame({"orig_w": [10, 20, 10, 0, np.nan], "orig_h": [5, 5, 5, 5, 5]})
    idx = candidate_index(meta)
    assert set(idx) == {(10, 5), (20, 5)}
    assert idx[(10, 5)].tolist() == [0, 2] and idx[(20, 5)].tolist() == [1]


def test_calibration_report_without_top2() -> None:
    t = _table([
        {"uid": "a", "ranzcr_patient_id": "p", "nih_filename": "f1", "nih_patient_id": 1, "d1": 2, "ncc1": 0.99},
        {"uid": "b", "ranzcr_patient_id": "p", "nih_filename": "", "nih_patient_id": -1, "d1": 5, "ncc1": math.nan,
         "status": "ambiguous", "route": "none", "d2": 7},
        {"uid": "c", "ranzcr_patient_id": "z", "nih_filename": "", "nih_patient_id": -1, "d1": 90,
         "ncc1": math.nan, "status": "unmatched", "route": "none", "d2": -1},
    ])
    rep = calibration_report(t, cut=30)
    assert rep["status_counts"]["ambiguous"] == 1 and rep["ambiguous_share_within_patient"] is None
    assert rep["ambiguous_uids_in_matched_patients_share"] == 1.0
    assert rep["cut"] == 30 and rep["match_rate"] == pytest.approx(1 / 3)
    assert rep["margin_hist"]["lo"] == 2 and sum(rep["margin_hist"]["counts"]) == 2
    assert rep["d1_hist"][90] == 1
