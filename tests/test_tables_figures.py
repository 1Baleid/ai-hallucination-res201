"""Tests for cxrdev.tables and cxrdev.figures (Tables 1-3, Figure 1, PLUS Figure 2). Synthetic data only."""
from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from cxrdev import arm1 as A1
from cxrdev import figures as F
from cxrdev import metrics as M
from cxrdev import tables as T
from cxrdev.items import selection_report
from scripts import synthetic_dryrun as SD
from scripts import verify_join as VJ

RATE_RE = re.compile(r"^-?\d\.\d\d \[-?\d\.\d\d, -?\d\.\d\d\] \(n=\d+\)$")
FORBIDDEN = re.compile(r"radiologist[- ]label|counterfactual|verbatim|label[- ]clean|\bunseen\b", re.I)


@pytest.fixture(scope="module")
def synth():
    items = SD.make_items()
    parsed, _ = SD.make_parsed(items)
    frame = M.scoring_frame(parsed, items)
    summaries = []
    for mk in SD.MODELS:
        for pid in SD.TEST_PROMPTS:
            summaries.append(M.summarise(frame, mk, pid, n_boot=0))
            summaries.append(M.summarise(frame, mk, pid, unparsable_as_wrong=True, n_boot=0, auroc_prompts=()))
    blind = pd.concat([M.blind_rows(parsed, mk) for mk in SD.MODELS], ignore_index=True)
    jf = A1.join_detector(frame, SD.make_detector(items, True))
    arm1 = {mk: A1.run_arm1(jf, jf, mk, n_perm=49, n_boot=0) for mk in SD.MODELS}
    return {"items": items, "parsed": parsed, "frame": frame, "summaries": summaries, "blind": blind, "arm1": arm1}


# ----------------------------------------------------------------------------- formatting
def test_formatters():
    assert T.fmt_num(0.6234) == "0.62" and T.fmt_num(float("nan")) == "n/a" and T.fmt_num(None) == "n/a"
    assert T.fmt_num(True) == "n/a"
    assert T.fmt_ci({"estimate": 0.6234, "lo": 0.53, "hi": 0.701, "n": 118}) == "0.62 [0.53, 0.70] (n=118)"
    assert T.fmt_rate({"estimate": 0.5, "lo": float("nan"), "hi": float("nan"), "n": 4}) == "0.50 (n=4)"
    assert T.fmt_rate({"estimate": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}) == "n/a (n=0)"
    assert T.fmt_ci(None) == "n/a"
    assert T.fmt_ci({"estimate": -0.07, "lo": -0.12, "hi": -0.03}, with_n=False) == "-0.07 [-0.12, -0.03]"
    assert T.fmt_p(0.0004) == "<0.001" and T.fmt_p(0.04123) == "0.041" and T.fmt_p(None) == "n/a"
    assert T.model_label("qwen3vl2b") == "Qwen3-VL-2B" and T.model_label("other") == "other"


# ----------------------------------------------------------------------------- Table 1
def test_table1_without_inputs_says_not_available():
    t1 = T.table1(None, None, None, None, None, None)
    assert list(t1.columns) == list(T.TABLE1_COLUMNS)
    assert (t1["value"].str.contains("not available")).sum() == 5
    assert t1["section"].nunique() == 5


def test_table1_full(synth, tmp_path):
    items = synth["items"]
    rep = selection_report(items, {"rung": "standard", "h1_substitute": False, "n_eligible_patients": 150})
    reading = SD.make_reading(items, tmp_path)
    excl, vsum, _ = VJ.join_verification(items, pd.read_csv(reading["paths"]["reading_key"], dtype=str),
                                         VJ.read_verified(reading["paths"]["verified_items"]),
                                         VJ.read_verified(reading["paths"]["retest"]))
    det = A1.detector_summary(items, SD.make_detector(items, True), n_boot=49)
    match = {"n_ranzcr": 100, "status_counts": {"matched": 90, "ambiguous": 6, "unmatched": 4}, "match_rate": 0.9,
             "ambiguous_share_within_patient": 0.5, "cut": 40, "route_counts": {"margin": 85, "ncc_tiebreak": 5},
             "spotcheck_correct": 60, "run": {"n_spotcheck_pairs": 60}}
    yld = pd.DataFrame({"metric": ["n_images", "n_patients"], "value": [30083, 3255]})
    t1 = T.table1(match, yld, rep, vsum, None, det)
    look = dict(zip(t1["quantity"], t1["value"]))
    assert look["Match rate"] == "0.900" and look["Visual spot check correct"] == "60 of 60"
    assert look["n images"] == "30,083"
    assert look["Pairs (A/B, within-patient contrast)"] == "120"
    assert look["U stratum line"] == "60" and look["U stratum no line"] == "60"
    assert look["Images read"] == "360" and look["Items excluded"] == str(len(excl))
    assert look["Retest Cohen's kappa (>= 48 h later)"].endswith(f"(n={vsum['retest']['n_compared']})")
    assert vsum["retest"]["n"] == 100 and vsum["retest"]["n_compared"] >= 98  # skipped items drop out
    assert RATE_RE.match(look["Agreement with RANZCR CLiP labels"])
    assert look["Dev-pool AUROC"].startswith("0.")
    assert not t1.astype(str).apply(lambda c: c.str.contains(FORBIDDEN)).any().any()
    t1k = T.table1(None, None, None, None, {"kappa": 0.8123, "n": 97}, None)
    assert "0.81 (n=97)" in t1k["value"].tolist()
    assert "0.75" in T.table1(None, None, None, None, 0.75, None)["value"].tolist()


# ----------------------------------------------------------------------------- Table 2
def test_table2_columns_rows_and_cells(synth):
    t2 = T.table2(synth["summaries"], synth["blind"])
    cols = list(t2.columns)
    assert set(T.TABLE2_BASE_COLUMNS) <= set(cols)
    assert cols.index("FAR_U_line") == cols.index("FAR_U") + 1 and cols.index("FAR_U_noline") == cols.index("FAR_U") + 2
    assert len(t2) == 2 * 3 * 2 + 2 * 6
    prim = t2.loc[t2["row"] == "primary"]
    assert all(RATE_RE.match(v) for v in prim["hit_A"]) and all(RATE_RE.match(v) for v in prim["FAR_U_noline"])
    assert prim.loc[prim["prompt"] == "P0", "H0"].tolist() == ["GO", "CONDITIONAL"]
    assert (prim.loc[prim["prompt"] != "P0", "H0"] == "").all()
    assert (prim.loc[prim["prompt"] != "P0", "AUROC_AB"] == "").all()
    sens = t2.loc[t2["row"] == "unparsable as wrong"]
    assert len(sens) == 6 and (sens["unparsable"] == "").all()
    blind = t2.loc[t2["row"].isin(["BLIND-text", "BLIND-grey"])]
    assert len(blind) == 12 and blind["blind_greedy"].isin(["yes", "no", "abstain"]).all()
    assert t2["model"].tolist()[0] == "Qwen3-VL-2B"
    assert T.table2(synth["summaries"], None).shape[0] == 12


# ----------------------------------------------------------------------------- Table 3
def test_table3(synth):
    t3 = T.table3(synth["arm1"])
    assert list(t3.columns) == list(T.TABLE3_COLUMNS)
    assert len(t3) == 2 * (10 + 1)
    q = t3.loc[t3["model"] == "Qwen3-VL-2B"]
    assert q["row"].iloc[0] == "Greedy text answer" and q["row"].iloc[-1] == "Greedy text vs beta = 0 agreement"
    fused = q.loc[q["row"].str.startswith("Fused, beta = 1")].iloc[0]
    assert fused["p"] != "" and fused["delta_BA"] != "" and fused["rule"] == "beta = 1.000"
    assert "fused minus matched" in q.loc[q["row"].str.startswith("Constant bias"), "note"].iloc[0]
    skipped = T.table3([{"model_key": "chexone", "skipped_reason": "no detector scores"}])
    assert skipped["row"].tolist() == ["Arm 1 not run"] and skipped["note"].iloc[0] == "no detector scores"


# ----------------------------------------------------------------------------- writers
def test_write_table_csv_md_tex(tmp_path):
    df = pd.DataFrame({"a_b": ["x & y", "50% |z| > 5"], "c": ["~^\\ {}$#", "plain"]})
    paths = T.write_table(df, "tableX", tmp_path)
    assert {p.name for p in paths.values()} == {"tableX.csv", "tableX.md", "tableX.tex"}
    back = pd.read_csv(paths["csv"], dtype=str, keep_default_na=False)
    assert back.equals(df)
    md = paths["md"].read_text(encoding="utf-8").splitlines()
    assert md[0] == "| a_b | c |" and md[1] == "| --- | --- |" and "50% \\|z\\| > 5" in md[3]
    tex = paths["tex"].read_text(encoding="utf-8")
    assert r"a\_b" in tex and r"x \& y" in tex and r"50\% |z| \textgreater{} 5" in tex
    assert r"\textasciitilde{}\textasciicircum{}\textbackslash{} \{\}\$\#" in tex
    assert tex.startswith(r"\begin{tabular}{ll}") and r"\toprule" in tex and r"\bottomrule" in tex
    first = {k: p.read_bytes() for k, p in paths.items()}
    T.write_table(df, "tableX", tmp_path)
    assert {k: p.read_bytes() for k, p in paths.items()} == first
    assert b"\r\n" not in first["md"] and b"\r\n" not in first["csv"]
    wrapped = T.to_latex(df, caption="Arm 1 & controls", label="tab:arm1")
    assert r"\caption{Arm 1 \& controls}" in wrapped and r"\label{tab:arm1}" in wrapped


# ----------------------------------------------------------------------------- figures
def test_yes_rate_frame_and_group_order(synth):
    yr = F.yes_rate_frame(synth["summaries"])
    assert list(yr.columns) == list(F.YES_RATE_COLUMNS)
    assert len(yr) == 2 * 3 * 4  # sensitivity rows are left out
    assert F.group_order(yr["group"]) == ["A", "B", "U_line", "U_noline"]
    s = [x for x in synth["summaries"] if x["model_key"] == "qwen3vl2b" and x["prompt_id"] == "P0"
         and not x["unparsable_as_wrong"]][0]
    row = yr.loc[(yr["model_key"] == "qwen3vl2b") & (yr["prompt_id"] == "P0") & (yr["group"] == "U_noline")].iloc[0]
    assert row["rate"] == s["FAR_U_noline"]["estimate"] and row["n"] == s["FAR_U_noline"]["n"]
    assert F.group_order(["U_eq1", "B", "zz", "U_ge2", "A"]) == ["A", "B", "U_ge2", "U_eq1", "zz"]


def test_figure1_files_deterministic_and_pair(synth, tmp_path):
    yr = F.yes_rate_frame(synth["summaries"])
    paths = F.figure1(yr, synth["blind"], tmp_path / "figure1")
    assert paths["pdf"].stat().st_size > 1000 and paths["png"].stat().st_size > 1000
    with Image.open(paths["png"]) as im:
        assert im.size[0] > im.size[1] > 300
    first = (paths["pdf"].read_bytes(), paths["png"].read_bytes())
    F.figure1(yr, synth["blind"], tmp_path / "figure1")
    assert (paths["pdf"].read_bytes(), paths["png"].read_bytes()) == first  # no timestamps in the files
    # panel a with two synthetic grey images (not real radiographs) and student arrows
    rng = np.random.default_rng(0)
    for name in ("a.png", "b.png"):
        Image.fromarray(rng.integers(0, 255, (256, 256), dtype=np.uint8)).save(tmp_path / name)
    pair = {"a_path": str(tmp_path / "a.png"), "b_path": str(tmp_path / "b.png"), "a_arrow": [128, 60], "b_arrow": [250, 10]}
    p2 = F.figure1(yr, None, tmp_path / "with_pair", pair)
    assert p2["png"].exists() and p2["png"].read_bytes() != first[1]
    broken = F.figure1(yr, None, tmp_path / "broken", {"a_path": str(tmp_path / "missing.png"), "b_path": "x.png"})
    assert broken["pdf"].exists()
    empty = F.figure1(pd.DataFrame(columns=list(F.YES_RATE_COLUMNS)), None, tmp_path / "empty")
    assert empty["png"].exists()


def test_figure2(synth, tmp_path):
    curves = {mk: M.risk_coverage(synth["frame"], mk) for mk in SD.MODELS}
    verbal = pd.DataFrame([{"model_key": s["model_key"], "prompt_id": s["prompt_id"], "coverage": s["answered"]["estimate"],
                            "risk": s["risk"]["estimate"]} for s in synth["summaries"]
                           if s["prompt_id"] in ("P1", "P2hi") and not s["unparsable_as_wrong"]])
    paths = F.figure2(curves, verbal, tmp_path / "figure2")
    assert paths["pdf"].exists() and paths["png"].exists()
    assert F.figure2({}, None, tmp_path / "figure2_empty")["png"].exists()
    assert math.isclose(curves["qwen3vl2b"]["coverage"].iloc[-1], 1.0)
