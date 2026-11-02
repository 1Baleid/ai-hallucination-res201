"""End-to-end tests of scripts/synthetic_dryrun.py and scripts/analyze.py. Synthetic data only.

The dry run plants known effects (``truth.json``), runs ``verify_join`` and the whole analysis
with small resample counts, then Arm 1 once more with a pure-noise detector. These tests check
that every output exists with its documented columns and that the analysis recovers what was
planted: hit_A, FAR_B and FAR_U inside their Wilson intervals, the H1 co-occurrence difference,
the H3 abstention increase under P2hi, a positive Arm 1 gain with an informative detector and none
with a noise detector, the exclusions, the H0 classes and the numbers.json keys. They also run
analyze.py with optional inputs missing. No RANZCR or NIH record is read or written.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from cxrdev import metrics as M
from cxrdev import tables as T
from cxrdev.items import read_items
from scripts import analyze as AZ
from scripts import synthetic_dryrun as SD

N_SMALL = 199  # resamples and permutations for the main run (the full analysis uses 9,999)
N_TINY = 19    # runs that only check plumbing
FORBIDDEN = re.compile(r"radiologist[- ]label|counterfactual|verbatim|label[- ]clean|\bunseen\b", re.I)
RATE_KEYS = ("hit_A", "FAR_B", "FAR_U", "FAR_U_line", "FAR_U_noline", "yes_rate", "abstention", "accuracy",
             "unparsable_rate", "truncated_rate", "persistence", "pair_correct")
#: Report placeholders that no analysis input carries; the student supplies them with analyze.py --number.
EXTERNAL_NUMBER_KEYS = {"parse.handcheck_agree"}  # run.medsiglip.dtype now comes from S9
REPORT_DIR = Path(__file__).resolve().parents[1] / "report"
PLACEHOLDER_RE = re.compile(r"\[\[NUM:([A-Za-z0-9_.\-]+)(?:\|(pct|ci|p|int))?\]\]")


def _table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def dry(tmp_path_factory):
    out = tmp_path_factory.mktemp("dryrun")
    r = SD.run_dryrun(out, N_SMALL, N_SMALL, log=None)
    res = r["results_dir"]
    r["items"] = read_items(r["paths"]["items"])
    r["parsed"] = M.read_parsed(r["paths"]["parsed"])
    r["exclusions"] = M.read_exclusions(r["paths"]["exclusions"])
    r["frame"] = M.scoring_frame(r["parsed"], r["items"])
    r["numbers"] = _json(res / "numbers.json")
    r["hyp"] = _json(res / "hypotheses.json")
    r["summ"] = {(s["model_key"], s["prompt_id"]): s for s in r["analysis"]["summaries"] if not s["unparsable_as_wrong"]}
    return r


def _planted_far_u(planted: dict, items: pd.DataFrame) -> float:
    """FAR_U implied by the planted stratum rates, weighted by the stratum sizes."""
    u = items.loc[items["arm"] == "U", "stratum"]
    n1 = int(u.isin(["line", "ge2"]).sum())
    n0 = int(len(u) - n1)
    return (planted["U_1"][0] * n1 + planted["U_0"][0] * n0) / (n1 + n0)


# ----------------------------------------------------------------------------- outputs
def test_every_output_file_exists_with_documented_columns(dry):
    res = dry["results_dir"]
    for name in AZ.MANAGED_FILES:
        assert (res / name).is_file() and (res / name).stat().st_size > 0, name
    t1, t2, t3 = (_table(res / f"table{i}.csv") for i in (1, 2, 3))
    assert list(t1.columns) == list(T.TABLE1_COLUMNS)
    cols2 = list(T.TABLE2_BASE_COLUMNS)
    cols2[cols2.index("FAR_U") + 1:cols2.index("FAR_U") + 1] = ["FAR_U_line", "FAR_U_noline"]
    assert list(t2.columns) == cols2
    assert list(t3.columns) == list(T.TABLE3_COLUMNS)
    # md and tex carry exactly the csv table
    for stem, t in (("table1", t1), ("table2", t2), ("table3", t3)):
        assert (res / f"{stem}.md").read_text(encoding="utf-8") == T.to_markdown(t)
        assert (res / f"{stem}.tex").read_text(encoding="utf-8") == T.to_latex(t)
    # 2 models x 3 prompts x (primary + sensitivity) + 2 x 6 blind rows; Arm 1: 10 rows + agreement per model
    assert len(t2) == 2 * 3 * 2 + 2 * 6
    assert len(t3) == 2 * 11
    assert set(t1["section"]) == {
        "Image matching (RANZCR CLiP JPEG to NIH PNG)", "RANZCR CLiP label yield (43 trained labellers; before matching)",
        "Item selection", "Blind reading (student; blind to arm and to model output)",
        "Automatic verifier (MedSigLIP zero-shot; never used to exclude)"}
    assert not t1["value"].str.contains("not available").any()
    for stem in ("figure1", "figure2"):
        assert (res / f"{stem}.pdf").read_bytes()[:5] == b"%PDF-"
        with Image.open(res / f"{stem}.png") as im:
            assert im.format == "PNG" and min(im.size) > 400
    assert {"meta", "H0", "H1", "H2", "H3", "H5", "detector"} <= set(dry["hyp"])
    assert dry["hyp"]["meta"]["n_boot"] == N_SMALL and dry["hyp"]["meta"]["n_perm"] == N_SMALL
    assert dry["hyp"]["meta"]["exclusions_applied"] is True and dry["hyp"]["meta"]["log_skips"] == []
    # undefined numbers are null (None), never NaN (valid JSON)
    assert all(v is None or isinstance(v, (bool, int, float, str)) for v in dry["numbers"].values())
    assert not any(isinstance(v, float) and v != v for v in dry["numbers"].values())
    assert (Path(dry["paths"]["noise_control"])).is_file() and (Path(dry["paths"]["truth"])).is_file()


def test_results_hold_aggregates_only_and_project_wording(dry):
    items = dry["items"]
    key = pd.read_csv(dry["paths"]["reading_key"], dtype=str)
    secrets = sorted(set(items["item_id"]) | set(items["nih_filename"]) | set(items["uid"]) | set(key["read_id"]))
    leak = re.compile("|".join(re.escape(s) for s in secrets))
    texts = [p for p in sorted(dry["results_dir"].iterdir()) if p.suffix not in (".png", ".pdf")]
    assert {p.name for p in texts} >= {"table1.csv", "hypotheses.json", "numbers.json", "analysis_log.txt"}
    for p in texts:
        text = p.read_text(encoding="utf-8")
        assert not FORBIDDEN.search(text), p.name
        m = leak.search(text)
        assert m is None, f"{p.name} holds an item-level value ({m.group(0) if m else ''})"


def test_analysis_is_deterministic(dry, tmp_path):
    p = dry["paths"]
    inputs = AZ.AnalysisInputs(items=p["items"], parsed=p["parsed"], detector=p["detector"], exclusions=p["exclusions"],
                               match_stats=p["match_stats"], yield_table=p["yield"], selection=p["selection"],
                               verification=p["verification"], run_dir=p["run_dir"], spotcheck_correct=59)
    AZ.run_analysis(inputs, tmp_path, N_SMALL, N_SMALL, log=None)
    for name in AZ.MANAGED_FILES:
        assert (tmp_path / name).read_bytes() == (dry["results_dir"] / name).read_bytes(), name


# ----------------------------------------------------------------------------- planted rates
def test_counts_without_exclusions_equal_the_planted_counts(dry):
    counts = dry["truth"]["counts"]
    for mk in SD.MODELS:
        for pid in SD.TEST_PROMPTS:
            c = counts[mk][pid]
            s = M.summarise(dry["frame"], mk, pid, None, n_boot=0)
            assert (s["hit_A"]["k"], s["hit_A"]["n"]) == (c["A"]["n_yes"], c["A"]["n_valid"])
            assert (s["FAR_B"]["k"], s["FAR_B"]["n"]) == (c["B"]["n_yes"], c["B"]["n_valid"])
            assert s["FAR_U"]["k"] == c["U_1"]["n_yes"] + c["U_0"]["n_yes"]
            assert s["FAR_U"]["n"] == c["U_1"]["n_valid"] + c["U_0"]["n_valid"]
            groups = ("A", "B", "U_1", "U_0")
            assert s["abstention"]["k"] == sum(c[g]["n_abstain"] for g in groups)
            assert s["unparsable_rate"]["k"] == sum(c[g]["n_unparsable"] for g in groups)
            assert s["truncated_rate"]["k"] == sum(c[g]["n_truncated"] for g in groups)
            assert s["n_rows"] == 360 and s["n_rows_excluded"] == 0


@pytest.mark.parametrize("mk", SD.MODELS)
@pytest.mark.parametrize("pid", SD.TEST_PROMPTS)
def test_rates_recover_planted_values_within_wilson_ci(dry, mk, pid):
    planted = dry["truth"]["planted"][mk][pid]
    s = dry["summ"][(mk, pid)]
    targets = {"hit_A": planted["A"][0], "FAR_B": planted["B"][0], "FAR_U": _planted_far_u(planted, dry["items"]),
               "FAR_U_line": planted["U_1"][0], "FAR_U_noline": planted["U_0"][0]}
    n = dry["numbers"]
    for key, value in targets.items():
        r = s[key]
        assert r["method"] == "wilson" and r["lo"] <= value <= r["hi"], (key, r, value)
        assert abs(r["estimate"] - value) < 0.05, (key, r["estimate"], value)
        base = f"{mk}.{pid}.{key}"
        assert n[base] == pytest.approx(r["estimate"]) and n[f"{base}.n"] == r["n"]
        assert (n[f"{base}.lo"], n[f"{base}.hi"]) == pytest.approx((r["lo"], r["hi"]))


# ----------------------------------------------------------------------------- exclusions
def test_exclusions_reduce_n_as_planted(dry):
    truth, excl, frame, n = dry["truth"], dry["exclusions"], dry["frame"], dry["numbers"]
    assert dict(zip(excl["item_id"], excl["reason"])) == truth["planted_exclusions"]
    by_arm = truth["exclusions_by_arm"]
    # A: 1 disagreement, 3 tracheostomy flags (2 read "absent"), 1 out of field; every A image is read
    assert by_arm == {"A": 5, "B": 5, "U": 2}
    assert n["excl.n_items"] == 12 and all(n[f"excl.{a}.n"] == k for a, k in by_arm.items()) and n["excl.DEV.n"] == 0
    ids = set(excl["item_id"])
    dropped_total = 0
    for mk in SD.MODELS:
        for pid in SD.TEST_PROMPTS:
            s = dry["summ"][(mk, pid)]
            full = M.summarise(frame, mk, pid, None, n_boot=0)
            assert s["n_rows_excluded"] == 12 and s["n_rows"] == full["n_rows"] - 12 == 348
            sub = frame.loc[(frame["model_key"] == mk) & (frame["prompt_id"] == pid)]
            gone = sub["item_id"].isin(ids)
            for arm, key in (("A", "hit_A"), ("B", "FAR_B"), ("U", "FAR_U")):
                ok = (sub["arm"] == arm) & sub["valid"]
                assert s[key]["n"] == int((ok & ~gone).sum())
                assert full[key]["n"] - s[key]["n"] == int((ok & gone).sum())
                dropped_total += int((ok & gone).sum())
    assert dropped_total > 0
    for mk, res in dry["analysis"]["arm1"].items():  # Arm 1 uses every retained item with z and s
        assert (res["n_A"], res["n_B"], res["n_U"]) == (120 - by_arm["A"], 120 - by_arm["B"], 120 - by_arm["U"])
    for r in dry["hyp"]["H3"]["primary"]:
        assert r["n_candidates"] == 240 - by_arm["B"] - by_arm["U"]


# ----------------------------------------------------------------------------- hypotheses
def test_h1_finds_the_planted_cooccurrence_difference(dry):
    fam = dry["hyp"]["H1"]["primary"]
    assert [r["model_key"] for r in fam] == list(SD.MODELS)
    for r in fam:
        p0 = dry["truth"]["planted"][r["model_key"]]["P0"]
        planted_diff = p0["U_1"][0] - p0["U_0"][0]
        assert r["contrast"] == "U line minus U noline" and r["substitute"] is False
        assert r["tested"] and r["family_m"] == 2 and r["method"] in ("bca", "percentile")
        assert r["p_holm"] < 0.05 and r["p_value"] <= r["p_holm"]
        assert 0 < r["lo"] <= planted_diff <= r["hi"]
        assert r["estimate"] == pytest.approx(r["rate_exposed"]["estimate"] - r["rate_reference"]["estimate"])
        assert dry["numbers"][f"h1.{r['model_key']}.p_holm"] == pytest.approx(r["p_holm"])
    assert dry["numbers"]["h1.contrast"] == "U line minus U noline"


def test_h3_finds_the_planted_abstention_increase(dry):
    frame, ids = dry["frame"], set(dry["exclusions"]["item_id"])
    items = dry["items"]
    for r in dry["hyp"]["H3"]["primary"]:
        mk = r["model_key"]
        pl = dry["truth"]["planted"][mk]
        sizes = {"B": int((items["arm"] == "B").sum()),
                 "U_1": int(((items["arm"] == "U") & items["stratum"].isin(["line", "ge2"])).sum())}
        sizes["U_0"] = int((items["arm"] == "U").sum()) - sizes["U_1"]
        increase = sum((pl["P2hi"][g][1] - pl["P1"][g][1]) * k for g, k in sizes.items()) / sum(sizes.values())
        assert r["p_method"] == "exact McNemar" and r["tested"] and r["p_holm"] < 0.05
        assert r["b"] > r["c"] and r["estimate"] == pytest.approx((r["b"] - r["c"]) / r["n"])
        assert r["lo"] <= increase <= r["hi"] and r["lo"] > 0
        assert r["abstention_pen"]["estimate"] > r["abstention_ref"]["estimate"]
        sub = frame.loc[(frame["model_key"] == mk) & frame["arm"].isin(["B", "U"]) & ~frame["item_id"].isin(ids)]
        valid = {p: set(sub.loc[(sub["prompt_id"] == p) & sub["valid"], "item_id"]) for p in ("P1", "P2hi")}
        cand = set(sub["item_id"])
        assert r["n"] == len(cand & valid["P1"] & valid["P2hi"]) and r["n_dropped"] == len(cand) - r["n"]
        assert dry["numbers"][f"h3.{mk}.b"] == r["b"] and dry["numbers"][f"h3.{mk}.n_dropped"] == r["n_dropped"]


def test_arm1_gain_with_informative_detector_and_none_with_noise(dry):
    fam = dry["hyp"]["H5"]["primary"]
    assert [r["model_key"] for r in fam] == list(SD.MODELS)
    for r in fam:
        assert r["tested"] and r["sided"] == "one-sided (greater)" and r["n_perm"] == N_SMALL
        assert r["estimate"] > 0.03 and r["lo"] > 0 and r["estimate"] > r["null_q975"]
        assert r["p_value"] <= 2 / (1 + N_SMALL) and r["p_holm"] < 0.05
        assert r["estimate"] == pytest.approx(r["ba_fused"] - r["ba_beta0"])
        rows = {x["row"]: x for x in dry["analysis"]["arm1"][r["model_key"]]["rows"]}
        assert rows["fused_beta1"]["p_value"] == r["p_value"]
        assert rows["fused_beta1"]["delta_ba"]["estimate"] == pytest.approx(r["estimate"])
    det = dry["hyp"]["detector"]["dev_auroc"]
    assert det["n"] == 300 and det["lo"] <= dry["truth"]["detector"]["expected_auroc"] <= det["hi"]
    noise = dry["noise_control"]
    assert set(noise) == set(SD.MODELS) and noise == _json(Path(dry["paths"]["noise_control"]))
    for mk, nc in noise.items():
        assert abs(nc["dev_auroc_s"] - 0.5) < 0.1
        assert abs(nc["gain"]) < 0.05, (mk, nc)
        assert nc["p_value"] > 0.1 and nc["null_q025"] <= nc["gain"] <= nc["null_q975"], (mk, nc)


def test_h0_gate_returns_the_planted_class(dry):
    expected = dry["truth"]["h0_expected"]
    assert expected == {"qwen3vl2b": "GO", "medgemma15": "CONDITIONAL"}  # both non-trivial branches exercised
    for mk, cls in expected.items():
        s = dry["summ"][(mk, "P0")]
        assert dry["hyp"]["H0"][mk]["class"] == cls == dry["numbers"][f"h0.{mk}.class"] == s["h0_class"]
        assert M.h0_gate(s["ba"]["estimate"], s["hit_A"]["estimate"], s["yes_rate"]["estimate"]) == cls
        assert "p_value" not in s["ba"]  # balanced accuracy is an interval, not a test
    t2 = _table(dry["results_dir"] / "table2.csv")
    prim = t2.loc[(t2["row"] == "primary") & (t2["prompt"] == "P0")]
    assert prim["H0"].tolist() == ["GO", "CONDITIONAL"]


# ----------------------------------------------------------------------------- numbers.json
def test_numbers_json_has_stable_keys(dry):
    n = dry["numbers"]
    assert all(AZ.NUMBER_KEY_RE.fullmatch(k) for k in n)
    for mk in SD.MODELS:
        for pid in SD.TEST_PROMPTS:
            for key in RATE_KEYS:
                base = f"{mk}.{pid}.{key}"
                assert all(base + sfx in n for sfx in ("", ".lo", ".hi", ".n", ".k")), base
                assert n[f"{base}.lo"] <= n[base] <= n[f"{base}.hi"]
            for key in ("ba", "ER_0.9", "ER_0.5"):
                base = f"{mk}.{pid}.{key}"
                assert all(base + sfx in n for sfx in ("", ".lo", ".hi", ".n")), base
            assert f"{mk}.{pid}.sens.hit_A.n" in n and f"{mk}.{pid}.n_rows_excluded" in n
        for key in ("auroc_AB", "auroc_AU"):
            assert all(f"{mk}.P0.{key}{sfx}" in n for sfx in ("", ".lo", ".hi", ".n"))
        for key in (f"h0.{mk}.class", f"h1.{mk}.estimate", f"h1.{mk}.lo", f"h1.{mk}.hi", f"h1.{mk}.p_value",
                    f"h1.{mk}.p_holm", f"h3.{mk}.estimate", f"h3.{mk}.p_value", f"h3.{mk}.p_holm", f"h3.{mk}.c",
                    f"h5.{mk}.estimate", f"h5.{mk}.p_value", f"h5.{mk}.p_holm", f"h5.{mk}.n_perm",
                    f"arm1.{mk}.fused_beta1.delta", f"arm1.{mk}.fused_beta1.delta.lo", f"arm1.{mk}.beta0.ba.n",
                    f"arm1.{mk}.agreement_greedy_beta0", f"arm1.{mk}.matched_bias.fused_minus_matched",
                    f"blind.{mk}.P0.text.p_yes", f"blind.{mk}.P2hi.grey.greedy", f"run.{mk}.revision",
                    f"run.{mk}.s_per_item", f"smoke.{mk}.coverage_median", f"smoke.{mk}.n"):
            assert key in n, key
    assert n["qwen3vl2b.P0.hit_A"] == pytest.approx(dry["summ"][("qwen3vl2b", "P0")]["hit_A"]["estimate"])
    assert n["h5.qwen3vl2b.n_perm"] == N_SMALL and n["smoke.qwen3vl2b.n"] == 20
    assert (n["pairs.n"], n["pairs.rung"], n["u.line.n"], n["u.noline.n"]) == (120, "standard", 60, 60)
    assert n["u.h1_substitute"] is False and n["match.spotcheck_correct"] == 59 and n["yield.n_images"] == 30083
    assert n["verify.n_read"] == 360 and n["verify.kappa"] < 1 and n["detector.dev_auroc.n"] == 300
    assert n["verify.agreement.lo"] <= n["verify.agreement"] <= n["verify.agreement.hi"]


@pytest.mark.skipif(not (REPORT_DIR / "sections").is_dir(), reason="report sections not present")
def test_report_placeholders_resolve_from_numbers_json(dry):
    n = dry["numbers"]
    wanted: dict[str, set] = {}
    for p in sorted(REPORT_DIR.rglob("*.md")):
        for m in PLACEHOLDER_RE.finditer(p.read_text(encoding="utf-8")):
            wanted.setdefault(m.group(1), set()).add(m.group(2))
    missing = sorted(k for k in wanted if k not in n and k not in EXTERNAL_NUMBER_KEYS)
    assert not missing, f"report placeholders the analysis does not produce: {missing}"
    no_ci = sorted(k for k, styles in wanted.items() if "ci" in styles and k in n and not {f"{k}.lo", f"{k}.hi"} <= set(n))
    assert not no_ci


# ----------------------------------------------------------------------------- missing inputs
def test_analyze_cli_without_verification_logs_the_skip(dry, tmp_path, capsys):
    p = dry["paths"]
    out = tmp_path / "results"
    argv = ["--items", str(p["items"]), "--parsed", str(p["parsed"]), "--detector", str(p["detector"]),
            "--exclusions", str(p["exclusions"]), "--match-stats", str(p["match_stats"]), "--yield", str(p["yield"]),
            "--selection", str(p["selection"]), "--verification", str(tmp_path / "verification_summary.json"),
            "--out", str(out), "--n-boot", str(N_TINY), "--n-perm", str(N_TINY), "--spotcheck-correct", "58",
            "--number", "parse.handcheck_agree=97", "--number", "run.medsiglip.dtype=float32", "--number", "pairs.n=5"]
    assert AZ.main(argv) == 0
    printed = capsys.readouterr().out
    skip = "[skip] verification summary: verification_summary.json not found; Table 1 reading rows and kappa say not available"
    log = (out / "analysis_log.txt").read_text(encoding="utf-8").splitlines()
    assert skip in printed and skip in log
    hyp, n = _json(out / "hypotheses.json"), _json(out / "numbers.json")
    assert hyp["meta"]["log_skips"] == [skip] and "verification" not in hyp["meta"]["inputs"]
    assert hyp["meta"]["exclusions_applied"] is True
    for name in AZ.OUTPUT_FILES:
        assert (out / name).is_file(), name
    t1 = _table(out / "table1.csv")
    look = dict(zip(t1["quantity"], t1["value"]))
    assert look["Verification"] == "not available (reading not done yet)"
    assert not any(q.startswith("Retest Cohen's kappa") for q in t1["quantity"])
    assert look["Visual spot check correct"] == "58 of 60"
    assert not any(k.startswith("verify.") for k in n)
    assert n["match.spotcheck_correct"] == 58 and n["parse.handcheck_agree"] == 97 and n["run.medsiglip.dtype"] == "float32"
    assert n["pairs.n"] == 120  # a computed key is never overwritten by --number
    assert any(line.startswith("[note] extra number pairs.n ignored") for line in log)


def test_analyze_without_detector_or_exclusions_skips_arm1_and_removes_stale_outputs(dry, tmp_path):
    p = dry["paths"]
    out = tmp_path / "results"
    out.mkdir()
    for name in ("table3.csv", "table3.md", "table3.tex"):
        (out / name).write_text("left by an earlier run\n", encoding="utf-8")
    inputs = AZ.AnalysisInputs(items=p["items"], parsed=p["parsed"], detector=None, exclusions=tmp_path / "exclusions.csv")
    r = AZ.run_analysis(inputs, out, N_TINY, N_TINY, log=None)
    log = r["log"]
    assert "[skip] detector scores (S9): no path given; Arm 1, H5, Table 3 and the verifier rows of Table 1 are skipped" in log
    assert ("[skip] exclusions (S12): exclusions.csv not found; no item is excluded (the blind reading is pending); "
            "results are provisional") in log
    assert "[skip] Table 3: Arm 1 not run (no detector scores)" in log
    for name in ("table3.csv", "table3.md", "table3.tex"):
        assert not (out / name).exists()
        assert f"[stale] removed {name}: left by an earlier run and not produced by this one" in log
    hyp = _json(out / "hypotheses.json")
    assert hyp["H5"] == {"skipped": "no detector scores"} and hyp["meta"]["exclusions_applied"] is False
    assert len(hyp["meta"]["log_skips"]) == 10
    assert not any(k.startswith(("arm1.", "h5.", "detector.", "excl.")) for k in r["numbers"])
    s = {(x["model_key"], x["prompt_id"]): x for x in r["summaries"] if not x["unparsable_as_wrong"]}
    c = dry["truth"]["counts"]["qwen3vl2b"]["P0"]["A"]
    assert s[("qwen3vl2b", "P0")]["n_rows_excluded"] == 0 and s[("qwen3vl2b", "P0")]["hit_A"]["n"] == c["n_valid"]
    t1 = _table(out / "table1.csv")
    assert (t1["value"].str.startswith("not available")).sum() == 5  # matching, yield, selection, reading, detector


# ----------------------------------------------------------------------------- registered variants
def test_variants_h1_substitute_and_two_pairs_per_patient(tmp_path):
    r = SD.run_dryrun(tmp_path, 49, 49, h1_substitute=True, n_two_pair_patients=20, log=None)
    n, h = r["analysis"]["numbers"], r["analysis"]["hypotheses"]
    assert n["h1.contrast"] == "U ge2 minus U eq1" and n["u.h1_substitute"] is True
    assert n["pairs.rung"] == "two_per_patient" and n["u.ge2.n"] == n["u.eq1.n"] == 60
    items = read_items(r["paths"]["items"])
    assert items.loc[items["arm"] == "A", "nih_patient_id"].nunique() == 100
    for res in h["H1"]["primary"]:
        assert res["substitute"] is True and res["tested"] and res["p_holm"] < 0.05
    for res in h["H3"]["primary"]:
        assert res["p_method"].startswith("clustered BCa CI inversion") and res["ci_role"] == "primary"
        assert res["tested"] and res["p_holm"] < 0.05 and res["n_clusters"] < res["n"]
        assert res["mcnemar_check"]["p_value"] < 0.05 and res["mcnemar_check"]["n"] < res["n"]
    t1 = _table(r["results_dir"] / "table1.csv")
    strata = [q for q in t1["quantity"] if q.startswith("U stratum")]
    assert strata == ["U stratum >=2 lines", "U stratum 1 line"]  # exposed stratum first


# ----------------------------------------------------------------------------- integration gaps (e, f, lead items)
def _strict_json(path: Path):
    def refuse(token):
        raise ValueError(f"{path.name}: non-standard JSON constant {token}")
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=refuse)


def test_json_outputs_are_strict_json_with_null_for_undefined(dry):
    res = dry["results_dir"]
    for name in ("numbers.json", "hypotheses.json", "metrics.json", "verification_summary.json"):
        _strict_json(res / name)  # no NaN / Infinity anywhere
    n = _strict_json(res / "numbers.json")
    assert any(v is None for v in n.values())  # undefined numbers are kept as null, not dropped
    hyp = _strict_json(res / "hypotheses.json")
    assert hyp["H0"] and hyp["H1"]["primary"]


def test_numbers_for_dimension_key_gpu_and_detector_dtype(dry):
    n = dry["numbers"]
    assert n["match.dimkey_share_found"] == 0.985 and n["match.dimkey_used"] is True
    assert n["run.gpu_name"] == "NVIDIA RTX A5000 (synthetic)"
    assert n["run.medsiglip.dtype"] == "float32"
    log = (dry["results_dir"] / "analysis_log.txt").read_text(encoding="utf-8")
    assert "[input] run.gpu_name" in log


def test_dryrun_reads_every_a_and_counts_flagged_trach_not_as_disagreement(dry):
    v = dry["verification"]
    items = dry["items"]
    n_abu = int(items["arm"].isin(["A", "B", "U"]).sum())
    assert v["n_read"] == n_abu and v["by_arm"]["A"]["n_read"] == int((items["arm"] == "A").sum())
    planted = dry["truth"]["planted_exclusions"]
    trach = sorted(i for i, r in planted.items() if r == "flag_tracheostomy")
    assert len(trach) == 3
    assert v["exclusions"]["by_reason"]["flag_tracheostomy"] == 3
    assert v["n_disagreements"] == sum(1 for r in planted.values() if r == "label_disagreement") == 4
    assert v["n_flagged_answered_contrary_to_label"] >= 2  # the two A images read "absent" with the flag
    assert v["retest"]["gap_ok"] is True and v["warnings"] == []
