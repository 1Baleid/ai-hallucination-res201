"""End-to-end test on REAL public NIH ChestX-ray14 images with a simulated RANZCR side (slow).

Runs ``scripts/e2e_real_nih.py``: the bundled hash, match and export kernels on a fake Kaggle input tree
built from the local NIH PNGs, item selection, the pod scripts with fake model backends, parsing, a
simulated blind reading with a 49-hour retest, verification, the fast analysis and the report build.
Skipped cleanly when ``data/nih_box`` (or the v2020 metadata) is not on this machine. No RANZCR CLiP
file is used: the competition side is synthetic.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from cxrdev import constants as C
from scripts import e2e_real_nih as E2E

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not E2E.data_available(), reason="NIH images (data/nih_box) or metadata (data/nih) not present"),
]


@pytest.fixture(scope="module")
def e2e():
    cfg = E2E.E2EConfig(log=lambda s: None)
    return {"cfg": cfg, "s": E2E.run_e2e(cfg)}


def test_bundled_kernels_match_every_present_source_and_no_absent_one(e2e):
    m = e2e["s"]["matching"]
    assert m["n_jpegs_source_in_nih"] >= 550 and m["n_jpegs_source_absent"] == 30
    assert m["n_wrong"] == 0, m["status_of_absent"]
    assert m["n_correct"] == m["n_jpegs_source_in_nih"] and m["match_accuracy"] == 1.0, m["status_of_present"]
    assert m["n_absent_unmatched"] == 30, m["status_of_absent"]
    assert m["d1_true"]["max"] <= m["cut"] < m["d1_absent"]["min"]
    assert m["ncc_true"]["min"] >= 0.95
    # the absent-source JPEGs may have sizes no NIH-part image has: they fall back to the global key
    assert m["dimkey_share_found"] >= 0.95 and m["key_used_counts"]["dim"] >= m["n_jpegs_source_in_nih"]
    assert m["margin_true"] is None or m["margin_true"]["min"] >= 16 or m["route_counts"]["ncc_tiebreak"] > 0
    assert m["seconds"]["hash_per_png"] > 0 and m["extrapolated_kaggle_hours"]["hash_kernel"] > 0


def test_selection_invariants_substitute_and_probe_exclusion(e2e):
    sel = e2e["s"]["selection"]  # check_selection raised inside the run if any S5/S6/S10 invariant failed
    assert sel["rung"] == "standard" and sel["by_arm"]["A"] == sel["by_arm"]["B"] == e2e["cfg"].n_pairs
    assert sel["h1_substitute"] is True and sel["h1_contrast"] == {"exposed": "ge2", "reference": "eq1"}
    assert set(sel["u_strata"]) <= {"ge2", "eq1", "noline"} and {"ge2", "eq1"} <= set(sel["u_strata"])
    assert sel["n_reading_list"] == sel["by_arm"]["A"] + sel["by_arm"]["B"] + sel["by_arm"]["U"]
    assert sel["n_practice"] == 10 and sel["n_smoke"] == e2e["cfg"].n_smoke
    assert sel["excluded_filenames"]["n_listed"] == 6 and sel["excluded_filenames"]["n_present_in_matched_pool"] == 5
    ex = e2e["s"]["export"]
    assert ex["n_exported"] == sel["n_items"] and ex["sha_mismatch"] == 0 and ex["missing"] == []


def test_pod_scripts_with_fake_backends_and_parse(e2e):
    s = e2e["s"]
    sel = s["selection"]
    n_full = sel["by_arm"]["A"] + sel["by_arm"]["B"] + sel["by_arm"]["U"]
    expected = 3 * n_full + sel["by_arm"]["DEV"] + 6
    assert s["pod"]["n_records"] == {mk: expected for mk in C.CORE_MODELS}
    assert all(v["decision"] == "first_token" for v in s["pod"]["smoke"].values())
    assert s["pod"]["smoke"]["qwen3vl2b"]["visual_tokens"] == 1024 and s["pod"]["smoke"]["medgemma15"]["visual_tokens"] == 256
    assert s["parse"]["n_rows"] == 2 * expected and s["parse"]["n_duplicates_dropped"] == 0
    assert sorted(s["parse"]["skipped_files"]) == sorted(f"{mk}.smoke.jsonl" for mk in C.CORE_MODELS)


def test_reading_tracheostomy_exclusions_and_retest_clock(e2e):
    v = e2e["s"]["verification"]
    assert v["planted_matches"] is True
    assert v["trach_a_items_reason"] == ["flag_tracheostomy"] and v["trach_a_in_disagreements"] == 0
    assert v["exclusions_by_reason"]["flag_tracheostomy"] == 3 and v["n_disagreements"] == 2
    assert v["retest_gap_hours"] == pytest.approx(49.0, abs=0.2) and v["retest_gap_ok"] is True and v["warnings"] == []
    r = e2e["s"]["reading"]
    assert r["n_read"] == e2e["s"]["selection"]["n_reading_list"] and r["n_practice"] == 10


def test_analysis_outputs_numbers_and_report(e2e):
    s, root = e2e["s"], e2e["cfg"].root
    files = set(s["analysis"]["files"])
    for stem in ("table1", "table2", "table3"):
        assert {f"{stem}.csv", f"{stem}.md", f"{stem}.tex"} <= files
    assert {"figure1.png", "figure1.pdf", "hypotheses.json", "numbers.json", "verification_summary.json"} <= files

    def strict(name):
        return json.loads((root / "results" / name).read_text(encoding="utf-8"),
                          parse_constant=lambda c: (_ for _ in ()).throw(ValueError(f"{name}: {c}")))

    numbers, hyp = strict("numbers.json"), strict("hypotheses.json")
    strict("verification_summary.json")
    assert numbers["u.h1_substitute"] is True and numbers["run.medsiglip.dtype"] == "float32"
    assert numbers["match.dimkey_used"] is True and numbers["match.dimkey_share_found"] >= 0.95
    assert numbers["match.spotcheck_correct"] == s["matching"]["n_spotcheck_pairs"] == 60
    assert numbers["select.excluded_filenames.n_listed"] == 6 and numbers["select.excluded_filenames.n_present"] == 5
    assert numbers["excl.flag_tracheostomy.n"] == 3 and numbers["verify.n_disagreements"] == 2
    assert numbers["verify.n_read"] == s["selection"]["n_reading_list"]
    for r in hyp["H1"]["primary"]:
        assert r["contrast"] == "U ge2 minus U eq1" and r["substitute"] is True
        assert r["contrast_source"] == "selection report"
        assert r["n_exposed"] + r["n_reference"] <= 2 * e2e["cfg"].n_u_per_stratum
    assert {r["model_key"] for r in hyp["H3"]["primary"]} == set(C.CORE_MODELS)
    assert all(r["tested"] for r in hyp["H5"]["primary"])
    assert (root / "report" / "report.md").is_file() and s["report"]["n_sections"] > 0


def test_no_identifier_leaks_into_results_or_report(e2e):
    assert e2e["s"]["leaks"] == {"results": {}, "report": {}}
    exclusions = pd.read_csv(e2e["cfg"].root / "items" / "exclusions.csv", dtype=str)
    assert set(exclusions["reason"]) <= {"flag_ambiguous", "flag_tracheostomy", "flag_out_of_field",
                                         "label_disagreement", "skip"}
