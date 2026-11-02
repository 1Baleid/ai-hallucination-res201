"""Tests for the gaps closed between modules during integration (synthetic data only).

* parse_runs: a run directory's smoke-test and rotated JSONL files never replace the main run's records;
* verify_join: skip and flag exclusions precede the label check (an A image read "absent" with the
  tracheostomy flag is a flag_tracheostomy exclusion, never a disagreement), the 48-h retest clock is
  checked from the meta files and recorded as a warning, and the summary is strict JSON (null, not NaN);
* pod/run_medsiglip.py writes the S9 ``dtype`` column and scripts/analyze.py turns it into
  ``run.medsiglip.dtype``; analyze reads ``run.gpu_name`` from the pod's run settings files.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cxrdev import constants as C
from cxrdev import fileio
from cxrdev.items import ITEM_COLS
from cxrdev.prompts import prompt_sha256, prompt_text
from pod import common
from scripts import analyze as AZ
from scripts import parse_runs as PR
from scripts import verify_join as VJ


# ============================================================================= parse_runs
def _record(item: str, raw: str, run_id: str = "core-1", signal: str = "first_token") -> dict:
    text = prompt_text("P0")
    return common.make_record({
        "run_id": run_id, "model_key": "qwen3vl2b", "model_id": C.MODELS["qwen3vl2b"]["hf_id"], "revision": "a" * 40,
        "transformers_version": "5.22.1", "torch_version": "2.9", "dtype": "bfloat16", "device": "cuda:0",
        "item_id": item, "nih_filename": "00000001_000.png", "image_sha256": "c" * 64, "condition": "image",
        "prompt_id": "P0", "prompt_sha256": prompt_sha256(text), "prompt_text": text, "visual_tokens": 1024,
        "seed": 1, "max_new_tokens": 16, "raw_output": raw, "n_generated_tokens": 2, "ended_with_eos": True,
        "parsed": "yes", "parser_version": "2", "logp_yes": -0.1, "logp_no": -3.0, "logp_abstain": float("nan"),
        "z": 2.9, "coverage": 0.95, "signal": signal, "yes_ids": [1], "no_ids": [2], "wall_s": 0.3,
        "timestamp_utc": "2026-11-02T10:00:00Z",
    })


def test_parse_runs_skips_smoke_and_rotated_files(tmp_path: Path) -> None:
    run = tmp_path / "core-1"
    fileio.append_jsonl(run / "qwen3vl2b.jsonl", _record("a000000001", "No."))
    # the smoke run repeats the same dev item under the same key and would be read last
    fileio.append_jsonl(run / "smoke" / "qwen3vl2b.smoke.jsonl", _record("a000000001", "Yes.", "core-1-smoke"))
    fileio.append_jsonl(run / "smoke" / "qwen3vl2b.smoke.prev.jsonl", _record("a000000001", "Yes.", "core-1-smoke"))
    out = tmp_path / "parsed.csv"
    s = PR.parse_runs([run], out, tmp_path / "summary.json")
    df = pd.read_csv(out, dtype=str, keep_default_na=False)
    assert len(df) == 1 and df.loc[0, "run_id"] == "core-1" and df.loc[0, "parsed"] == "no"
    assert len(s["skipped_files"]) == 2 and s["n_duplicates_dropped"] == 0
    assert json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))["skipped_files"] == s["skipped_files"]
    both = PR.parse_runs([run], tmp_path / "all.csv", include_smoke=True)
    assert both["n_duplicates_dropped"] == 2 and both["skipped_files"] == []
    explicit = PR.parse_runs([run / "smoke" / "qwen3vl2b.smoke.jsonl"], tmp_path / "smoke.csv")
    assert explicit["n_rows"] == 1  # a file named on the command line is always read


# ============================================================================= verify_join
def _items() -> pd.DataFrame:
    rows = []
    specs = [("a%02d" % i, "A", 1) for i in range(6)] + [("b%02d" % i, "B", 0) for i in range(6)] + \
            [("u%02d" % i, "U", 0) for i in range(6)]
    for k, (iid, arm, ett) in enumerate(specs):
        item_id = f"{k:02d}{iid}aaaaa"[:10].replace("_", "a")
        rows.append({
            "item_id": item_id, "uid": f"1.2.3.{k}", "nih_filename": f"{k + 1:08d}_000.png", "nih_patient_id": k + 1,
            "arm": arm, "pair_id": "", "stratum": "line" if arm == "U" else "", "ett": ett, "ngt": 0,
            "ngt_incomplete": 0, "cvc": 1, "swan": 0, "n_other_lines": 1, "view": "AP", "followup": 0, "age": 50,
            "sex": "M", "direction": "", "gap": -1, "rung": "standard", "smoke": 0,
        })
    return pd.DataFrame(rows, columns=list(ITEM_COLS))


def _reading(items: pd.DataFrame, plan: dict[int, tuple[str, tuple[str, ...]]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Key and S11 reading: the label's answer unless ``plan`` gives (response, flags) for a row."""
    read_ids = [f"{0xA0000000 + k:08X}" for k in range(len(items))]
    key = pd.DataFrame({"read_id": read_ids, "item_id": items["item_id"]})
    rows = []
    for k, r in enumerate(items.itertuples(index=False)):
        response, flags = plan.get(k, ("present" if r.ett == 1 else "absent", ()))
        rows.append({"read_id": read_ids[k], "nih_filename": r.nih_filename, "response": response,
                     **{f: int(f.removeprefix("flag_") in flags) for f in VJ.FLAG_REASONS}, "note": "",
                     "seconds": 5.0, "order_index": k, "timestamp_utc": f"2026-11-10T09:{k:02d}:00Z"})
    return key, pd.DataFrame(rows, columns=list(VJ.VERIFIED_COLUMNS))


PLAN = {
    0: ("absent", ("tracheostomy",)),    # A: only a tracheostomy tube, read "absent" -> flag, not disagreement
    1: ("present", ("tracheostomy",)),   # A: tracheostomy flagged although read "present"
    2: ("absent", ()),                    # A: a genuine disagreement
    6: ("present", ("ambiguous",)),      # B: flagged and contrary to the label -> flag only
    7: ("present", ()),                   # B: a genuine disagreement
    12: ("skip", ("out_of_field",)),     # U: skip wins over the flag
    13: ("absent", ("out_of_field",)),   # U
}


def test_flags_and_skips_precede_the_label_check() -> None:
    items = _items()
    key, verified = _reading(items, PLAN)
    excl, s, dis = VJ.join_verification(items, key, verified)
    reason = dict(zip(excl["item_id"], excl["reason"]))
    ids = items["item_id"].tolist()
    assert reason == {ids[0]: "flag_tracheostomy", ids[1]: "flag_tracheostomy", ids[2]: "label_disagreement",
                      ids[6]: "flag_ambiguous", ids[7]: "label_disagreement", ids[12]: "skip",
                      ids[13]: "flag_out_of_field"}
    assert sorted(dis["item_id"]) == sorted([ids[2], ids[7]])  # flagged images never reach the re-match list
    assert s["n_disagreements"] == 2 and s["exclusions"]["by_reason"]["label_disagreement"] == 2
    assert s["by_arm"]["A"]["n_disagreements"] == 1 and s["by_arm"]["A"]["by_reason"]["flag_tracheostomy"] == 2
    # agreement is over answered, unflagged images: 18 read - 1 skip - 4 flagged = 13, of which 11 agree
    assert (s["agreement"]["k"], s["agreement"]["n"]) == (11, 13)
    assert (s["agreement_all_read"]["k"], s["agreement_all_read"]["n"]) == (13, 17)  # 2 of 4 flagged agree
    assert s["n_flagged_answered_contrary_to_label"] == 2


def _write_reading(tmp_path: Path, items: pd.DataFrame, saved_read: str | None, retest_hours: float | None):
    key, verified = _reading(items, PLAN)
    paths = {"items": tmp_path / "items.csv", "key": tmp_path / "reading_key.csv",
             "verified": tmp_path / "verified_items.csv", "retest": tmp_path / "retest.csv"}
    fileio.write_csv(items, paths["items"])
    fileio.write_csv(key, paths["key"])
    sha = fileio.write_csv(verified, paths["verified"])
    if saved_read is not None:
        (tmp_path / "verified_items.meta.json").write_text(json.dumps(
            {"sha256": sha, "saved_utc": saved_read, "seed": 1, "n": len(verified), "mode": "read",
             "tool_version": "1.0.0"}), encoding="utf-8")
    rt = verified.iloc[[3, 4, 5, 8, 9]].copy()
    start = pd.Timestamp(saved_read or "2026-11-10T10:00:00Z") + pd.Timedelta(hours=retest_hours or 0)
    rt["timestamp_utc"] = [(start + pd.Timedelta(minutes=i)).strftime("%Y-%m-%dT%H:%M:%SZ") for i in range(len(rt))]
    rt["order_index"] = range(len(rt))
    sha_rt = fileio.write_csv(rt, paths["retest"])
    (tmp_path / "retest.meta.json").write_text(json.dumps(
        {"sha256": sha_rt, "saved_utc": rt["timestamp_utc"].iloc[-1], "seed": 2, "n": len(rt), "mode": "retest",
         "tool_version": "1.0.0"}), encoding="utf-8")
    return paths


@pytest.mark.parametrize(("hours", "ok"), [(49.0, True), (47.5, False)])
def test_retest_clock_is_checked_from_the_meta_files(tmp_path: Path, hours: float, ok: bool) -> None:
    paths = _write_reading(tmp_path, _items(), "2026-11-10T10:00:00Z", hours)
    s = VJ.run(paths["items"], paths["key"], paths["verified"], tmp_path / "exclusions.csv",
               tmp_path / "summary.json", retest_path=paths["retest"])
    r = s["retest"]
    assert r["gap_hours_after_read"] == pytest.approx(hours) and r["gap_ok"] is ok
    assert r["read_saved_utc"] == "2026-11-10T10:00:00Z" and r["min_hours"] == 48.0
    assert (len(s["warnings"]) == 0) is ok
    if not ok:
        assert "earlier than the registered 48 h" in s["warnings"][0]
    text = (tmp_path / "summary.json").read_text(encoding="utf-8")
    json.loads(text, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))  # strict JSON
    assert "NaN" not in text


def test_retest_clock_without_read_meta_is_a_warning_not_an_error(tmp_path: Path) -> None:
    paths = _write_reading(tmp_path, _items(), None, 60.0)
    s = VJ.run(paths["items"], paths["key"], paths["verified"], tmp_path / "exclusions.csv",
               tmp_path / "summary.json", retest_path=paths["retest"])
    assert s["retest"]["gap_ok"] is None and s["retest"]["gap_hours_after_read"] is None
    assert any("could not be checked" in w for w in s["warnings"])
    assert any("no meta file" in w for w in s["warnings"])
    assert (tmp_path / "exclusions.csv").is_file()


def test_undefined_kappa_is_written_as_null(tmp_path: Path) -> None:
    items = _items()
    items = items[items["arm"] != "A"].reset_index(drop=True)  # every label is "absent": kappa undefined
    key, verified = _reading(items, {})
    for name, df in (("items.csv", items), ("reading_key.csv", key), ("verified_items.csv", verified)):
        fileio.write_csv(df, tmp_path / name)
    s = VJ.run(tmp_path / "items.csv", tmp_path / "reading_key.csv", tmp_path / "verified_items.csv",
               tmp_path / "exclusions.csv", tmp_path / "summary.json")
    assert s["kappa_vs_ranzcr"] is None
    assert json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))["kappa_vs_ranzcr"] is None


# ============================================================================= S9 dtype and run numbers
def test_s9_dtype_column_reaches_numbers_json(tmp_path: Path) -> None:
    from pod import run_medsiglip as RM
    assert common.S9_COLUMNS[-1] == "dtype"

    class Proc:
        def __call__(self, text=None, images=None, padding=None, return_tensors=None):
            import torch
            return {"pixel_values": torch.zeros((len(images), 3, 2, 2))}

    class Model:
        def __call__(self, pixel_values=None, **kw):
            import torch
            from types import SimpleNamespace
            n = pixel_values.shape[0]
            return SimpleNamespace(logits_per_image=torch.tensor([[1.0, -1.0]] * n))

    from PIL import Image
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    Image.new("L", (64, 64), 100).save(img_dir / "00000001_000.png")
    items = [common.PodItem("abcdef0123", "00000001_000.png", "full", 0)]
    scorer = RM.MedSiglipScorer(Model(), Proc(), model_id=C.EXPERT["hf_id"], revision="d" * 40, device="cpu",
                                dtype_name="bfloat16")
    rows = RM.score_items(scorer, items, img_dir, log=lambda s: None)
    assert rows[0]["dtype"] == "bfloat16" and list(rows[0]) == list(common.S9_COLUMNS)
    det = pd.DataFrame(rows)
    numbers: dict = {}
    parsed = pd.DataFrame({"condition": ["image"], "model_key": ["qwen3vl2b"], "transformers_version": ["5"],
                           "torch_version": ["2"], "device": ["cuda:0"], "revision": ["r"], "model_id": ["m"],
                           "dtype": ["bfloat16"], "signal": ["first_token"], "wall_s": [0.2], "visual_tokens": [1024]})
    AZ.run_numbers(numbers, parsed, det)
    assert numbers["run.medsiglip.dtype"] == "bfloat16" and numbers["run.medsiglip.revision"] == "d" * 40
    old = det.drop(columns="dtype")  # an S9 file written before the column existed still works
    numbers2: dict = {}
    AZ.run_numbers(numbers2, parsed, old)
    assert "run.medsiglip.dtype" not in numbers2


def test_gpu_names_from_run_settings(tmp_path: Path) -> None:
    run = tmp_path / "core-1"
    common.append_session_meta(run / "qwen3vl2b.jsonl", {"mode": "items", "backend": {"gpu_name": "NVIDIA RTX A5000"}})
    common.update_smoke_report(run / "smoke_report.json", "medgemma15", {"gpu_name": "NVIDIA RTX A5000"})
    assert AZ.gpu_names(run) == ["NVIDIA RTX A5000"]
    common.append_session_meta(run / "medgemma15.jsonl", {"mode": "items", "backend": {"gpu_name": None}})
    assert AZ.gpu_names(run) == ["NVIDIA RTX A5000"]
    assert AZ.gpu_names(tmp_path / "missing") == []


def test_input_numbers_dimension_key() -> None:
    numbers: dict = {}
    ms = {"status_counts": {"matched": 9}, "match_rate": 0.9, "config": {"use_dimension_key": False},
          "run": {"dimkey_share_found": 0.31}, "ambiguous_share_within_patient": float("nan")}
    AZ.input_numbers(numbers, ms, None, None, None, None, None, None)
    assert numbers["match.dimkey_share_found"] == 0.31 and numbers["match.dimkey_used"] is False
    assert "match.ambiguous_within_patient_share" in numbers and numbers["match.ambiguous_within_patient_share"] is None
    numbers2: dict = {}
    AZ.input_numbers(numbers2, {"run": {"config_used": {"use_dimension_key": True}}}, None, None, None, None, None, None)
    assert numbers2["match.dimkey_used"] is True and "match.dimkey_share_found" not in numbers2
