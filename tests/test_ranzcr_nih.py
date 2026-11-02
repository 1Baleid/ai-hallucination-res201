"""Tests for cxrdev.ranzcr (label presence, yield table) and cxrdev.nih (metadata, discovery).

All frames are synthetic; only the RANZCR column names are real (from cxrdev.constants).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cxrdev.constants import CVC_COLS, ETT_COLS, NGT_COLS, NGT_INCOMPLETE_COL, RANZCR_PATIENT, RANZCR_UID, SWAN_COL
from cxrdev.nih import (
    NIH_META_COLUMNS,
    find_inputs,
    find_nih_image_folders,
    find_nih_meta_csv,
    image_folders,
    load_nih_meta,
    locate_all_images,
    locate_images,
    nih_patient_from_filename,
)
from cxrdev.ranzcr import PRESENCE_COLUMNS, YIELD_METRICS, is_train_csv, presence, read_train_csv, yield_table

TRAIN_COLUMNS = [RANZCR_UID, *ETT_COLS, *NGT_COLS, *CVC_COLS, SWAN_COL, RANZCR_PATIENT]  # real RANZCR order


def _train(rows: list[dict]) -> pd.DataFrame:
    """Synthetic train.csv rows: each dict gives uid, patient and the columns set to 1."""
    out = []
    for r in rows:
        rec = {c: 0 for c in TRAIN_COLUMNS}
        rec[RANZCR_UID] = r["uid"]
        rec[RANZCR_PATIENT] = r["pid"]
        for c in r.get("on", ()):
            rec[c] = 1
        out.append(rec)
    return pd.DataFrame(out, columns=TRAIN_COLUMNS)


# --------------------------------------------------------------------------- ranzcr.presence
def test_presence_any_of_rules() -> None:
    df = _train([
        {"uid": "u1", "pid": "p1", "on": [ETT_COLS[0]]},                               # ETT abnormal
        {"uid": "u2", "pid": "p1", "on": [ETT_COLS[2], NGT_INCOMPLETE_COL]},           # ETT normal + NGT incompl.
        {"uid": "u3", "pid": "p2", "on": [CVC_COLS[1], SWAN_COL, NGT_COLS[3]]},        # 3 other lines
        {"uid": "u4", "pid": "p2", "on": []},                                          # nothing
        {"uid": "u5", "pid": "p3", "on": [ETT_COLS[1], CVC_COLS[0], CVC_COLS[2]]},     # CVC counted once
    ])
    p = presence(df)
    assert tuple(p.columns) == PRESENCE_COLUMNS
    rec = p.set_index("uid")
    assert rec.loc["u1", ["ett", "ngt", "cvc", "swan", "n_other_lines"]].tolist() == [1, 0, 0, 0, 0]
    assert rec.loc["u2", ["ett", "ngt", "ngt_incomplete", "n_other_lines"]].tolist() == [1, 1, 1, 1]
    assert rec.loc["u3", ["ett", "ngt", "cvc", "swan", "n_other_lines"]].tolist() == [0, 1, 1, 1, 3]
    assert rec.loc["u4", ["ett", "n_other_lines"]].tolist() == [0, 0]
    assert rec.loc["u5", ["ett", "cvc", "n_other_lines"]].tolist() == [1, 1, 1]
    for c in ("ett", "ngt", "ngt_incomplete", "cvc", "swan", "n_other_lines"):
        assert p[c].dtype == np.int64
    assert rec.loc["u3", "ranzcr_patient_id"] == "p2"


def test_presence_tolerates_blank_and_text_values() -> None:
    df = _train([{"uid": "u1", "pid": 7}, {"uid": "u2", "pid": 8}]).astype(object)
    df.loc[0, ETT_COLS[0]] = "1"
    df.loc[0, SWAN_COL] = np.nan
    df.loc[1, CVC_COLS[0]] = ""
    df.loc[1, NGT_COLS[0]] = 1.0
    p = presence(df).set_index("uid")
    assert p.loc["u1", "ett"] == 1 and p.loc["u1", "swan"] == 0
    assert p.loc["u2", "cvc"] == 0 and p.loc["u2", "ngt"] == 1
    assert p.loc["u1", "ranzcr_patient_id"] == "7"


def test_presence_validation() -> None:
    df = _train([{"uid": "u1", "pid": "p1"}, {"uid": "u1", "pid": "p1"}])
    with pytest.raises(ValueError, match="duplicated"):
        presence(df)
    with pytest.raises(ValueError, match="lacks columns"):
        presence(df.drop(columns=[SWAN_COL]).iloc[:1])
    with pytest.raises(ValueError, match="empty"):
        presence(_train([{"uid": " ", "pid": "p1"}]))


# --------------------------------------------------------------------------- ranzcr.yield_table
def test_yield_table_counts() -> None:
    ngt, cvc = NGT_COLS[3], CVC_COLS[2]
    rows = [
        # p1: both states; absent image with 1 line
        {"uid": "a1", "pid": "p1", "on": [ETT_COLS[2]]},
        {"uid": "a2", "pid": "p1", "on": [cvc]},
        # p2: both states, 2 present + 2 absent (one noline, one with 2 lines)
        {"uid": "b1", "pid": "p2", "on": [ETT_COLS[0]]},
        {"uid": "b2", "pid": "p2", "on": [ETT_COLS[1], ngt]},
        {"uid": "b3", "pid": "p2", "on": []},
        {"uid": "b4", "pid": "p2", "on": [ngt, cvc]},
        # p3: absent only, noline
        {"uid": "c1", "pid": "p3", "on": []},
        # p4: absent only, NGT incompletely imaged (counts as 1 line)
        {"uid": "d1", "pid": "p4", "on": [NGT_INCOMPLETE_COL]},
        # p5: present only
        {"uid": "e1", "pid": "p5", "on": [ETT_COLS[2], SWAN_COL]},
    ]
    y = yield_table(presence(_train(rows)))
    assert list(y.columns) == ["metric", "value"]
    assert y["metric"].tolist()[: len(YIELD_METRICS)] == list(YIELD_METRICS)
    assert y["metric"].is_unique and y["value"].dtype == np.int64
    v = dict(zip(y["metric"], y["value"]))
    expected = {
        "n_images": 9, "n_patients": 5, "ett_present_images": 4, "ett_absent_images": 5,
        "patients_with_both_ett_states": 2, "ett_absent_line_images": 3, "ett_absent_line_patients": 3,
        "ett_absent_noline_images": 2, "ett_absent_noline_patients": 2, "ett_absent_ge2_patients": 1,
        "ett_absent_eq1_patients": 2, "ngt_incomplete_images": 1, "ett_present_patients": 3,
        "ett_absent_patients": 4, "patients_ge2_present_ge2_absent": 1,
        "ett_absent_line_patients_not_both_states": 1, "ett_absent_noline_patients_not_both_states": 1,
        "ngt_present_images": 3, "cvc_present_images": 2, "swan_present_images": 1,
    }
    for k, val in expected.items():
        assert v[k] == val, k


def test_yield_table_empty_and_validation() -> None:
    empty = presence(_train([]))
    y = dict(zip(*yield_table(empty).T.values))
    assert y["n_images"] == 0 and y["patients_with_both_ett_states"] == 0
    with pytest.raises(ValueError):
        yield_table(pd.DataFrame({"uid": ["a"]}))


def test_train_csv_helpers(tmp_path) -> None:
    df = _train([{"uid": "1.2.3", "pid": "0012"}])
    path = tmp_path / "train.csv"
    df.to_csv(path, index=False)
    assert is_train_csv(path)
    back = read_train_csv(path)
    assert back.loc[0, RANZCR_PATIENT] == "0012" and back.loc[0, RANZCR_UID] == "1.2.3"
    other = tmp_path / "other.csv"
    pd.DataFrame({"a": [1]}).to_csv(other, index=False)
    assert not is_train_csv(other) and not is_train_csv(tmp_path / "missing.csv")


# --------------------------------------------------------------------------- nih.load_nih_meta
KAGGLE_HEADER = ("Image Index,Finding Labels,Follow-up #,Patient ID,Patient Age,Patient Gender,View Position,"
                 "OriginalImage[Width,Height],OriginalImagePixelSpacing[x,y],")
V2020_HEADER = ("Image Index,Finding Labels,Follow-up #,Patient ID,Patient Age,Patient Gender,View Position,"
                "OriginalImage[Width,Height],OriginalImagePixelSpacing[x,y]")
ROWS = [
    ("00000001_000.png", "Cardiomegaly", "0", "1", "58", "M", "PA", "2682", "2749", "0.143", "0.143"),
    ("00000001_001.png", "Cardiomegaly|Emphysema", "1", "1", "58", "M", "AP", "2894", "2729", "0.143", "0.143"),
    ("00000002_000.png", "No Finding", "0", "2", "81", "F", "PA", "2500", "2048", "0.168", "0.168"),
]


def _write(path, header: str, rows, trailing: bool, bom: bool = False) -> None:
    lines = [header] + [",".join(r) + ("," if trailing else "") for r in rows]
    text = "\n".join(lines) + "\n"
    path.write_bytes((b"\xef\xbb\xbf" if bom else b"") + text.encode("utf-8"))


def _check_meta(df: pd.DataFrame) -> None:
    assert tuple(df.columns) == NIH_META_COLUMNS
    assert df["nih_filename"].tolist() == [r[0] for r in ROWS]
    assert df["nih_patient_id"].tolist() == [1, 1, 2]
    assert df["followup"].tolist() == [0, 1, 0]
    assert df["age"].tolist() == [58, 58, 81]
    assert df["sex"].tolist() == ["M", "M", "F"]
    assert df["view"].tolist() == ["PA", "AP", "PA"]
    assert df["orig_w"].tolist() == [2682, 2894, 2500]
    assert df["orig_h"].tolist() == [2749, 2729, 2048]
    assert df["finding_labels"].tolist() == ["Cardiomegaly", "Cardiomegaly|Emphysema", "No Finding"]
    for c in ("nih_patient_id", "followup", "age", "orig_w", "orig_h"):
        assert df[c].dtype == np.int64


def test_load_nih_meta_kaggle_split_header_with_trailing_column(tmp_path) -> None:
    path = tmp_path / "Data_Entry_2017.csv"
    _write(path, KAGGLE_HEADER, ROWS, trailing=True)
    raw = pd.read_csv(path)
    assert "OriginalImage[Width" in raw.columns and "Height]" in raw.columns and "y]" in raw.columns
    _check_meta(load_nih_meta(path))


def test_load_nih_meta_v2020(tmp_path) -> None:
    path = tmp_path / "Data_Entry_2017_v2020.csv"
    _write(path, V2020_HEADER, ROWS, trailing=False, bom=True)
    _check_meta(load_nih_meta(path))


def test_load_nih_meta_header_variants(tmp_path) -> None:
    spaced = V2020_HEADER.replace("[Width,Height]", "[Width, Height]").replace("[x,y]", "[x, y]")
    p1 = tmp_path / "spaced.csv"
    _write(p1, spaced, ROWS, trailing=True)
    _check_meta(load_nih_meta(p1))
    quoted = V2020_HEADER.replace("OriginalImage[Width,Height]", '"OriginalImage[Width,Height]"').replace(
        "OriginalImagePixelSpacing[x,y]", '"OriginalImagePixelSpacing[x,y]"')
    p2 = tmp_path / "quoted.csv"
    _write(p2, quoted, ROWS, trailing=False)
    _check_meta(load_nih_meta(p2))
    old_age = [r[:4] + ("0" + r[4] + "Y",) + r[5:] for r in ROWS]
    p3 = tmp_path / "old_age.csv"
    _write(p3, KAGGLE_HEADER, old_age, trailing=True)
    _check_meta(load_nih_meta(p3))


def test_load_nih_meta_errors(tmp_path) -> None:
    p = tmp_path / "bad.csv"
    _write(p, KAGGLE_HEADER.replace("View Position,", ""), [r[:6] + r[7:] for r in ROWS], trailing=True)
    with pytest.raises(ValueError, match="lacks"):
        load_nih_meta(p)
    _write(p, KAGGLE_HEADER, ROWS + [ROWS[0]], trailing=True)
    with pytest.raises(ValueError, match="duplicated"):
        load_nih_meta(p)
    _write(p, KAGGLE_HEADER, [("x.png",) + ROWS[0][1:]], trailing=True)
    with pytest.raises(ValueError, match="unexpected image names"):
        load_nih_meta(p)
    _write(p, KAGGLE_HEADER, [ROWS[0][:7] + ("wide",) + ROWS[0][8:]], trailing=True)
    with pytest.raises(ValueError, match="not an integer"):
        load_nih_meta(p)
    p.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_nih_meta(p)


def test_nih_patient_from_filename() -> None:
    assert nih_patient_from_filename("00000013_005.png") == 13
    assert nih_patient_from_filename("x.png") == -1


# --------------------------------------------------------------------------- nih discovery
def _touch(path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def test_image_folders_and_locate(tmp_path) -> None:
    root = tmp_path / "data"
    _touch(root / "images_001" / "images" / "00000001_000.png")
    _touch(root / "images_001" / "images" / "00000001_001.png")
    _touch(root / "images_002" / "images" / "00000002_000.png")
    _touch(root / "images_003" / "00000003_000.png")                       # flat images_* layout
    _touch(root / "images_004" / "images" / "00000001_000.png")            # duplicate name: first wins
    _touch(root / "images_005" / "notes.txt")                               # no PNG: ignored
    folders = image_folders(root)
    assert [n for n, _ in folders] == ["images_001", "images_002", "images_003", "images_004"]
    loc = locate_images(root)
    assert set(loc) == {"00000001_000.png", "00000001_001.png", "00000002_000.png", "00000003_000.png"}
    assert loc["00000001_000.png"].parent.parent.name == "images_001"
    flat = tmp_path / "box"
    _touch(flat / "images" / "00000009_000.png")
    assert [n for n, _ in image_folders(flat)] == ["images"]
    bare = tmp_path / "bare"
    _touch(bare / "00000010_000.png")
    assert image_folders(bare)[0][0] == "bare"
    assert image_folders(tmp_path / "missing") == []


def test_find_inputs_prunes_image_dirs(tmp_path) -> None:
    inp = tmp_path / "input"
    _touch(inp / "competitions" / "ranzcr" / "train.csv")
    _touch(inp / "competitions" / "ranzcr" / "train" / "nested.csv")          # inside a pruned dir
    _touch(inp / "datasets" / "nih" / "data" / "Data_Entry_2017.csv")
    _touch(inp / "datasets" / "nih" / "data" / "Data_Entry_2017_v2020.csv")
    _touch(inp / "datasets" / "nih" / "data" / "images_001" / "images" / "00000001_000.png")
    _touch(inp / "datasets" / "nih" / "data" / "images_001" / "images" / "deep.csv")
    _touch(inp / "res201-hash" / "nih_hashes_images_001.csv")
    csvs = find_inputs(inp, "*.csv")
    names = [p.name for p in csvs]
    assert "nested.csv" not in names and "deep.csv" not in names
    assert {"train.csv", "Data_Entry_2017.csv", "nih_hashes_images_001.csv"} <= set(names)
    assert find_inputs(inp, "nih_hashes_*.csv")[0].parent.name == "res201-hash"
    assert [p.name for p in find_inputs(inp, "*.csv", max_depth=1)] == ["nih_hashes_images_001.csv"]
    assert find_inputs(inp, "*.csv", max_depth=0) == []
    assert [p.name for p in find_inputs(inp, "images_*", kind="dir")] == ["images_001"]
    assert find_nih_meta_csv(inp).name == "Data_Entry_2017.csv"
    assert find_nih_meta_csv(tmp_path / "nothing") is None
    with pytest.raises(ValueError):
        find_inputs(inp, "*", kind="link")


def test_find_nih_image_folders_and_locate_all(tmp_path) -> None:
    inp = tmp_path / "input"
    _touch(inp / "data" / "Data_Entry_2017.csv")
    _touch(inp / "data" / "images_001" / "images" / "00000001_000.png")
    _touch(inp / "data" / "images_002" / "images" / "00000002_000.png")
    folders = find_nih_image_folders(inp)
    assert [n for n, _ in folders] == ["images_001", "images_002"]
    assert set(locate_all_images(inp)) == {"00000001_000.png", "00000002_000.png"}
    # no metadata CSV next to the images: fall back to images_* directories anywhere
    inp2 = tmp_path / "input2"
    _touch(inp2 / "mount" / "x" / "images_007" / "images" / "00000007_000.png")
    assert [n for n, _ in find_nih_image_folders(inp2)] == ["images_007"]
