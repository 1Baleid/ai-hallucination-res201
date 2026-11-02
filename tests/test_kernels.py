"""Tests for the Kaggle side: bundler, the three bundled kernels on a fake /kaggle/input tree, and the
push/poll/download and file-list dataset helpers (no network: the Kaggle CLI is replaced by fakes).

The fake tree is synthetic: random smooth "NIH" PNGs, "RANZCR" JPEGs made from some of them, a fake
train.csv that uses the real RANZCR column names, and a fake Data_Entry_2017.csv in the Kaggle
split-header form. Each bundled kernel.py runs in a separate interpreter in isolated mode (-I), so
it cannot import the repository: the bundle must be self-contained.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from scipy.ndimage import gaussian_filter

from cxrdev.constants import (
    CVC_COLS, ETT_COLS, FILELIST_DATASET_SLUG, KAGGLE_COMPETITION, KAGGLE_NIH_DATASET, KERNEL_SLUGS, NGT_COLS,
    NGT_INCOMPLETE_COL, RANZCR_PATIENT, RANZCR_UID, SWAN_COL,
)
from cxrdev.imghash import bits_to_hex, load_gray, phash_bits
from cxrdev.matching import MATCH_COLUMNS
from kaggle_kernels import build_kernels as bk
from kaggle_kernels import make_filelist_dataset as mfd
from kaggle_kernels import run_kernel as rk

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_COLUMNS = [RANZCR_UID, *ETT_COLS, *NGT_COLS, *CVC_COLS, SWAN_COL, RANZCR_PATIENT]
SIZES = [(300, 280), (320, 300), (280, 320)]
KAGGLE_HEADER = ("Image Index,Finding Labels,Follow-up #,Patient ID,Patient Age,Patient Gender,View Position,"
                 "OriginalImage[Width,Height],OriginalImagePixelSpacing[x,y],")


# --------------------------------------------------------------------------- fake input tree
def _smooth(rng: np.random.Generator, side: int = 128) -> np.ndarray:
    field = gaussian_filter(rng.normal(size=(side, side)), sigma=6)
    field = (field - field.min()) / (field.max() - field.min() + 1e-9)
    yy, xx = np.mgrid[0:side, 0:side] / side
    img = 30 + 170 * field + 30 * rng.normal() * (xx - 0.5) + 30 * rng.normal() * (yy - 0.5)
    return np.clip(np.round(img), 0, 255).astype(np.uint8)


def _save_jpeg(arr: np.ndarray, size: tuple[int, int], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).resize(size, Image.Resampling.BICUBIC).save(path, format="JPEG", quality=95)


def _ranzcr_hex(path: Path) -> str:
    """The pHash the match kernel computes for a JPEG (draft decode at >= 128 px)."""
    with Image.open(path) as im:
        im.draft("L", (128, 128))
        im.load()
        gray = np.array(im.convert("L"))
    return bits_to_hex(phash_bits(gray))


@pytest.fixture(scope="module")
def tree(tmp_path_factory) -> dict:
    rng = np.random.default_rng(20261102)
    root = tmp_path_factory.mktemp("kaggle")
    inp = root / "input"
    nih_root = inp / "datasets" / "nih-chest-xrays" / "data"
    comp = inp / "competitions" / KAGGLE_COMPETITION
    films = {  # name -> (folder, size index)
        "00000001_000.png": ("images_001", 0), "00000001_001.png": ("images_001", 1),
        "00000002_000.png": ("images_001", 0), "00000003_000.png": ("images_001", 2),
        "00000004_000.png": ("images_002", 1), "00000005_000.png": ("images_002", 2),
        "00000006_000.png": ("images_002", 0), "00000007_000.png": ("images_002", 1),
    }
    pixels = {}
    meta_lines = [KAGGLE_HEADER]
    for name, (folder, si) in films.items():
        px = _smooth(rng)
        pixels[name] = px
        path = nih_root / folder / "images" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(px).save(path)
        pid, follow = int(name[:8]), int(name[9:12])
        w, h = SIZES[si]
        meta_lines.append(f"{name},No Finding,{follow},{pid},{40 + pid},{'MF'[pid % 2]},{'PA' if pid % 3 else 'AP'},"
                          f"{w},{h},0.143,0.143,")
    (nih_root / "images_002" / "images" / "00000099_000.png").write_bytes(b"not a png")   # unreadable
    (nih_root / "Data_Entry_2017.csv").write_text("\n".join(meta_lines) + "\n", encoding="utf-8")
    (nih_root / "BBox_List_2017.csv").write_text("Image Index,Finding Label\n", encoding="utf-8")

    # RANZCR copies: uid -> (source NIH name or "", jpeg size, patient, label columns set to 1)
    plan = {
        "1.2.3.test.0001": ("00000001_000.png", SIZES[0], "RP1", [ETT_COLS[2]]),
        "1.2.3.test.0002": ("00000001_001.png", SIZES[1], "RP1", [CVC_COLS[2]]),
        "1.2.3.test.0003": ("00000002_000.png", SIZES[0], "RP2", []),
        "1.2.3.test.0004": ("00000004_000.png", SIZES[1], "RP4", [NGT_INCOMPLETE_COL]),
        "1.2.3.test.0005": ("", SIZES[0], "RP9", [ETT_COLS[1], SWAN_COL]),             # source absent
        "1.2.3.test.0006": ("00000005_000.png", (333, 301), "RP5", [CVC_COLS[0], NGT_COLS[3]]),  # off-key size
    }
    rows = []
    for uid, (src, size, rp, on) in plan.items():
        px = pixels[src] if src else _smooth(rng)
        _save_jpeg(px, size, comp / "train" / f"{uid}.jpg")
        rec = {c: 0 for c in TRAIN_COLUMNS}
        rec.update({RANZCR_UID: uid, RANZCR_PATIENT: rp})
        for c in on:
            rec[c] = 1
        rows.append(rec)
    _save_jpeg(_smooth(rng), SIZES[2], comp / "train" / "1.2.3.test.9999.jpg")   # JPEG without a label row
    _save_jpeg(_smooth(rng), SIZES[2], comp / "test" / "1.2.3.test.8888.jpg")    # test split: never used
    pd.DataFrame(rows, columns=TRAIN_COLUMNS).to_csv(comp / "train.csv", index=False)
    pd.DataFrame({"StudyInstanceUID": ["x"], "ETT - Abnormal": [0]}).to_csv(comp / "sample_submission.csv",
                                                                           index=False)
    truth = {uid: src for uid, (src, *_rest) in plan.items()}
    ranzcr_hex = {uid: _ranzcr_hex(comp / "train" / f"{uid}.jpg") for uid in plan}
    build = root / "build"
    bk.build_all("testuser", build)
    return {"root": root, "input": inp, "nih_root": nih_root, "comp": comp, "truth": truth, "build": build,
            "ranzcr_hex": ranzcr_hex, "pixels": pixels}


def _run_bundle(build: Path, slug: str, input_root: Path, working: Path, temp: Path, *extra: str):
    cmd = [sys.executable, "-I", str(build / slug / "kernel.py"), "--input-root", str(input_root),
           "--working", str(working), "--temp", str(temp), "--workers", "1", *extra]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=str(working.parent))


@pytest.fixture(scope="module")
def hash_run(tree) -> dict:
    working = tree["root"] / "hash_working"
    working.mkdir()
    proc = _run_bundle(tree["build"], "res201-hash", tree["input"], working, tree["root"] / "hash_temp")
    return {"proc": proc, "working": working}


@pytest.fixture(scope="module")
def match_run(tree, hash_run) -> dict:
    mount = tree["input"] / "notebooks" / "testuser" / "res201-hash"     # the hash output as a kernel source
    mount.mkdir(parents=True)
    for p in hash_run["working"].iterdir():
        shutil.copy2(p, mount / p.name)
    working = tree["root"] / "match_working"
    working.mkdir()
    temp = tree["root"] / "match_temp"
    proc = _run_bundle(tree["build"], "res201-match", tree["input"], working, temp)
    return {"proc": proc, "working": working, "temp": temp}


@pytest.fixture(scope="module")
def export_run(tree) -> dict:
    ds = tree["input"] / "datasets" / "testuser" / FILELIST_DATASET_SLUG
    ds.mkdir(parents=True)
    names = ["00000004_000.png", "00000001_000.png", "00000002_000.png", "00000077_000.png"]  # last one missing
    (ds / "nih_filelist.csv").write_text("nih_filename\n" + "\n".join(names) + "\n", encoding="utf-8")
    working = tree["root"] / "export_working"
    working.mkdir()
    temp = tree["root"] / "export_temp"
    proc = _run_bundle(tree["build"], "res201-export", tree["input"], working, temp)
    return {"proc": proc, "working": working, "temp": temp, "names": names}


# --------------------------------------------------------------------------- bundler
def test_kernel_metadata_fields() -> None:
    for kind in ("hash", "match", "export"):
        m = bk.kernel_metadata(kind, "someone")
        assert m["id"] == f"someone/{KERNEL_SLUGS[kind]}"
        assert m["code_file"] == "kernel.py" and m["language"] == "python" and m["kernel_type"] == "script"
        assert m["is_private"] is True and m["enable_gpu"] is False and m["enable_internet"] is False
        assert m["enable_tpu"] is False and m["model_sources"] == []
        assert len(m["title"]) >= 5
        assert m["title"].lower().replace(" ", "-") == KERNEL_SLUGS[kind]
    assert bk.kernel_metadata("hash", "u")["competition_sources"] == [KAGGLE_COMPETITION]
    assert bk.kernel_metadata("match", "u")["competition_sources"] == [KAGGLE_COMPETITION]
    assert bk.kernel_metadata("export", "u")["competition_sources"] == []
    assert bk.kernel_metadata("hash", "u")["dataset_sources"] == [KAGGLE_NIH_DATASET]
    assert bk.kernel_metadata("match", "u")["dataset_sources"] == [KAGGLE_NIH_DATASET]
    assert bk.kernel_metadata("export", "u")["dataset_sources"] == [KAGGLE_NIH_DATASET, "u/res201-nih-filelist"]
    assert bk.kernel_metadata("match", "u")["kernel_sources"] == ["u/res201-hash"]
    assert bk.kernel_metadata("hash", "u")["kernel_sources"] == []
    assert bk.kernel_metadata("export", "u")["kernel_sources"] == []
    for bad in ("", "a/b", "has space", "-lead"):
        with pytest.raises(ValueError):
            bk.kernel_metadata("hash", bad)


def test_titles_resolve_to_slugs() -> None:
    slugify = pytest.importorskip("slugify").slugify   # python-slugify, the function the Kaggle CLI uses
    for kind, title in bk.TITLES.items():
        assert slugify(title) == KERNEL_SLUGS[kind]


def test_build_outputs_and_determinism(tree, tmp_path) -> None:
    again = bk.build_all("testuser", tmp_path / "b2")
    for kind, folder in again.items():
        slug = KERNEL_SLUGS[kind]
        assert sorted(p.name for p in folder.iterdir()) == ["kernel-metadata.json", "kernel.py"]
        assert (folder / "kernel.py").read_bytes() == (tree["build"] / slug / "kernel.py").read_bytes()
        meta = json.loads((folder / "kernel-metadata.json").read_text(encoding="utf-8"))
        assert meta == bk.kernel_metadata(kind, "testuser")
        text = (folder / "kernel.py").read_text(encoding="utf-8")
        assert "from cxrdev" not in text and "import cxrdev" not in text and "BUNDLE-STRIP" not in text
        assert text.count("from __future__ import annotations") == 1
        assert bk.check_python_compat(text, (3, 10)) == []
        assert "phash" not in folder.name and "mapping" not in folder.name


def test_module_order() -> None:
    assert bk.module_order("hash") == ["constants", "imghash", "nih", "ranzcr"]
    match = bk.module_order("match")
    assert set(match) == {"constants", "imghash", "matching", "nih", "ranzcr"}
    assert match.index("imghash") < match.index("matching")
    assert match.index("constants") < match.index("ranzcr")
    assert bk.module_order("export") == ["nih"]


def test_strip_source_and_rejections() -> None:
    src = ("from __future__ import annotations\nimport os\nfrom cxrdev.imghash import ncc, thumb\n"
           "# BUNDLE-STRIP-START\nimport sys\n# BUNDLE-STRIP-END\nX = 1\n")
    out, deps = bk.strip_source(src, "x.py")
    assert deps == ["imghash"] and out == "import os\nX = 1\n"
    for bad in ("import cxrdev\n", "from cxrdev.imghash import ncc as n\n", "from cxrdev import imghash\n",
                "from cxrdev.imghash import (\n    ncc,\n)\n", "def f():\n    from cxrdev.imghash import ncc\n",
                "# BUNDLE-STRIP-START\nx = 1\n", "# BUNDLE-STRIP-END\n"):
        with pytest.raises(bk.BundleError):
            bk.strip_source(bad, "bad.py")


def test_collision_and_compat_checks() -> None:
    with pytest.raises(bk.BundleError, match="defined in"):
        bk.check_collisions([("a.py", "def f():\n    pass\n"), ("b.py", "f = 2\n")])
    with pytest.raises(bk.BundleError, match="alias"):
        bk.check_collisions([("a.py", "import numpy as np\n"), ("b.py", "import pandas as np\n")])
    with pytest.raises(bk.BundleError):
        bk.check_collisions([("a.py", "def np():\n    pass\n"), ("b.py", "import numpy as np\n")])
    bk.check_collisions([("a.py", "import numpy as np\ntry:\n    from x import y as _z\nexcept ImportError:\n"
                                  "    from w import y as _z\n"), ("b.py", "import numpy as np\n__all__ = []\n"),
                         ("c.py", "__all__ = []\n")])
    assert bk.check_python_compat("type X = int\n")
    assert bk.check_python_compat("except_star = 1\ntry:\n    pass\nexcept* ValueError:\n    pass\n")
    if sys.version_info >= (3, 12):
        assert bk.check_python_compat('d = {}\nx = f"{d["a"]}"\n')
        assert bk.check_python_compat("x = f\"{'\\n'.join([])}\"\n")
    assert bk.check_python_compat("d = {}\nx = f\"{d['a']}\"\n") == []
    assert bk.check_python_compat('x = f"""{"a"}"""\n') == []


def test_overrides(tmp_path) -> None:
    src = bk.bundle_kernel("hash", hash_folders=["images_005", "images_002"])
    assert "FOLDERS_OVERRIDE = ['images_002', 'images_005']" in src
    src = bk.bundle_kernel("match", match_config={"cut": 30, "ncc_min": 0.9})
    assert "CONFIG_OVERRIDES = {'cut': 30, 'ncc_min': 0.9}" in src
    with pytest.raises(bk.BundleError, match="unknown MatchConfig"):
        bk.bundle_kernel("match", match_config={"cutoff": 3})
    with pytest.raises(bk.BundleError):
        bk.bundle_kernel("hash", hash_folders=["../etc"])
    with pytest.raises(ValueError):
        bk.bundle_kernel("other")
    assert bk.main(["--username", "bad/user", "--out", str(tmp_path / "x")]) == 2
    assert bk.main(["--username", "okuser", "--out", str(tmp_path / "y"), "--kernels", "export"]) == 0
    assert (tmp_path / "y" / "res201-export" / "kernel.py").is_file()


# --------------------------------------------------------------------------- hash kernel
def test_hash_kernel_outputs(tree, hash_run) -> None:
    proc, working = hash_run["proc"], hash_run["working"]
    assert proc.returncode == 0, proc.stdout + proc.stderr
    names = sorted(p.name for p in working.iterdir())
    assert names == ["dimkey_test.json", "nih_hashes_images_001.csv", "nih_hashes_images_002.csv",
                     "run_summary.json", "yield_table.csv"]
    shard = pd.read_csv(working / "nih_hashes_images_001.csv", dtype=str)
    assert list(shard.columns) == ["nih_filename", "folder", "ph", "dh"]
    assert sorted(shard["nih_filename"]) == ["00000001_000.png", "00000001_001.png", "00000002_000.png",
                                             "00000003_000.png"]
    for row in shard.itertuples():
        png = tree["nih_root"] / "images_001" / "images" / row.nih_filename
        assert row.ph == bits_to_hex(phash_bits(load_gray(png)))
        assert len(row.ph) == 64 and len(row.dh) == 64 and row.folder == "images_001"
    shard2 = pd.read_csv(working / "nih_hashes_images_002.csv", dtype=str)
    assert "00000099_000.png" not in set(shard2["nih_filename"]) and len(shard2) == 4
    summary = json.loads((working / "run_summary.json").read_text(encoding="utf-8"))
    assert all(v["status"] == "done" for v in summary["steps"].values())
    assert summary["folders"]["images_001"]["status"] == "done"
    assert summary["folders"]["images_002"]["n_errors"] == 1
    assert summary["n_hashed_total"] == 8 and summary["finished_utc"]
    y = dict(pd.read_csv(working / "yield_table.csv").itertuples(index=False, name=None))
    assert y["n_images"] == 6 and y["n_patients"] == 5
    assert y["ett_present_images"] == 2 and y["ett_absent_images"] == 4
    assert y["patients_with_both_ett_states"] == 1
    assert y["ett_absent_line_images"] == 3 and y["ett_absent_noline_images"] == 1
    assert y["ett_absent_ge2_patients"] == 1 and y["ett_absent_eq1_patients"] == 2
    assert y["ngt_incomplete_images"] == 1
    dk = json.loads((working / "dimkey_test.json").read_text(encoding="utf-8"))
    assert dk["n_jpegs_total"] == 7 and dk["n_sampled"] == 7 and dk["n_read"] == 7
    assert dk["n_found"] == 6 and dk["share_found"] == pytest.approx(6 / 7)   # (333, 301) is not an NIH size
    assert {d["jpg_w"] for d in dk["per_size"]} == {300, 320, 280, 333}
    assert "1.2.3.test" not in json.dumps(dk)


def test_hash_kernel_selected_folders_and_empty_input(tree, tmp_path) -> None:
    working = tmp_path / "w1"
    proc = _run_bundle(tree["build"], "res201-hash", tree["input"], working, tmp_path / "t1", "--folders",
                       "images_002,images_404")
    assert proc.returncode == 0, proc.stderr
    assert sorted(p.name for p in working.glob("nih_hashes_*.csv")) == ["nih_hashes_images_002.csv"]
    summary = json.loads((working / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["folders"]["images_001"]["status"] == "not_selected"
    assert summary["folders"]["images_404"]["status"] == "not_found"
    empty = tmp_path / "empty_input"
    empty.mkdir()
    working2 = tmp_path / "w2"
    proc = _run_bundle(tree["build"], "res201-hash", empty, working2, tmp_path / "t2")
    assert proc.returncode == 0
    s2 = json.loads((working2 / "run_summary.json").read_text(encoding="utf-8"))
    assert s2["steps"]["yield_table"]["status"] == "failed" and s2["steps"]["dimkey_test"]["status"] == "failed"
    assert s2["folders"] == {}


def test_hash_kernel_per_folder_try_except(tree, tmp_path, monkeypatch) -> None:
    path = REPO_ROOT / "kaggle_kernels" / "hash_kernel" / "kernel_main.py"
    spec = importlib.util.spec_from_file_location("res201_hash_kernel_main", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    real = mod.hash_folder

    def flaky(name, folder, workers):
        if name == "images_001":
            raise OSError("simulated read failure")
        return real(name, folder, workers)

    monkeypatch.setattr(mod, "hash_folder", flaky)
    summary = mod.main(str(tree["input"]), str(tmp_path / "w"), str(tmp_path / "t"), workers=1)
    assert summary["folders"]["images_001"]["status"] == "failed"
    assert "simulated read failure" in summary["folders"]["images_001"]["error"]
    assert summary["folders"]["images_002"]["status"] == "done"
    assert (tmp_path / "w" / "nih_hashes_images_002.csv").is_file()
    assert not (tmp_path / "w" / "nih_hashes_images_001.csv").exists()
    monkeypatch.setattr(mod, "hash_folder", real)
    assert mod.main_cli(["--input-root", str(tree["input"]), "--working", str(tmp_path / "w"),
                         "--temp", str(tmp_path / "t"), "--workers", "1"]) == 0   # resumes: skips done folders
    s = json.loads((tmp_path / "w" / "run_summary.json").read_text(encoding="utf-8"))
    assert s["folders"]["images_002"]["status"] == "skipped_existing"
    assert s["folders"]["images_001"]["status"] == "done"


# --------------------------------------------------------------------------- match kernel
def test_match_kernel_outputs(tree, match_run) -> None:
    proc, working = match_run["proc"], match_run["working"]
    assert proc.returncode == 0, proc.stdout + proc.stderr
    names = sorted(p.name for p in working.iterdir())
    assert names == ["calibration.png", "match_stats.json", "match_table.csv", "spotcheck_01.png"]
    table = pd.read_csv(working / "match_table.csv", dtype={"nih_filename": str, "uid": str},
                        keep_default_na=False)
    assert tuple(table.columns) == MATCH_COLUMNS
    assert sorted(table["uid"]) == sorted(tree["truth"])          # unlabeled and test JPEGs are not matched
    t = table.set_index("uid")
    for uid, src in tree["truth"].items():
        if src:
            assert t.loc[uid, "status"] == "matched", t.loc[uid].to_dict()
            assert t.loc[uid, "nih_filename"] == src
            assert t.loc[uid, "nih_patient_id"] == int(src[:8])
        else:
            assert t.loc[uid, "status"] == "unmatched"
    # 6 of the 7 sampled JPEG sizes are NIH original sizes (0.86 >= 0.5): the key stays on, and the
    # JPEG whose size has no NIH bucket falls back to global matching
    assert t.loc["1.2.3.test.0006", "key_used"] == "global"
    assert (t.drop(index="1.2.3.test.0006")["key_used"] == "dim").all()
    assert (t.loc[t["status"] == "matched", "ncc1"].astype(float) >= 0.95).all()
    stats = json.loads((working / "match_stats.json").read_text(encoding="utf-8"))
    assert stats["status_counts"]["matched"] == 5 and stats["status_counts"]["unmatched"] == 1
    assert stats["run"]["n_ranzcr_hashed"] == 6 and stats["run"]["n_jpegs"] == 7
    assert stats["run"]["n_spotcheck_pairs"] == 5 and stats["run"]["config_used"]["use_dimension_key"] is True
    assert "share_found" in stats["run"]["config_reason"]
    with Image.open(working / "spotcheck_01.png") as im:
        assert im.size == (2 * 256 + 8, 5 * (256 + 16))
    with Image.open(working / "calibration.png") as im:
        assert im.format == "PNG"


def test_match_kernel_writes_no_ranzcr_hash(tree, match_run) -> None:
    working = match_run["working"]
    blobs = {p.name: p.read_bytes() for p in working.iterdir()}
    for uid, hx in tree["ranzcr_hex"].items():
        for name, data in blobs.items():
            assert hx.encode() not in data and hx.upper().encode() not in data, (uid, name)
    assert not any("hash" in n or n.endswith(".npy") for n in blobs)
    assert not match_run["temp"].exists() or not any(match_run["temp"].rglob("*hash*"))


def test_match_kernel_dimension_key_off(tree, match_run, tmp_path) -> None:
    # a dimension-key report below the threshold switches the key off: every image is matched globally
    inp = tmp_path / "input"
    shutil.copytree(tree["input"], inp)
    dk = next(inp.rglob("dimkey_test.json"))
    report = json.loads(dk.read_text(encoding="utf-8"))
    report["share_found"] = 0.3
    dk.write_text(json.dumps(report), encoding="utf-8")
    working = tmp_path / "w"
    working.mkdir()
    proc = _run_bundle(tree["build"], "res201-match", inp, working, tmp_path / "t")
    assert proc.returncode == 0, proc.stderr
    table = pd.read_csv(working / "match_table.csv", keep_default_na=False).set_index("uid")
    assert (table["key_used"] == "global").all() and (table["n_candidates"] == 8).all()
    assert (table["status"] == "matched").sum() == 5
    stats = json.loads((working / "match_stats.json").read_text(encoding="utf-8"))
    assert stats["run"]["config_used"]["use_dimension_key"] is False


def test_match_kernel_config_override_runs(tree, match_run, tmp_path) -> None:
    build = tmp_path / "build"
    bk.build_all("testuser", build, kinds=("match",), match_config={"cut": 10, "margin_bits": 20})
    working = tmp_path / "w"
    working.mkdir()
    proc = _run_bundle(build, "res201-match", tree["input"], working, tmp_path / "t")
    assert proc.returncode == 0, proc.stderr
    stats = json.loads((working / "match_stats.json").read_text(encoding="utf-8"))
    assert stats["cut"] == 10 and stats["cut_rule"] == "fixed" and stats["config"]["margin_bits"] == 20
    assert "override cut = 10" in stats["run"]["config_reason"]
    assert stats["status_counts"]["matched"] == 5


def test_match_kernel_fails_without_hashes(tree, tmp_path) -> None:
    inp = tmp_path / "input"
    shutil.copytree(tree["input"], inp, ignore=shutil.ignore_patterns("notebooks"))
    working = tmp_path / "w"
    working.mkdir()
    proc = _run_bundle(tree["build"], "res201-match", inp, working, tmp_path / "t")
    assert proc.returncode != 0 and "nih_hashes" in proc.stderr
    assert not (working / "match_table.csv").exists()


# --------------------------------------------------------------------------- export kernel
def test_export_kernel_outputs(tree, export_run) -> None:
    proc, working = export_run["proc"], export_run["working"]
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert sorted(p.name for p in working.iterdir()) == ["manifest.csv", "selected_nih_png.zip"]
    man = pd.read_csv(working / "manifest.csv", dtype={"nih_filename": str, "sha256": str})
    assert list(man.columns) == ["nih_filename", "sha256", "bytes"]
    assert man["nih_filename"].tolist() == ["00000001_000.png", "00000002_000.png", "00000004_000.png"]
    with zipfile.ZipFile(working / "selected_nih_png.zip") as zf:
        assert zf.namelist() == man["nih_filename"].tolist()
        for row in man.itertuples():
            data = zf.read(row.nih_filename)
            folder = "images_002" if row.nih_filename.startswith("00000004") else "images_001"
            src = (tree["nih_root"] / folder / "images" / row.nih_filename).read_bytes()
            assert data == src and hashlib.sha256(src).hexdigest() == row.sha256 and len(src) == row.bytes
            assert zf.getinfo(row.nih_filename).compress_type == zipfile.ZIP_STORED
    assert "00000077_000.png" in proc.stdout and "1 missing" in proc.stdout
    assert not (export_run["temp"] / "selected_nih_png").exists()


def test_export_kernel_refuses_extra_columns_and_is_deterministic(tree, export_run, tmp_path) -> None:
    inp = tmp_path / "input"
    nih_copy = inp / "nih"
    shutil.copytree(tree["nih_root"], nih_copy)
    ds = inp / "ds"
    ds.mkdir()
    (ds / "nih_filelist.csv").write_text("nih_filename,arm\n00000001_000.png,A\n", encoding="utf-8")
    working = tmp_path / "w"
    working.mkdir()
    proc = _run_bundle(tree["build"], "res201-export", inp, working, tmp_path / "t")
    assert proc.returncode != 0 and "refusing" in proc.stderr
    assert not any(working.iterdir())
    # the same list gives the same zip bytes
    (ds / "nih_filelist.csv").write_text("nih_filename\n" + "\n".join(export_run["names"]) + "\n", encoding="utf-8")
    proc = _run_bundle(tree["build"], "res201-export", inp, working, tmp_path / "t")
    assert proc.returncode == 0, proc.stderr
    assert (working / "selected_nih_png.zip").read_bytes() == \
        (export_run["working"] / "selected_nih_png.zip").read_bytes()


# --------------------------------------------------------------------------- run_kernel.py (no network)
class FakeKaggle:
    """Stands in for the Kaggle CLI: scripted push, status and output responses."""

    def __init__(self, statuses, push=(0, "Kernel version 7 successfully pushed.  Please check progress at x"),
                 output_files=None, output_code=0):
        self.statuses = list(statuses)
        self.push = push
        self.output_files = output_files or {}
        self.output_code = output_code
        self.calls: list[list[str]] = []

    def __call__(self, cmd):
        cmd = list(cmd)
        self.calls.append(cmd)
        if cmd[1:3] == ["kernels", "push"]:
            return self.push
        if cmd[1:3] == ["kernels", "status"]:
            item = self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
            return item if isinstance(item, tuple) else (0, f'{cmd[3]} has status "{item}"')
        if cmd[1:3] == ["kernels", "output"]:
            dest = Path(cmd[cmd.index("-p") + 1])
            for name, data in self.output_files.items():
                (dest / name).write_bytes(data)
            return self.output_code, "Output file downloaded"
        raise AssertionError(cmd)


def _fake_build(tmp_path: Path, private: bool = True) -> Path:
    build = tmp_path / "build"
    folder = build / "res201-hash"
    folder.mkdir(parents=True)
    (folder / "kernel.py").write_text("print('hi')\n", encoding="utf-8")
    meta = bk.kernel_metadata("hash", "u")
    meta["is_private"] = private
    (folder / "kernel-metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    return build


class Clock:
    def __init__(self):
        self.t = 0.0
        self.sleeps: list[float] = []

    def sleep(self, s):
        self.sleeps.append(s)
        self.t += s

    def now(self):
        return self.t


def _ok_git(cmd):
    return 0, ""


def test_run_kernel_full_cycle(tmp_path) -> None:
    build = _fake_build(tmp_path)
    summary = json.dumps({"folders": {"images_001": {"status": "done"}}, "steps": {"discover": {"status": "done"}},
                          "n_hashed_total": 3}).encode()
    fake = FakeKaggle(["KernelWorkerStatus.QUEUED", "KernelWorkerStatus.RUNNING", "KernelWorkerStatus.COMPLETE"],
                      output_files={"run_summary.json": summary, "yield_table.csv": b"metric,value\nn_images,3\n"})
    clock = Clock()
    logs: list[str] = []
    code = rk.run("hash", "u", build, tmp_path / "out", repo_root=tmp_path, runner=fake, poll_s=30,
                  initial_delay_s=10, sleep=clock.sleep, clock=clock.now, log=logs.append, exe="kaggle", git=_ok_git)
    assert code == 0
    assert fake.calls[0] == ["kaggle", "kernels", "push", "-p", str(build / "res201-hash")]
    status_calls = [c for c in fake.calls if c[1:3] == ["kernels", "status"]]
    assert len(status_calls) == 3 and all(c[3] == "u/res201-hash/7" for c in status_calls)
    assert fake.calls[-1] == ["kaggle", "kernels", "output", "u/res201-hash/7", "-p",
                              str(tmp_path / "out" / "res201-hash"), "-o"]
    assert clock.sleeps == [10, 30, 30]
    text = "\n".join(logs)
    assert "complete" in text and "folders done 1" in text and "n_images=3" in text


def test_run_kernel_error_timeout_and_failures(tmp_path) -> None:
    build = _fake_build(tmp_path)
    kw = dict(repo_root=tmp_path, poll_s=60, initial_delay_s=0, exe="kaggle", git=_ok_git, log=lambda s: None)
    clock = Clock()
    fake = FakeKaggle(["running", "error"])
    assert rk.run("hash", "u", build, tmp_path / "o1", runner=fake, sleep=clock.sleep, clock=clock.now, **kw) == 1
    assert fake.calls[-1][1:3] == ["kernels", "output"]                       # the log is still downloaded
    clock = Clock()
    fake = FakeKaggle(["running"])
    assert rk.run("hash", "u", build, tmp_path / "o2", runner=fake, timeout_s=300, sleep=clock.sleep,
                  clock=clock.now, **kw) == 1
    assert not any(c[1:3] == ["kernels", "output"] for c in fake.calls)
    clock = Clock()
    fake = FakeKaggle([(1, "401 - Unauthorized")])
    assert rk.run("hash", "u", build, tmp_path / "o3", runner=fake, sleep=clock.sleep, clock=clock.now, **kw) == 1
    assert len([c for c in fake.calls if c[1:3] == ["kernels", "status"]]) == 5
    fake = FakeKaggle(["complete"], push=(1, "403 Client Error: Forbidden"))
    assert rk.run("hash", "u", build, tmp_path / "o4", runner=fake, sleep=clock.sleep, clock=clock.now, **kw) == 2
    assert len(fake.calls) == 1
    fake = FakeKaggle(["complete"], output_code=1)
    assert rk.run("hash", "u", build, tmp_path / "o5", runner=fake, sleep=clock.sleep, clock=clock.now, **kw) == 1
    fake = FakeKaggle(["complete"])
    assert rk.run("hash", "u", build, tmp_path / "o6", runner=fake, push=False, wait=False, version=4,
                  sleep=clock.sleep, clock=clock.now, **kw) == 0
    assert fake.calls == [["kaggle", "kernels", "output", "u/res201-hash/4", "-p",
                           str(tmp_path / "o6" / "res201-hash"), "-o"]]


def test_run_kernel_refusals(tmp_path) -> None:
    kw = dict(repo_root=tmp_path, exe="kaggle", log=lambda s: None, sleep=lambda s: None)
    fake = FakeKaggle(["complete"])
    assert rk.run("hash", "u", _fake_build(tmp_path / "pub", private=False), tmp_path / "o", runner=fake,
                  git=_ok_git, **kw) == 2
    assert fake.calls == []
    assert rk.run("hash", "other", _fake_build(tmp_path / "x"), tmp_path / "o", runner=fake, git=_ok_git, **kw) == 2
    assert rk.run("hash", "u", tmp_path / "missing", tmp_path / "o", runner=fake, git=_ok_git, **kw) == 2

    def not_ignored(cmd):
        return (1, "") if "check-ignore" in cmd else (0, "")

    def tracked(cmd):
        return (0, "data/kaggle/res201-hash/x.csv\n") if "ls-files" in cmd else (0, "")

    for git in (not_ignored, tracked, lambda cmd: (128, "fatal: not a git repository")):
        assert rk.run("hash", "u", _fake_build(tmp_path / f"b{id(git)}"), tmp_path / "o", runner=fake, git=git,
                      **kw) == 2
        assert fake.calls == []
    with pytest.raises(rk.RefusedError):
        rk.check_local_target(tmp_path / "inside", repo_root=tmp_path, git=not_ignored)
    rk.check_local_target(tmp_path.parent / "elsewhere", repo_root=tmp_path, git=not_ignored)  # outside: allowed


def test_check_local_target_with_real_git() -> None:
    if shutil.which("git") is None or not (REPO_ROOT / ".git").exists():
        pytest.skip("git or the repository is not available")
    rk.check_local_target(REPO_ROOT / "data" / "kaggle" / "res201-hash")          # git-ignored: allowed
    with pytest.raises(rk.RefusedError):
        rk.check_local_target(REPO_ROOT / "research")                           # tracked files
    with pytest.raises(rk.RefusedError):
        rk.check_local_target(REPO_ROOT / "kaggle_kernels" / "downloads")       # not ignored


def test_status_and_push_parsing() -> None:
    assert rk.parse_status('u/res201-hash has status "KernelWorkerStatus.COMPLETE"') == "complete"
    assert rk.parse_status('u/k has status "running"') == "running"
    assert rk.parse_status('u/k has status "KernelWorkerStatus.CANCEL_ACKNOWLEDGED"') == "cancel_acknowledged"
    assert rk.parse_status("status: KernelWorkerStatus.ERROR") == "error"
    assert rk.parse_status("") is None and rk.parse_status("nothing here") is None
    assert rk.parse_push_version("Kernel version 12 successfully pushed.  Please check") == 12
    assert rk.parse_push_version("Kernel push error: x") is None
    assert rk.push_failed(0, "Kernel push error: see previous output")
    assert rk.push_failed(1, "")
    assert not rk.push_failed(0, "Kernel version 3 successfully pushed.")
    assert rk.kernel_ref("u", "s") == "u/s" and rk.kernel_ref("u", "s", 3) == "u/s/3"
    assert rk.status_cmd("u/s", exe="k") == ["k", "kernels", "status", "u/s"]
    assert Path(rk.kaggle_exe()).name.lower().startswith("kaggle")


def test_summaries_of_real_kernel_outputs(hash_run, match_run, export_run) -> None:
    s = rk.summarize("hash", hash_run["working"])
    assert "folders done 2" in s and "dimension key" in s and "n_images=6" in s
    s = rk.summarize("match", match_run["working"])
    assert "status counts" in s and "spot-check grids: 1" in s and "1.2.3.test" not in s
    s = rk.summarize("export", export_run["working"])
    assert "manifest rows 3" in s and "zip entries 3" in s and "MISMATCH" not in s


# --------------------------------------------------------------------------- make_filelist_dataset.py
def _filelist(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "nih_filelist.csv"
    p.write_text(text, encoding="utf-8")
    return p


def test_filelist_validation(tmp_path) -> None:
    good = _filelist(tmp_path, "nih_filename\n00000002_000.png\n00000001_000.png\n\n")
    assert mfd.read_filelist(good) == ["00000002_000.png", "00000001_000.png"]
    for text, msg in [("nih_filename,arm\n00000001_000.png,A\n", "refusing columns"),
                      ("uid\nx\n", "refusing columns"),
                      ("nih_filename\n00000001_000.png,extra\n", "fields"),
                      ("nih_filename\n1.2.3.jpg\n", "not an NIH"),
                      ("nih_filename\n00000001_000.png\n00000001_000.png\n", "duplicate"),
                      ("nih_filename\n", "no file names"), ("", "empty")]:
        with pytest.raises(ValueError, match=msg):
            mfd.read_filelist(_filelist(tmp_path, text))


def test_dataset_metadata_private() -> None:
    m = mfd.dataset_metadata("someone")
    assert m["id"] == f"someone/{FILELIST_DATASET_SLUG}" and m["isPrivate"] is True
    assert 6 <= len(m["title"]) <= 50 and 6 <= len(FILELIST_DATASET_SLUG) <= 50
    assert len(m["licenses"]) == 1 and m["licenses"][0]["name"]
    with pytest.raises(ValueError):
        mfd.dataset_metadata("bad user")


def test_stage_writes_only_two_files(tmp_path) -> None:
    src = _filelist(tmp_path, "nih_filename\n00000002_000.png\n00000001_000.png\n")
    stage = tmp_path / "stage"
    mfd.stage(src, stage, "someone", repo_root=tmp_path, git=_ok_git)
    assert sorted(p.name for p in stage.iterdir()) == ["dataset-metadata.json", "nih_filelist.csv"]
    assert (stage / "nih_filelist.csv").read_text(encoding="utf-8") == \
        "nih_filename\n00000001_000.png\n00000002_000.png\n"
    assert json.loads((stage / "dataset-metadata.json").read_text(encoding="utf-8")) == mfd.dataset_metadata("someone")
    mfd.stage(src, stage, "someone", repo_root=tmp_path, git=_ok_git)          # restaging the same folder is fine
    (stage / "items.csv").write_text("x", encoding="utf-8")
    with pytest.raises(rk.RefusedError, match="other files"):
        mfd.stage(src, stage, "someone", repo_root=tmp_path, git=_ok_git)
    with pytest.raises(rk.RefusedError):
        mfd.stage(src, tmp_path / "stage2", "someone", repo_root=tmp_path,
                  git=lambda cmd: (1, "") if "check-ignore" in cmd else (0, ""))


def test_publish_create_or_version(tmp_path) -> None:
    calls: list[list[str]] = []

    def runner_for(status):
        def runner(cmd):
            calls.append(list(cmd))
            if cmd[1:3] == ["datasets", "status"]:
                return status
            return 0, "Your private Dataset is being created. Please check progress at x"
        return runner

    stage = tmp_path / "stage"
    action, code = mfd.publish(stage, "u", "msg", runner=runner_for((1, "404 Client Error: Not Found")), exe="k",
                               log=lambda s: None)
    assert (action, code) == ("create", 0)
    assert calls[-1] == ["k", "datasets", "create", "-p", str(stage)]
    action, code = mfd.publish(stage, "u", "new list", runner=runner_for((0, "ready")), exe="k", log=lambda s: None)
    assert (action, code) == ("version", 0)
    assert calls[-1] == ["k", "datasets", "version", "-p", str(stage), "-m", "new list"]
    assert all("-u" not in c and "--public" not in c for c in calls)
    assert calls[0] == ["k", "datasets", "status", "u/res201-nih-filelist"]
    n = len(calls)
    assert mfd.publish(stage, "u", "m", runner=runner_for((0, "ready")), exe="k", dry_run=True,
                       log=lambda s: None) == ("version", 0)
    assert len(calls) == n + 1                                                  # only the status call ran

    def failing(cmd):
        return (0, "ready") if cmd[1:3] == ["datasets", "status"] else (1, "Dataset version error")

    assert mfd.publish(stage, "u", "m", runner=failing, exe="k", log=lambda s: None)[1] == 1
