"""NIH ChestX-ray14 metadata loading and image discovery (kernel-side).

Pixels come from the NIH ChestX-ray14 release (NIH Clinical Center; Wang et al., CVPR 2017; the
NIH Box release). This module runs inside the private Kaggle kernels and locally; it uses only the
standard library and pandas, and Python 3.10 syntax. Private helpers carry the ``_nih_`` prefix.

``load_nih_meta`` returns schema S2 of ``docs/PIPELINE_SPEC.md``::

    nih_filename, nih_patient_id, followup, age, sex, view, orig_w, orig_h, finding_labels

and accepts both header variants of ``Data_Entry_2017*.csv``: the bracketed headers
``OriginalImage[Width,Height]`` and ``OriginalImagePixelSpacing[x,y]`` are unquoted, so a CSV reader
splits them into ``OriginalImage[Width`` / ``Height]`` and ``OriginalImagePixelSpacing[x`` / ``y]``;
the Kaggle copy also ends every line with a comma (a trailing unnamed column). A quoted, combined
header token is expanded to its two value positions as well.
"""
from __future__ import annotations

import csv
import fnmatch
import os
import re
from pathlib import Path

import pandas as pd

NIH_META_COLUMNS = (
    "nih_filename", "nih_patient_id", "followup", "age", "sex", "view", "orig_w", "orig_h", "finding_labels",
)
NIH_FILENAME_RE = re.compile(r"^(\d{8})_(\d{3})\.png$")

#: directory names the input walker never descends into (large image folders)
DEFAULT_PRUNE_DIRS = ("images", "train", "test", "train_tfrecords", "test_tfrecords")

_NIH_SIMPLE = {
    "imageindex": ["nih_filename"],
    "findinglabels": ["finding_labels"],
    "follow-up#": ["followup"],
    "followup#": ["followup"],
    "follow-up": ["followup"],
    "followup": ["followup"],
    "patientid": ["nih_patient_id"],
    "patientage": ["age"],
    "patientgender": ["sex"],
    "patientsex": ["sex"],
    "viewposition": ["view"],
    "originalimage[width,height]": ["orig_w", "orig_h"],
    "originalimage[width": ["orig_w"],
    "originalimagewidth": ["orig_w"],
    "height]": ["orig_h"],
    "originalimageheight": ["orig_h"],
    "originalimagepixelspacing[x,y]": ["spacing_x", "spacing_y"],
    "originalimagepixelspacing[x": ["spacing_x"],
    "y]": ["spacing_y"],
}


def _nih_header_names(header: list[str]) -> list[str | None]:
    """Map raw header tokens to canonical names by position (None = ignored column)."""
    names: list[str | None] = []
    for token in header:
        key = re.sub(r"\s+", "", token.replace("﻿", "")).lower()
        if key in _NIH_SIMPLE:
            names.extend(_NIH_SIMPLE[key])
        else:
            names.append(None)
    return names


def _nih_int(values: list[str], column: str) -> list[int]:
    """Parse integer fields (accepting '12.0'); raise ValueError naming the column on bad input."""
    out = []
    for i, v in enumerate(values):
        s = v.strip()
        try:
            out.append(int(s) if re.fullmatch(r"[+-]?\d+", s) else int(float(s)))
        except ValueError:
            raise ValueError(f"NIH metadata: column {column!r} row {i + 2}: not an integer: {v!r}") from None
    return out


def _nih_age(value: str) -> int:
    """Age in years from '57' or the older '057Y' form; months/days give 0; unparsable gives -1."""
    m = re.match(r"^\s*(\d+)\s*([YyMmDdWw]?)\s*$", value)
    if not m:
        return -1
    n = int(m.group(1))
    unit = m.group(2).upper()
    if unit in ("M", "D", "W"):
        return n // 12 if unit == "M" else 0
    return n


def load_nih_meta(path: str | Path) -> pd.DataFrame:
    """Load ``Data_Entry_2017.csv`` (Kaggle copy) or ``Data_Entry_2017_v2020.csv`` as an S2 frame.

    Raises ValueError when a required column is missing, a filename does not look like
    ``00000001_000.png``, or a filename appears twice.
    """
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError(f"{path}: empty file") from None
        rows = [r for r in reader if r and any(c.strip() for c in r)]
    names = _nih_header_names(header)
    positions: dict[str, int] = {}
    for pos, name in enumerate(names):
        if name is not None and name not in positions:
            positions[name] = pos
    missing = [c for c in NIH_META_COLUMNS if c not in positions]
    if missing:
        raise ValueError(f"{path}: NIH metadata header lacks {missing}; header was {header}")

    def column(name: str) -> list[str]:
        p = positions[name]
        return [r[p] if p < len(r) else "" for r in rows]

    filenames = [v.strip() for v in column("nih_filename")]
    bad = [f for f in filenames if not NIH_FILENAME_RE.match(f)]
    if bad:
        raise ValueError(f"{path}: {len(bad)} unexpected image names, e.g. {bad[:3]}")
    df = pd.DataFrame({
        "nih_filename": filenames,
        "nih_patient_id": _nih_int(column("nih_patient_id"), "Patient ID"),
        "followup": _nih_int(column("followup"), "Follow-up #"),
        "age": [_nih_age(v) for v in column("age")],
        "sex": [v.strip().upper() for v in column("sex")],
        "view": [v.strip().upper() for v in column("view")],
        "orig_w": _nih_int(column("orig_w"), "OriginalImage[Width"),
        "orig_h": _nih_int(column("orig_h"), "Height]"),
        "finding_labels": [v.strip() for v in column("finding_labels")],
    })
    for c in ("nih_patient_id", "followup", "age", "orig_w", "orig_h"):
        df[c] = df[c].astype("int64")
    dup = df["nih_filename"].duplicated()
    if dup.any():
        raise ValueError(f"{path}: duplicated image names, e.g. {df.loc[dup, 'nih_filename'].head(3).tolist()}")
    return df.loc[:, list(NIH_META_COLUMNS)].reset_index(drop=True)


def nih_patient_from_filename(name: str) -> int:
    """NIH patient id encoded in the filename ('00000013_005.png' -> 13); -1 if the name does not match."""
    m = NIH_FILENAME_RE.match(str(name).strip())
    return int(m.group(1)) if m else -1


# --------------------------------------------------------------------------- image discovery
def image_folders(root: str | Path) -> list[tuple[str, Path]]:
    """NIH image folders under one dataset root, as (folder name, directory holding the PNGs).

    Recognised layouts: ``images_*/images/*.png`` (Kaggle and Box), ``images_*/*.png``,
    ``images/*.png`` and PNGs directly in ``root`` (name = the root's own name). Sorted by name.
    """
    root = Path(root)
    found: list[tuple[str, Path]] = []
    seen: set[str] = set()
    if not root.is_dir():
        return found
    for sub in sorted(p for p in root.iterdir() if p.is_dir() and fnmatch.fnmatch(p.name, "images_*")):
        inner = sub / "images"
        if inner.is_dir() and any(inner.glob("*.png")):
            found.append((sub.name, inner))
            seen.add(sub.name)
        elif any(sub.glob("*.png")):
            found.append((sub.name, sub))
            seen.add(sub.name)
    flat = root / "images"
    if flat.is_dir() and "images" not in seen and any(flat.glob("*.png")):
        found.append(("images", flat))
    if not found and any(root.glob("*.png")):
        found.append((root.name or "root", root))
    return sorted(found, key=lambda t: t[0])


def locate_images(root: str | Path) -> dict[str, Path]:
    """Map NIH PNG filename -> path under ``root`` (layouts as in ``image_folders``; first one wins)."""
    out: dict[str, Path] = {}
    for _, folder in image_folders(root):
        for p in sorted(folder.glob("*.png")):
            out.setdefault(p.name, p)
    return out


def find_inputs(root: str | Path, pattern: str, kind: str = "file", max_depth: int = 6,
                prune: tuple[str, ...] = DEFAULT_PRUNE_DIRS) -> list[Path]:
    """Bounded walk under ``root`` returning files (or directories) whose name matches ``pattern``.

    Used by the kernels to discover inputs under ``/kaggle/input`` without hard-coding mount names.
    Directories named in ``prune`` are never entered (but can still be returned when kind="dir").
    Results are sorted by path.
    """
    if kind not in ("file", "dir"):
        raise ValueError("kind must be 'file' or 'dir'")
    root = Path(root)
    results: list[Path] = []
    if not root.is_dir():
        return results
    stack: list[tuple[Path, int]] = [(root, 0)]
    while stack:
        current, depth = stack.pop()
        try:
            entries = list(os.scandir(current))
        except OSError:
            continue
        for entry in entries:
            try:
                is_dir = entry.is_dir()
            except OSError:
                continue
            if is_dir:
                if kind == "dir" and fnmatch.fnmatch(entry.name, pattern):
                    results.append(Path(entry.path))
                if depth + 1 <= max_depth and entry.name not in prune:
                    stack.append((Path(entry.path), depth + 1))
            elif kind == "file" and fnmatch.fnmatch(entry.name, pattern):
                results.append(Path(entry.path))
    return sorted(results)


def find_nih_meta_csv(input_root: str | Path) -> Path | None:
    """The NIH metadata CSV under ``input_root`` (prefers ``Data_Entry_2017.csv``), or None."""
    hits = find_inputs(input_root, "Data_Entry_2017*.csv")
    if not hits:
        return None
    exact = [p for p in hits if p.name == "Data_Entry_2017.csv"]
    return (exact or hits)[0]


def find_nih_image_folders(input_root: str | Path) -> list[tuple[str, Path]]:
    """All NIH image folders under ``input_root``: next to each metadata CSV, else any ``images_*`` dir."""
    roots = sorted({p.parent for p in find_inputs(input_root, "Data_Entry_2017*.csv")})
    folders: dict[str, Path] = {}
    for r in roots:
        for name, folder in image_folders(r):
            folders.setdefault(name, folder)
    if not folders:
        parents = sorted({p.parent for p in find_inputs(input_root, "images_*", kind="dir")})
        for r in parents:
            for name, folder in image_folders(r):
                folders.setdefault(name, folder)
    return sorted(folders.items(), key=lambda t: t[0])


def locate_all_images(input_root: str | Path) -> dict[str, Path]:
    """Map NIH PNG filename -> path over every NIH image folder under ``input_root``."""
    out: dict[str, Path] = {}
    for _, folder in find_nih_image_folders(input_root):
        for p in sorted(folder.glob("*.png")):
            out.setdefault(p.name, p)
    return out
