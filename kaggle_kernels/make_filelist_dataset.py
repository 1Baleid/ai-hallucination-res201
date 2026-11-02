"""Create or version the PRIVATE Kaggle dataset ``<user>/res201-nih-filelist`` (the export kernel's input).

Usage (from the repository root)::

    python kaggle_kernels/make_filelist_dataset.py --username <kaggle-user> --filelist items/nih_filelist.csv
        [--stage data/kaggle/res201-nih-filelist-upload] [--message "selection of 2026-11-02"] [--dry-run]

The dataset holds exactly one file, ``nih_filelist.csv``, with exactly one column, ``nih_filename``
(NIH ChestX-ray14 names such as ``00000001_000.png``). Any other column, a malformed name or a
duplicate is refused: the file list carries NIH file names only, no arm, label, UID or patient
column (PROTOCOL section 9). The script writes the list and ``dataset-metadata.json`` into a staging
folder that must be git-ignored and may hold nothing else, then runs ``kaggle datasets status``:
when the dataset exists it runs ``kaggle datasets version -p <stage> -m <message>``, otherwise
``kaggle datasets create -p <stage>``. Datasets are private unless ``--public``/``-u`` is passed to
``datasets create``; this script never passes it and refuses any command that contains it.
The Kaggle CLI is called through ``subprocess`` only.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Callable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cxrdev.constants import FILELIST_DATASET_SLUG  # noqa: E402
from kaggle_kernels.run_kernel import RefusedError, check_local_target, default_runner, kaggle_exe  # noqa: E402

FILELIST_NAME = "nih_filelist.csv"
METADATA_NAME = "dataset-metadata.json"
DEFAULT_STAGE = REPO_ROOT / "data" / "kaggle" / "res201-nih-filelist-upload"
NIH_NAME_RE = re.compile(r"^\d{8}_\d{3}\.png$")
DATASET_TITLE = "RES201 NIH filelist"
_PUBLIC_FLAGS = ("-u", "--public")


def read_filelist(path: Path) -> list[str]:
    """Validated NIH filenames from a CSV with exactly one column ``nih_filename``.

    Raises ValueError on any other column, a row with several fields, a malformed name, a duplicate
    or an empty list.
    """
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        raise ValueError(f"{path}: empty file")
    header = [h.strip() for h in rows[0]]
    if header != ["nih_filename"]:
        raise ValueError(f"{path}: refusing columns {header}: the file list may hold only 'nih_filename'")
    names: list[str] = []
    seen: set[str] = set()
    for i, row in enumerate(rows[1:], start=2):
        if not row or not any(c.strip() for c in row):
            continue
        if len(row) != 1:
            raise ValueError(f"{path}: line {i} has {len(row)} fields; only one column is allowed")
        name = row[0].strip()
        if not NIH_NAME_RE.match(name):
            raise ValueError(f"{path}: line {i}: {name!r} is not an NIH file name")
        if name in seen:
            raise ValueError(f"{path}: line {i}: duplicate {name}")
        seen.add(name)
        names.append(name)
    if not names:
        raise ValueError(f"{path}: no file names")
    return names


def dataset_metadata(username: str) -> dict:
    """``dataset-metadata.json`` content for the private file-list dataset."""
    if not re.match(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$", username or ""):
        raise ValueError(f"invalid Kaggle username: {username!r}")
    return {
        "title": DATASET_TITLE,
        "id": f"{username}/{FILELIST_DATASET_SLUG}",
        "licenses": [{"name": "other"}],
        "isPrivate": True,
        "description": ("Private working list of NIH ChestX-ray14 image file names (NIH Clinical Center; Wang et "
                        "al., CVPR 2017; the NIH Box release), read by the private res201-export kernel. One "
                        "column, nih_filename; no labels and no other identifiers. Not for redistribution."),
        "keywords": [],
    }


def stage(filelist: Path, stage_dir: Path, username: str, repo_root: Path = REPO_ROOT,
          git: Callable | None = None) -> Path:
    """Write ``nih_filelist.csv`` (sorted) and ``dataset-metadata.json`` into ``stage_dir``.

    Refuses a staging folder inside the repository that is not git-ignored or holds tracked files,
    and a staging folder that holds any other file.
    """
    names = read_filelist(Path(filelist))
    stage_dir = Path(stage_dir)
    check_local_target(stage_dir, repo_root, git)
    stage_dir.mkdir(parents=True, exist_ok=True)
    extra = sorted(p.name for p in stage_dir.iterdir() if p.name not in (FILELIST_NAME, METADATA_NAME))
    if extra:
        raise RefusedError(f"refusing {stage_dir}: it holds other files {extra}; the dataset must hold only "
                           f"{FILELIST_NAME}")
    with open(stage_dir / FILELIST_NAME, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["nih_filename"])
        writer.writerows([n] for n in sorted(names))
    with open(stage_dir / METADATA_NAME, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(dataset_metadata(username), indent=2) + "\n")
    return stage_dir


def status_cmd(username: str, exe: str | None = None) -> list[str]:
    return [exe or kaggle_exe(), "datasets", "status", f"{username}/{FILELIST_DATASET_SLUG}"]


def create_cmd(stage_dir: Path, exe: str | None = None) -> list[str]:
    return [exe or kaggle_exe(), "datasets", "create", "-p", str(stage_dir)]


def version_cmd(stage_dir: Path, message: str, exe: str | None = None) -> list[str]:
    return [exe or kaggle_exe(), "datasets", "version", "-p", str(stage_dir), "-m", message]


def dataset_exists(code: int, text: str) -> bool:
    """Interpret ``kaggle datasets status``: exists when it succeeds without an error message."""
    low = (text or "").lower()
    return code == 0 and not any(w in low for w in ("404", "not found", "error", "forbidden", "403"))


def publish(stage_dir: Path, username: str, message: str, runner: Callable[[Sequence[str]], tuple] = default_runner,
            exe: str | None = None, dry_run: bool = False, log: Callable[[str], None] = print) -> tuple[str, int]:
    """Create or version the dataset; returns (action, exit code). action is "create" or "version"."""
    code, out = runner(status_cmd(username, exe))
    action = "version" if dataset_exists(code, out) else "create"
    cmd = version_cmd(stage_dir, message, exe) if action == "version" else create_cmd(stage_dir, exe)
    if any(flag in cmd for flag in _PUBLIC_FLAGS):
        raise RefusedError("refusing a public dataset command")
    log(("dry run: " if dry_run else "") + " ".join(cmd))
    if dry_run:
        return action, 0
    code, out = runner(cmd)
    log(out.strip())
    if code != 0 or "error" in out.lower():
        return action, code or 1
    return action, 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Create or version the private res201-nih-filelist dataset.")
    ap.add_argument("--username", required=True)
    ap.add_argument("--filelist", required=True, help="CSV with the single column nih_filename")
    ap.add_argument("--stage", default=str(DEFAULT_STAGE), help="git-ignored staging folder")
    ap.add_argument("--message", default="RES201 NIH file list", help="version notes when the dataset exists")
    ap.add_argument("--dry-run", action="store_true", help="stage the files and print the command only")
    args = ap.parse_args(argv)
    try:
        folder = stage(Path(args.filelist), Path(args.stage), args.username)
        action, code = publish(folder, args.username, args.message, dry_run=args.dry_run)
    except (ValueError, RefusedError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    print(f"{action}: exit code {code}")
    return code


if __name__ == "__main__":
    sys.exit(main())
