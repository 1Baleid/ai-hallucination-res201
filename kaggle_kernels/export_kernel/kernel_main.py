"""res201-export: copy the selected NIH PNGs into one zip with a SHA-256 manifest (private Kaggle CPU kernel).

Inputs, found by globbing under ``--input-root`` (default ``/kaggle/input``):

* ``nih_filelist.csv`` from the private dataset ``<user>/res201-nih-filelist``: exactly one column,
  ``nih_filename`` (any other column is refused);
* NIH ChestX-ray14 PNGs (dataset ``nih-chest-xrays/data``; NIH Clinical Center; Wang et al., CVPR 2017;
  the NIH Box release).

Outputs, in ``--working`` (default ``/kaggle/working``), and nothing else:

* ``selected_nih_png.zip``: the listed PNGs, stored uncompressed under their NIH file names, with
  fixed entry timestamps so the same list gives the same zip bytes;
* ``manifest.csv``: nih_filename, sha256, bytes (one row per PNG in the zip, sorted by name).

The PNGs are first copied to a staging folder under ``--temp``, hashed, zipped and verified against
the manifest; the loose copies are then removed. Listed names that are not found are reported in the
log and left out of the zip and the manifest. No RANZCR file is read by this kernel.
Run locally with ``python kernel_main.py --input-root <dir> --working <dir> --temp <dir> --workers 1``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import shutil
import sys
import tempfile
import zipfile
from multiprocessing.pool import ThreadPool
from pathlib import Path

# BUNDLE-STRIP-START
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
# BUNDLE-STRIP-END
from cxrdev.nih import NIH_FILENAME_RE, find_inputs, locate_all_images

KERNEL_NAME = "res201-export"
FILELIST_NAME = "nih_filelist.csv"
ZIP_NAME = "selected_nih_png.zip"
MANIFEST_NAME = "manifest.csv"
ZIP_DATE = (1980, 1, 1, 0, 0, 0)


def read_filelist(path: Path) -> list:
    """Validated, de-duplicated NIH filenames from a one-column ``nih_filename`` CSV (order kept)."""
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        raise ValueError(f"{path}: empty file list")
    header = [h.strip() for h in rows[0]]
    if header != ["nih_filename"]:
        raise ValueError(f"{path}: refusing a file list whose columns are {header}; only 'nih_filename' is allowed")
    names: list = []
    seen: set = set()
    for i, row in enumerate(rows[1:], start=2):
        if not row or not any(c.strip() for c in row):
            continue
        if len(row) != 1:
            raise ValueError(f"{path}: line {i} has {len(row)} fields; only one column is allowed")
        name = row[0].strip()
        if not NIH_FILENAME_RE.match(name):
            raise ValueError(f"{path}: line {i}: {name!r} is not an NIH file name")
        if name not in seen:
            seen.add(name)
            names.append(name)
    return names


def _sha256_file(path: Path) -> tuple:
    h = hashlib.sha256()
    size = 0
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
            size += len(block)
    return h.hexdigest(), size


def find_filelist(input_root: Path) -> Path:
    """The single ``nih_filelist.csv`` under ``input_root`` (several identical copies are accepted)."""
    hits = find_inputs(input_root, FILELIST_NAME)
    if not hits:
        raise FileNotFoundError(f"{FILELIST_NAME} not found under {input_root}")
    first = hits[0].read_bytes()
    for other in hits[1:]:
        if other.read_bytes() != first:
            raise ValueError(f"several different {FILELIST_NAME} files under {input_root}: {hits}")
    return hits[0]


def main(input_root: str = "/kaggle/input", working: str = "/kaggle/working", temp: str = "/kaggle/temp",
         workers: int = 4) -> dict:
    """Build the zip and the manifest; returns a summary dict (printed, not written)."""
    input_root, working, temp = Path(input_root), Path(working), Path(temp)
    working.mkdir(parents=True, exist_ok=True)
    filelist = find_filelist(input_root)
    names = read_filelist(filelist)
    located = locate_all_images(input_root)
    missing = [n for n in names if n not in located]
    present = sorted(n for n in names if n in located)
    print(f"[{KERNEL_NAME}] {len(names)} listed, {len(present)} found, {len(missing)} missing", flush=True)
    if missing:
        print(f"[{KERNEL_NAME}] missing (first 50): {missing[:50]}", flush=True)

    try:
        temp.mkdir(parents=True, exist_ok=True)
    except OSError:
        temp = Path(tempfile.mkdtemp(prefix="res201_export_"))
    staging = temp / "selected_nih_png"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    def copy_and_hash(name: str) -> tuple:
        dst = staging / name
        shutil.copyfile(located[name], dst)
        digest, size = _sha256_file(dst)
        return name, digest, size

    if workers > 1 and len(present) > 1:
        with ThreadPool(int(workers)) as pool:
            manifest = pool.map(copy_and_hash, present)
    else:
        manifest = [copy_and_hash(n) for n in present]
    manifest.sort(key=lambda r: r[0])

    zip_path = working / ZIP_NAME
    tmp_zip = working / (ZIP_NAME + ".part")
    with zipfile.ZipFile(tmp_zip, "w", compression=zipfile.ZIP_STORED) as zf:
        for name, _digest, _size in manifest:
            info = zipfile.ZipInfo(name, date_time=ZIP_DATE)
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o644 << 16
            zf.writestr(info, (staging / name).read_bytes())
    with zipfile.ZipFile(tmp_zip) as zf:
        if zf.testzip() is not None:
            raise RuntimeError("zip verification failed (CRC)")
        entries = zf.namelist()
        if entries != [r[0] for r in manifest]:
            raise RuntimeError("zip entries differ from the manifest")
        for name, digest, size in manifest:
            data = zf.read(name)
            if len(data) != size or hashlib.sha256(data).hexdigest() != digest:
                raise RuntimeError(f"zip entry {name} differs from its manifest row")
    os.replace(tmp_zip, zip_path)

    tmp_manifest = working / (MANIFEST_NAME + ".part")
    with open(tmp_manifest, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["nih_filename", "sha256", "bytes"])
        writer.writerows(manifest)
    os.replace(tmp_manifest, working / MANIFEST_NAME)

    shutil.rmtree(staging)
    summary = {"kernel": KERNEL_NAME, "filelist": str(filelist), "n_listed": len(names), "n_zipped": len(manifest),
               "n_missing": len(missing), "missing_head": missing[:50], "zip_bytes": zip_path.stat().st_size}
    print(f"[{KERNEL_NAME}] wrote {ZIP_NAME} ({summary['zip_bytes']} bytes, {len(manifest)} PNGs) and {MANIFEST_NAME}",
          flush=True)
    return summary


def main_cli(argv: list | None = None) -> int:
    """Command-line entry point (exit code 1 on a fatal error, so the Kaggle status shows it)."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--input-root", default="/kaggle/input")
    ap.add_argument("--working", default="/kaggle/working")
    ap.add_argument("--temp", default="/kaggle/temp")
    ap.add_argument("--workers", type=int, default=4)
    args, _unknown = ap.parse_known_args(argv)
    main(args.input_root, args.working, args.temp, args.workers)
    return 0


if __name__ == "__main__":
    sys.exit(main_cli())
