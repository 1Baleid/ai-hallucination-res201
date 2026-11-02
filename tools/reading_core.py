"""Blind ETT reading: the session logic behind ``tools/read_tool.py`` (no GUI imports).

PROTOCOL.md section 2 (verification, v0.4): the student reads every A, B and U image
(360 at the registered n; every A image is read because RANZCR CLiP labels tracheostomy
tubes as ETT) in one random order under opaque ids, blind to arm and to model output,
records present / absent / skip per image with three flags (ambiguous, tracheostomy,
out_of_field; an A image whose only airway tube is a tracheostomy is flagged
"tracheostomy" and later excluded), and saves the result with its
SHA-256 and a UTC timestamp before any join to arms. 100 items are re-read blind no
earlier than 48 h after the read was saved (retest mode); Cohen's kappa is computed
elsewhere.

Files (docs/PIPELINE_SPEC.md, S10 and S11):

* input: a reading list with exactly the columns ``read_id`` (8-char uppercase hex),
  ``nih_filename`` and ``image_path``. It is read in file order: the random order and
  its seed were fixed when the list was built. An optional sidecar
  ``<list stem>.meta.json`` (or ``<list file name>.meta.json``) supplies that seed.
* autosave: ``<out stem>.partial.csv`` is rewritten atomically after every response
  and is the resume point after a crash.
* output: ``<out>`` with the S11 columns, plus ``<out stem>.meta.json`` =
  {sha256, saved_utc, seed, n, mode, tool_version}. Example: ``--out
  items/verified_items.csv`` gives ``items/verified_items.partial.csv`` while reading
  and ``items/verified_items.meta.json`` at the end. A finalised output is never
  overwritten.

Blinding: a session opens only the reading list, its sidecar, its own partial, output
and meta files and the image files. It refuses to open ``reading_key.csv``,
``items.csv`` or model-output files, and refuses lists with any column beyond S10.
What a front end may show is limited to ``ReadingSession.view()`` ("k of N", the
read_id and the reader's own draft answer); NIH filenames and image paths are never
part of it, because two filenames with the same patient prefix would reveal a pair.

The labels of record are the RANZCR CLiP labels (43 trained labellers: radiologists,
radiology trainees, hospital medical officers); this reading is reported as agreement
with them.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
import tempfile
import time
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from cxrdev.constants import derive_seed

__all__ = [
    "TOOL_VERSION", "MODES", "RESPONSES", "FLAGS", "FLAG_COLUMNS", "LIST_COLUMNS",
    "OUTPUT_COLUMNS", "META_KEYS", "RETEST_MIN_HOURS", "PRACTICE_N", "CONTRAST_PRESETS",
    "BLINDED_FILE_NAMES", "STUDY_LIST_NAMES",
    "ReadingError", "ListError", "OutputExistsError", "ResumeError", "ResponseError",
    "ImageUnavailableError", "IncompleteError", "AutosaveError", "ConcurrentWriteError",
    "SessionClosedError", "MetaError", "RetestGapError",
    "ReadingItem", "Response", "ViewState", "FinaliseResult", "RunPlan", "ReadingSession",
    "format_utc", "parse_utc", "utc_now", "sha256_bytes", "sha256_file", "clean_note",
    "partial_path_for", "meta_path_for", "list_sidecar_paths", "read_list_seed",
    "resolve_image_path", "load_reading_list", "render_csv", "read_meta", "verify_output",
    "retest_earliest_start", "check_retest_gap", "load_display_image", "to_display_gray",
    "apply_contrast", "next_contrast", "fit_size", "fit_image", "check_practice_list",
    "make_practice_dir", "write_demo_list", "plan_run",
]

# ---------------------------------------------------------------- fixed values
TOOL_VERSION = "1.0.0"
MODES = ("read", "retest")
RESPONSES = ("present", "absent", "skip")
FLAGS = ("ambiguous", "tracheostomy", "out_of_field")
FLAG_COLUMNS = tuple(f"flag_{name}" for name in FLAGS)
LIST_COLUMNS = ("read_id", "nih_filename", "image_path")
OUTPUT_COLUMNS = (
    "read_id", "nih_filename", "response", *FLAG_COLUMNS,
    "note", "seconds", "order_index", "timestamp_utc",
)
META_KEYS = ("sha256", "saved_utc", "seed", "n", "mode", "tool_version")
READ_ID_PATTERN = re.compile(r"[0-9A-F]{8}")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
RETEST_MIN_HOURS = 48.0          # PROTOCOL section 2: retest no earlier than read end + 48 h
PRACTICE_N = 10                  # images in the practice run
CONTRAST_PRESETS = ("original", "autocontrast", "equalised")
AUTOCONTRAST_CUTOFF = 1          # percent of pixels clipped at each end by "autocontrast"
SKIP_REMAINING_NOTE = "not read: session ended early"

# Files the blind reader must never open (they reveal arms, labels or model output).
BLINDED_FILE_NAMES = frozenset({
    "reading_key.csv",        # read_id -> item_id (S10)
    "items.csv",              # arms and RANZCR CLiP labels (S5)
    "pod_items.csv",          # item ids and prompt sets (S6)
    "parsed_results.csv",     # model output (S8)
    "detector_scores.csv",    # MedSigLIP output (S9)
    "exclusions.csv",         # item ids (S12)
    "match_table.csv",        # RANZCR-derived linkage table (S3)
    "selection_report.json",  # arm counts
})
BLINDED_SUFFIXES = frozenset({".jsonl"})   # pod run records (S7)
# Tool outputs: never a reading list (opening one in retest mode would expose earlier answers).
OUTPUT_FILE_NAMES = frozenset({"verified_items.csv", "retest.csv"})
# Study lists: a practice run must not show study images before the blind read.
STUDY_LIST_NAMES = frozenset({"reading_list.csv", "retest_list.csv"})

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- errors
class ReadingError(Exception):
    """Base class for every refusal or failure of the reading tool."""


class ListError(ReadingError, ValueError):
    """The reading list or its sidecar is unusable or would unblind the reader."""


class OutputExistsError(ReadingError, FileExistsError):
    """A finalised output (or its meta file) already exists; it is never overwritten."""


class ResumeError(ReadingError, ValueError):
    """The partial (autosave) file cannot be resumed against this reading list."""


class ResponseError(ReadingError, ValueError):
    """An answer, flag or note is not valid for the current item."""


class ImageUnavailableError(ReadingError):
    """The image file of an item is missing or cannot be decoded."""


class IncompleteError(ReadingError):
    """Finalisation was requested while some items have no response."""


class AutosaveError(ReadingError, OSError):
    """The partial file could not be written; the answer is kept in memory."""


class ConcurrentWriteError(ReadingError):
    """The partial file was changed on disk by something other than this session."""


class SessionClosedError(ReadingError):
    """The session has been finalised; nothing can change any more."""


class MetaError(ReadingError, ValueError):
    """A meta sidecar is missing or malformed, or does not match its output file."""


class RetestGapError(ReadingError):
    """The retest would start less than 48 h after the first read was saved."""


# ---------------------------------------------------------------- records
@dataclass(frozen=True)
class ReadingItem:
    """One row of a reading list. ``nih_filename`` and ``image_path`` stay out of repr."""

    order_index: int
    read_id: str
    nih_filename: str = field(repr=False)
    image_path: Path = field(repr=False)


@dataclass(frozen=True)
class Response:
    """The recorded answer for one item."""

    response: str                 # "present" | "absent" | "skip"
    flags: frozenset[str]         # subset of FLAGS
    note: str
    seconds: float                # on-screen time summed over visits, up to the last confirmation
    timestamp_utc: str            # time of the last confirmation (ISO 8601, "Z")


@dataclass(frozen=True)
class ViewState:
    """Everything a front end may show. It never holds a filename or an image path."""

    position: int                 # 0-based index of the current item; equals n when complete
    n: int
    read_id: str                  # "" when complete
    complete: bool
    response: str | None          # the reader's draft answer for the current item
    flags: frozenset[str]
    note: str
    answered: int
    stepped_back: bool
    can_go_back: bool
    image_problem: str | None
    mode: str
    finalised: bool

    @property
    def progress(self) -> str:
        """The "k of N" text (1-based)."""
        if self.complete:
            return f"{self.n} of {self.n} answered"
        return f"{self.position + 1} of {self.n}"


@dataclass(frozen=True)
class FinaliseResult:
    """What ``ReadingSession.finalise`` wrote."""

    out_path: Path
    meta_path: Path
    sha256: str
    saved_utc: str
    n: int
    mode: str
    n_skip: int
    n_flagged: int
    partial_removed: bool
    partial_path: Path
    earliest_retest_utc: str | None   # read mode only: saved_utc + 48 h

    def summary_lines(self) -> list[str]:
        """Plain-text summary for the console and the closing dialog."""
        local = parse_utc(self.saved_utc).astimezone()
        lines = [
            f"Saved {self.n} answers ({self.mode}) to {self.out_path}",
            f"Fingerprint file: {self.meta_path}",
            f"SHA-256: {self.sha256}",
            f"Saved at (UTC): {self.saved_utc}; local time {local:%Y-%m-%d %H:%M:%S}",
            f"Skipped: {self.n_skip}; images with at least one flag: {self.n_flagged}",
        ]
        if self.earliest_retest_utc is not None:
            retest_local = parse_utc(self.earliest_retest_utc).astimezone()
            lines.append(
                f"Earliest retest start (48 h later): {self.earliest_retest_utc} UTC; "
                f"local time {retest_local:%Y-%m-%d %H:%M}"
            )
        if not self.partial_removed:
            lines.append(f"The autosave file could not be removed; delete it by hand: {self.partial_path}")
        return lines


@dataclass(frozen=True)
class RunPlan:
    """Resolved inputs for one run of the tool (see ``plan_run``)."""

    list_path: Path
    out_path: Path
    mode: str
    limit: int | None
    practice_dir: Path | None
    messages: tuple[str, ...]


# ---------------------------------------------------------------- small helpers
def format_utc(dt: datetime) -> str:
    """ISO 8601 UTC with a trailing "Z" at second precision, e.g. 2026-11-02T10:15:30Z."""
    if dt.tzinfo is None:
        raise ValueError("format_utc needs a timezone-aware datetime")
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_utc(text: str) -> datetime:
    """Parse an ISO 8601 timestamp with a time zone ("Z" or an offset) into aware UTC."""
    dt = datetime.fromisoformat(str(text).strip())
    if dt.tzinfo is None:
        raise ValueError(f"timestamp without a time zone: {text!r}")
    return dt.astimezone(timezone.utc)


def utc_now() -> str:
    """Current time as ``format_utc`` text."""
    return format_utc(datetime.now(timezone.utc))


def sha256_bytes(data: bytes) -> str:
    """Hex SHA-256 of a byte string."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Hex SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_note(note: str | None) -> str:
    """Normalise a free-text note to one line: control characters become spaces,
    runs of whitespace collapse to one space, and the ends are stripped."""
    if note is None:
        return ""
    text = "".join(" " if unicodedata.category(ch) in ("Cc", "Zl", "Zp") else ch for ch in str(note))
    return " ".join(text.split())


def partial_path_for(out_path: str | Path) -> Path:
    """Autosave path: ``items/verified_items.csv`` -> ``items/verified_items.partial.csv``."""
    return Path(out_path).with_suffix(".partial.csv")


def meta_path_for(out_path: str | Path) -> Path:
    """Meta sidecar path: ``items/verified_items.csv`` -> ``items/verified_items.meta.json``."""
    return Path(out_path).with_suffix(".meta.json")


def list_sidecar_paths(list_path: str | Path) -> tuple[Path, Path]:
    """Accepted sidecar names for a reading list, in lookup order:
    ``reading_list.meta.json`` and ``reading_list.csv.meta.json``."""
    path = Path(list_path)
    return path.with_suffix(".meta.json"), path.with_name(path.name + ".meta.json")


def _same_file(a: Path, b: Path) -> bool:
    return os.path.normcase(str(a.resolve())) == os.path.normcase(str(b.resolve()))


def _refuse_blinded(path: Path, what: str) -> None:
    """Refuse, before opening it, any file that could unblind the reader."""
    name = path.name.lower()
    if name in BLINDED_FILE_NAMES or path.suffix.lower() in BLINDED_SUFFIXES:
        raise ListError(
            f"{what} {path.name!r} is a file the blind reader must never open "
            "(it would reveal arms, labels or model output)"
        )


def _atomic_write(path: Path, data: bytes, attempts: int = 5) -> None:
    """Write ``data`` to ``path`` through a temporary file and an atomic replace.
    Retries briefly on PermissionError (Windows scanners and editors lock files)."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    delay = 0.05
    for attempt in range(attempts):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == attempts - 1:
                try:
                    tmp.unlink()
                except OSError:
                    pass
                raise
            time.sleep(delay)
            delay *= 2


def _publish_new(path: Path, data: bytes) -> None:
    """Create ``path`` with ``data`` atomically, refusing to replace an existing file."""
    if path.exists():
        raise OutputExistsError(f"refusing to overwrite an existing file: {path}")
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(tmp, path)          # atomic; fails if the target exists
    except FileExistsError:
        tmp.unlink()
        raise OutputExistsError(f"refusing to overwrite an existing file: {path}") from None
    except OSError:
        # Hard links unsupported on this file system: fall back to a checked rename
        # (on Windows os.rename itself refuses an existing target).
        if path.exists():
            tmp.unlink()
            raise OutputExistsError(f"refusing to overwrite an existing file: {path}") from None
        try:
            os.rename(tmp, path)
        except FileExistsError:
            tmp.unlink()
            raise OutputExistsError(f"refusing to overwrite an existing file: {path}") from None
    else:
        tmp.unlink()


def render_csv(columns: Sequence[str], rows: Iterable[Sequence[object]]) -> bytes:
    """Deterministic CSV bytes: UTF-8 without BOM, ``\\n`` line endings, minimal quoting."""
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
    writer.writerow(list(columns))
    for row in rows:
        writer.writerow(list(row))
    return buffer.getvalue().encode("utf-8")


# ---------------------------------------------------------------- reading lists
def read_list_seed(list_path: str | Path) -> int | None:
    """Seed of the list's random order from its sidecar (``seed`` key), else None.

    Both sidecar names of ``list_sidecar_paths`` are accepted; a malformed sidecar, a
    non-integer seed or two sidecars that disagree raise ``ListError``.
    """
    seeds: list[int] = []
    for candidate in list_sidecar_paths(list_path):
        if not candidate.is_file():
            continue
        try:
            meta = json.loads(candidate.read_text(encoding="utf-8-sig"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ListError(f"list sidecar {candidate.name!r} is not valid JSON: {exc}") from exc
        if not isinstance(meta, dict):
            raise ListError(f"list sidecar {candidate.name!r} must hold a JSON object")
        seed = meta.get("seed")
        if seed is None:
            continue
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ListError(f"list sidecar {candidate.name!r}: seed must be an integer, got {seed!r}")
        seeds.append(seed)
    if len(set(seeds)) > 1:
        raise ListError(f"the two list sidecars disagree on the seed: {seeds}")
    return seeds[0] if seeds else None


def resolve_image_path(raw: str, list_dir: Path) -> Path:
    """Resolve a list's ``image_path``: absolute paths as given; relative paths against the
    list's folder, then the current folder, then the repository root (first that exists;
    the list-folder candidate when none exists)."""
    path = Path(raw)
    if path.is_absolute():
        return path
    candidates = [Path(list_dir).resolve() / path, Path.cwd() / path, REPO_ROOT / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def load_reading_list(list_path: str | Path, limit: int | None = None) -> list[ReadingItem]:
    """Load an S10 reading list in file order (that order is the reading order).

    The file must be a .csv named neither like a blinded file nor like a tool output,
    with exactly the columns read_id, nih_filename, image_path (any column order).
    read_id must be 8-char uppercase hex; read_id and nih_filename must be unique; no
    field may be empty. Blank lines are ignored. ``limit`` keeps the first rows only.
    """
    path = Path(list_path)
    _refuse_blinded(path, "reading list")
    if path.name.lower() in OUTPUT_FILE_NAMES:
        raise ListError(f"{path.name!r} is a reading output, not a reading list")
    if path.suffix.lower() != ".csv":
        raise ListError(f"the reading list must be a .csv file: {path.name!r}")
    if limit is not None and limit < 1:
        raise ValueError("limit must be at least 1")
    if not path.is_file():
        raise ListError(f"reading list not found: {path}")
    try:
        text = path.read_bytes().decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ListError(f"reading list {path.name!r} is not UTF-8: {exc}") from exc
    reader = csv.reader(io.StringIO(text, newline=""))
    header = next(reader, None)
    if header is None:
        raise ListError(f"reading list {path.name!r} is empty")
    header = [cell.strip() for cell in header]
    extra = [col for col in header if col not in LIST_COLUMNS]
    if extra:
        raise ListError(
            f"reading list {path.name!r} has unexpected columns {extra}; a reading list has "
            f"exactly {list(LIST_COLUMNS)} (S10), and extra columns are refused so that "
            "nothing can unblind the reader"
        )
    missing = [col for col in LIST_COLUMNS if col not in header]
    if missing or len(header) != len(LIST_COLUMNS):
        raise ListError(f"reading list {path.name!r} must have the columns {list(LIST_COLUMNS)}; missing {missing}")
    col = {name: header.index(name) for name in LIST_COLUMNS}
    items: list[ReadingItem] = []
    seen_ids: set[str] = set()
    seen_files: set[str] = set()
    for line_no, row in enumerate(reader, start=2):
        if not row or all(not cell.strip() for cell in row):
            continue
        if len(row) != len(header):
            raise ListError(f"{path.name} line {line_no}: expected {len(header)} fields, found {len(row)}")
        read_id = row[col["read_id"]].strip()
        nih_filename = row[col["nih_filename"]].strip()
        raw_path = row[col["image_path"]].strip()
        if not READ_ID_PATTERN.fullmatch(read_id):
            raise ListError(f"{path.name} line {line_no}: read_id {read_id!r} is not 8-char uppercase hex")
        if not nih_filename or not raw_path:
            raise ListError(f"{path.name} line {line_no}: empty nih_filename or image_path")
        if read_id in seen_ids:
            raise ListError(f"{path.name} line {line_no}: duplicate read_id {read_id}")
        if nih_filename in seen_files:
            raise ListError(f"{path.name} line {line_no}: an image appears twice in the list")
        seen_ids.add(read_id)
        seen_files.add(nih_filename)
        items.append(ReadingItem(
            order_index=len(items),
            read_id=read_id,
            nih_filename=nih_filename,
            image_path=resolve_image_path(raw_path, path.parent),
        ))
    if not items:
        raise ListError(f"reading list {path.name!r} has no rows")
    return items[:limit] if limit is not None else items


# ---------------------------------------------------------------- meta files and the retest clock
def read_meta(out_path: str | Path) -> dict:
    """Read and validate the meta sidecar of a finalised output (keys exactly META_KEYS)."""
    meta_path = meta_path_for(out_path)
    if not meta_path.is_file():
        raise MetaError(f"meta file not found: {meta_path}")
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise MetaError(f"meta file {meta_path.name!r} is not valid JSON: {exc}") from exc
    if not isinstance(meta, dict) or set(meta) != set(META_KEYS):
        raise MetaError(f"meta file {meta_path.name!r} must have exactly the keys {list(META_KEYS)}")
    if not isinstance(meta["sha256"], str) or not SHA256_PATTERN.fullmatch(meta["sha256"]):
        raise MetaError("meta sha256 must be 64 lowercase hex characters")
    try:
        parse_utc(meta["saved_utc"])
    except (TypeError, ValueError) as exc:
        raise MetaError(f"meta saved_utc is not an ISO 8601 UTC time: {meta['saved_utc']!r}") from exc
    seed = meta["seed"]
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise MetaError(f"meta seed must be an integer or null, got {seed!r}")
    n = meta["n"]
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise MetaError(f"meta n must be a positive integer, got {n!r}")
    if meta["mode"] not in MODES:
        raise MetaError(f"meta mode must be one of {MODES}, got {meta['mode']!r}")
    if not isinstance(meta["tool_version"], str):
        raise MetaError("meta tool_version must be a string")
    return meta


def verify_output(out_path: str | Path) -> dict:
    """Check that a finalised output's bytes match the SHA-256 in its meta; return the meta."""
    meta = read_meta(out_path)
    actual = sha256_file(out_path)
    if actual != meta["sha256"]:
        raise MetaError(f"{Path(out_path).name}: SHA-256 {actual} does not match the meta file ({meta['sha256']})")
    return meta


def retest_earliest_start(first_read_out: str | Path) -> datetime:
    """Earliest retest start: the first read's saved_utc + 48 h (from its meta file only)."""
    meta = read_meta(first_read_out)
    if meta["mode"] != "read":
        raise MetaError(f"{Path(first_read_out).name} is a {meta['mode']!r} output, not the first read")
    return parse_utc(meta["saved_utc"]) + timedelta(hours=RETEST_MIN_HOURS)


def check_retest_gap(first_read_out: str | Path, now: datetime | None = None) -> datetime:
    """Raise ``RetestGapError`` before the earliest retest start; return that start otherwise."""
    earliest = retest_earliest_start(first_read_out)
    current = datetime.now(timezone.utc) if now is None else now
    if current.tzinfo is None:
        raise ValueError("now must be a timezone-aware datetime")
    if current < earliest:
        saved = earliest - timedelta(hours=RETEST_MIN_HOURS)
        raise RetestGapError(
            f"the retest may start only 48 h after the first read was saved ({format_utc(saved)}), "
            f"that is from {format_utc(earliest)}; it is now {format_utc(current)}"
        )
    return earliest


# ---------------------------------------------------------------- images (display helpers)
def to_display_gray(img: Image.Image) -> Image.Image:
    """8-bit grayscale ("L") copy for display; 16-bit and float images are min-max scaled."""
    if img.mode == "L":
        return img.copy()
    if img.mode.startswith("I") or img.mode == "F":
        arr = np.asarray(img, dtype=np.float64)
        lo, hi = float(arr.min()), float(arr.max())
        if hi <= lo:
            out = np.zeros(arr.shape, dtype=np.uint8)
        else:
            out = np.round((arr - lo) * (255.0 / (hi - lo))).astype(np.uint8)
        return Image.fromarray(out)
    return img.convert("L")


def load_display_image(path: str | Path) -> Image.Image:
    """Decode an image file into an 8-bit grayscale image (raises on missing or bad files)."""
    with Image.open(path) as img:
        img.load()
        return to_display_gray(img)


def apply_contrast(img: Image.Image, preset: str) -> Image.Image:
    """Contrast preset for display: "original", "autocontrast" (1% clipped at each end)
    or "equalised" (histogram equalisation)."""
    if preset not in CONTRAST_PRESETS:
        raise ValueError(f"unknown contrast preset {preset!r}; expected one of {CONTRAST_PRESETS}")
    gray = img if img.mode == "L" else to_display_gray(img)
    if preset == "original":
        return gray
    if preset == "autocontrast":
        return ImageOps.autocontrast(gray, cutoff=AUTOCONTRAST_CUTOFF)
    return ImageOps.equalize(gray)


def next_contrast(preset: str) -> str:
    """The preset after ``preset`` in the cycle original -> autocontrast -> equalised."""
    if preset not in CONTRAST_PRESETS:
        raise ValueError(f"unknown contrast preset {preset!r}")
    return CONTRAST_PRESETS[(CONTRAST_PRESETS.index(preset) + 1) % len(CONTRAST_PRESETS)]


def fit_size(width: int, height: int, box_width: int, box_height: int) -> tuple[int, int]:
    """Largest size with the image's aspect ratio that fits the box (up- or down-scaling)."""
    if width <= 0 or height <= 0:
        raise ValueError("image size must be positive")
    box_w, box_h = max(1, int(box_width)), max(1, int(box_height))
    scale = min(box_w / width, box_h / height)
    return max(1, min(box_w, int(round(width * scale)))), max(1, min(box_h, int(round(height * scale))))


def fit_image(img: Image.Image, box_width: int, box_height: int) -> Image.Image:
    """Resize ``img`` to ``fit_size`` (Lanczos); returns ``img`` itself when it already fits exactly."""
    size = fit_size(img.width, img.height, box_width, box_height)
    if size == img.size:
        return img
    return img.resize(size, Image.Resampling.LANCZOS)


# ---------------------------------------------------------------- the session
class ReadingSession:
    """One blind reading (or retest) of a reading list.

    Flow: ``select`` an answer, ``toggle_flag`` / ``set_note`` as needed, then ``confirm``
    (or call ``record`` directly). Every confirmation autosaves the partial file and moves
    to the next unanswered item. ``back`` moves one item back (one step only; confirm it
    again to move on). ``skip_remaining`` marks every unanswered item as skip.
    ``finalise`` writes the output and its meta file once every item has a response.

    Constructing a session refuses an output path whose output or meta file already
    exists, and resumes from ``<out stem>.partial.csv`` when present (the partial must
    match the list item by item). ``clock`` (monotonic seconds) and ``now`` (aware UTC
    datetime) are injectable for tests.
    """

    def __init__(
        self,
        list_path: str | Path,
        out_path: str | Path,
        mode: str = "read",
        *,
        limit: int | None = None,
        clock: Callable[[], float] = time.monotonic,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self._list_path = Path(list_path)
        _refuse_blinded(self._list_path, "reading list")   # before any file-system call on it
        self._out_path = Path(out_path)
        self._mode = mode
        self._clock = clock
        self._now = now if now is not None else (lambda: datetime.now(timezone.utc))
        if self._out_path.suffix.lower() != ".csv":
            raise ValueError(f"the output must be a .csv file: {self._out_path.name!r}")
        for own in (self._out_path, self.partial_path, self.meta_path):
            if _same_file(own, self._list_path):
                raise ValueError("the output, partial or meta path would overwrite the reading list")
        out_exists, meta_exists = self._out_path.exists(), self.meta_path.exists()
        if out_exists and meta_exists:
            raise OutputExistsError(
                f"{self._out_path} is already finalised: a finalised reading is never overwritten; "
                "choose a new --out path"
            )
        if out_exists:
            hint = (
                " The autosave file is still there, so a finalisation was probably interrupted: move the "
                "output file away and restart the tool to finalise again from the autosave file."
                if self.partial_path.exists() else ""
            )
            raise OutputExistsError(f"{self._out_path} already exists without its meta file; it is never overwritten.{hint}")
        if meta_exists:
            raise OutputExistsError(
                f"{self.meta_path} exists without its output: a finalisation was interrupted or a "
                "file was removed; check the folder by hand before reading again"
            )
        self._items: tuple[ReadingItem, ...] = tuple(load_reading_list(self._list_path, limit))
        self._seed = read_list_seed(self._list_path)
        self._responses: dict[int, Response] = {}
        self._pending: dict[int, float] = {}
        self._unreadable: set[int] = set()
        self._partial_sha: str | None = None
        self._finalised = False
        self._stepped_back = False
        self._position = 0
        self._visit_start: float | None = None
        self._draft_response: str | None = None
        self._draft_flags: set[str] = set()
        self._draft_note = ""
        self._resumed = self._load_partial()
        self._enter(self._first_unanswered())

    # ------------------------------------------------------------ properties
    @property
    def n(self) -> int:
        """Number of items in this session."""
        return len(self._items)

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def seed(self) -> int | None:
        """Seed of the list order from the list's sidecar, or None."""
        return self._seed

    @property
    def list_path(self) -> Path:
        return self._list_path

    @property
    def out_path(self) -> Path:
        return self._out_path

    @property
    def partial_path(self) -> Path:
        return partial_path_for(self._out_path)

    @property
    def meta_path(self) -> Path:
        return meta_path_for(self._out_path)

    @property
    def position(self) -> int:
        """0-based index of the current item (equals n when every item is answered)."""
        return self._position

    @property
    def complete(self) -> bool:
        return self._position >= self.n

    @property
    def finalised(self) -> bool:
        return self._finalised

    @property
    def resumed_count(self) -> int:
        """Number of responses loaded from the partial file at start."""
        return self._resumed

    @property
    def items(self) -> tuple[ReadingItem, ...]:
        """The items in reading order (equal to the list's file order)."""
        return self._items

    @property
    def read_order(self) -> tuple[str, ...]:
        return tuple(item.read_id for item in self._items)

    @property
    def responses(self) -> dict[int, Response]:
        """Copy of the recorded responses keyed by order_index."""
        return dict(self._responses)

    @property
    def current_item(self) -> ReadingItem | None:
        return None if self.complete else self._items[self._position]

    @property
    def can_go_back(self) -> bool:
        return not self._finalised and not self._stepped_back and self._position > 0

    def unanswered(self) -> list[int]:
        """Indices of items with no response yet."""
        return [i for i in range(self.n) if i not in self._responses]

    # ------------------------------------------------------------ images
    def _index(self, index: int | None) -> int:
        if index is None:
            if self.complete:
                raise ReadingError("there is no current item: every item is answered")
            return self._position
        if not 0 <= index < self.n:
            raise IndexError(f"item index {index} out of range 0..{self.n - 1}")
        return index

    def image_problem(self, index: int | None = None) -> str | None:
        """"image file missing" / "image file unreadable" for an item, or None (no paths)."""
        i = self._index(index)
        if not self._items[i].image_path.is_file():
            return "image file missing"
        if i in self._unreadable:
            return "image file unreadable"
        return None

    def missing_images(self) -> list[str]:
        """read_ids whose image file does not exist."""
        return [item.read_id for item in self._items if not item.image_path.is_file()]

    def load_image(self, index: int | None = None) -> Image.Image:
        """Decode an item's image (8-bit grayscale). Raises ``ImageUnavailableError``,
        whose message names no path; an undecodable file is remembered so that only
        "skip" can be recorded for it."""
        i = self._index(index)
        path = self._items[i].image_path
        if not path.is_file():
            raise ImageUnavailableError("image file missing")
        try:
            img = load_display_image(path)
        except Exception as exc:  # any decoder failure means the reader cannot see the image
            self._unreadable.add(i)
            raise ImageUnavailableError("image file unreadable") from exc
        self._unreadable.discard(i)
        return img

    # ------------------------------------------------------------ draft (select, flags, note)
    def _require_open(self) -> None:
        if self._finalised:
            raise SessionClosedError("the session is finalised; start a new one with a new output path")

    def _require_current(self) -> int:
        self._require_open()
        if self.complete:
            raise ReadingError("every item is answered: finalise, or go back one item")
        return self._position

    def select(self, response: str) -> None:
        """Set the draft answer of the current item ("present", "absent" or "skip")."""
        i = self._require_current()
        response = self._validate_response(response)
        self._check_image_for(i, response)
        self._draft_response = response

    def toggle_flag(self, flag: str) -> bool:
        """Toggle a draft flag of the current item; returns its new state."""
        self._require_current()
        if flag not in FLAGS:
            raise ResponseError(f"unknown flag {flag!r}; expected one of {FLAGS}")
        if flag in self._draft_flags:
            self._draft_flags.discard(flag)
            return False
        self._draft_flags.add(flag)
        return True

    def set_note(self, note: str | None) -> None:
        """Set the draft note of the current item (normalised to one line)."""
        self._require_current()
        self._draft_note = clean_note(note)

    def clear_draft(self) -> None:
        """Forget the draft answer, flags and note of the current item."""
        self._require_current()
        self._draft_response, self._draft_flags, self._draft_note = None, set(), ""

    def confirm(self) -> int:
        """Record the draft of the current item (see ``record``); returns its index."""
        self._require_current()
        if self._draft_response is None:
            raise ResponseError("choose present, absent or skip (P, A or S) before confirming")
        return self.record(self._draft_response, flags=self._draft_flags, note=self._draft_note)

    def restart_timer(self) -> None:
        """Restart the current item's clock (for example after a start dialog)."""
        if not self.complete:
            self._visit_start = self._clock()

    # ------------------------------------------------------------ recording
    @staticmethod
    def _validate_response(response: str) -> str:
        if response not in RESPONSES:
            raise ResponseError(f"response must be one of {RESPONSES}, got {response!r}")
        return response

    def _check_image_for(self, index: int, response: str) -> None:
        problem = self.image_problem(index)
        if problem is not None and response != "skip":
            raise ResponseError(f"{problem}: this image can only be skipped (S)")

    def _stamp(self) -> str:
        return format_utc(self._now())

    def record(self, response: str, flags: Iterable[str] = (), note: str | None = "") -> int:
        """Record a response for the current item, autosave, and move to the next
        unanswered item. Returns the recorded index.

        The item's ``seconds`` adds the on-screen time of every visit so far; the
        ``timestamp_utc`` is the time of this confirmation. Present and absent are refused
        when the image file is missing or unreadable (skip gets the note "image file
        missing" or "image file unreadable" when no note is given). If the autosave fails,
        ``AutosaveError`` is raised after the response is kept in memory; the next
        response or ``save_partial`` retries the write.
        """
        i = self._require_current()
        response = self._validate_response(response)
        flag_set = frozenset(flags)
        unknown = sorted(flag_set - set(FLAGS))
        if unknown:
            raise ResponseError(f"unknown flags {unknown}; expected a subset of {FLAGS}")
        text = clean_note(note)
        problem = self.image_problem(i)
        self._check_image_for(i, response)
        if problem is not None and not text:
            text = problem
        start = self._visit_start if self._visit_start is not None else self._clock()
        elapsed = self._pending.pop(i, 0.0) + max(0.0, self._clock() - start)
        previous = self._responses.get(i)
        total = (previous.seconds if previous is not None else 0.0) + elapsed
        self._responses[i] = Response(response, flag_set, text, round(total, 3), self._stamp())
        self._stepped_back = False
        self._enter(self._first_unanswered())
        self.save_partial()
        return i

    def back(self) -> bool:
        """Go back one item (one step only). The previous answer becomes the draft;
        confirm it (changed or not) to move on. Returns False when not allowed."""
        self._require_open()
        if not self.can_go_back:
            return False
        self._leave_current()
        self._enter(self._position - 1)
        self._stepped_back = True
        return True

    def skip_remaining(self, note: str = SKIP_REMAINING_NOTE) -> int:
        """Mark every unanswered item as skip (ending the session early); autosaves.
        Returns the number of items marked."""
        self._require_open()
        todo = self.unanswered()
        if not todo:
            return 0
        text = clean_note(note)
        stamp = self._stamp()
        self._leave_current()
        for i in todo:
            self._responses[i] = Response("skip", frozenset(), text, round(self._pending.pop(i, 0.0), 3), stamp)
        self._stepped_back = False
        self._enter(self._first_unanswered())
        self.save_partial()
        return len(todo)

    def _first_unanswered(self) -> int:
        for i in range(self.n):
            if i not in self._responses:
                return i
        return self.n

    def _leave_current(self) -> None:
        if self._position < self.n and self._visit_start is not None:
            spent = max(0.0, self._clock() - self._visit_start)
            self._pending[self._position] = self._pending.get(self._position, 0.0) + spent
        self._visit_start = None

    def _enter(self, index: int) -> None:
        self._position = index
        if index < self.n:
            self._visit_start = self._clock()
            existing = self._responses.get(index)
            self._draft_response = existing.response if existing else None
            self._draft_flags = set(existing.flags) if existing else set()
            self._draft_note = existing.note if existing else ""
        else:
            self._visit_start = None
            self._draft_response, self._draft_flags, self._draft_note = None, set(), ""

    # ------------------------------------------------------------ files
    def _row(self, index: int) -> list[str]:
        item, resp = self._items[index], self._responses[index]
        return [
            item.read_id, item.nih_filename, resp.response,
            *("1" if flag in resp.flags else "0" for flag in FLAGS),
            resp.note, f"{resp.seconds:.3f}", str(index), resp.timestamp_utc,
        ]

    def save_partial(self) -> Path:
        """Rewrite the partial file with every recorded response (atomic replace).

        Raises ``ConcurrentWriteError`` when the file on disk is not the one this session
        last wrote or loaded (another window on the same output), and ``AutosaveError``
        when the write fails.
        """
        self._require_open()
        data = render_csv(OUTPUT_COLUMNS, (self._row(i) for i in sorted(self._responses)))
        path = self.partial_path
        if path.exists():
            try:
                on_disk = sha256_file(path)
            except OSError as exc:
                raise AutosaveError(f"cannot read the partial file {path}: {exc}") from exc
            if on_disk != self._partial_sha:
                raise ConcurrentWriteError(
                    f"{path} was changed by another program or tool window since this session "
                    "last saved it; close the other window and restart the tool to resume"
                )
        try:
            _atomic_write(path, data)
        except OSError as exc:
            raise AutosaveError(
                f"autosave to {path} failed ({exc}); the answer is kept in memory and saving "
                "is retried after the next answer"
            ) from exc
        self._partial_sha = sha256_bytes(data)
        return path

    def _load_partial(self) -> int:
        path = self.partial_path
        if not path.exists():
            return 0
        data = path.read_bytes()
        name = path.name
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ResumeError(f"{name} is not UTF-8 ({exc}); move it away to start over") from exc
        reader = csv.DictReader(io.StringIO(text, newline=""))
        fields = reader.fieldnames
        if not fields or sorted(fields) != sorted(OUTPUT_COLUMNS):
            raise ResumeError(f"{name} does not have the S11 columns {list(OUTPUT_COLUMNS)}; move it away to start over")
        loaded: dict[int, Response] = {}
        for line_no, row in enumerate(reader, start=2):
            where = f"{name} line {line_no}"
            if None in row or any(value is None for value in row.values()):
                raise ResumeError(f"{where}: wrong number of fields")
            try:
                index = int(row["order_index"])
            except ValueError:
                raise ResumeError(f"{where}: order_index {row['order_index']!r} is not an integer") from None
            if not 0 <= index < self.n:
                raise ResumeError(f"{where}: order_index {index} is outside this list (n = {self.n})")
            if index in loaded:
                raise ResumeError(f"{where}: order_index {index} appears twice")
            item = self._items[index]
            if row["read_id"] != item.read_id or row["nih_filename"] != item.nih_filename:
                raise ResumeError(
                    f"{where}: the partial file does not match this reading list at position {index + 1} "
                    f"(read_id {row['read_id']} vs {item.read_id}); use the list it was made with, "
                    "or move the partial file away to start over"
                )
            if row["response"] not in RESPONSES:
                raise ResumeError(f"{where}: response {row['response']!r} is not one of {RESPONSES}")
            flags: set[str] = set()
            for flag, column in zip(FLAGS, FLAG_COLUMNS):
                if row[column] not in ("0", "1"):
                    raise ResumeError(f"{where}: {column} must be 0 or 1, got {row[column]!r}")
                if row[column] == "1":
                    flags.add(flag)
            try:
                seconds = float(row["seconds"])
            except ValueError:
                raise ResumeError(f"{where}: seconds {row['seconds']!r} is not a number") from None
            if not math.isfinite(seconds) or seconds < 0:
                raise ResumeError(f"{where}: seconds must be a finite number >= 0, got {row['seconds']!r}")
            try:
                parse_utc(row["timestamp_utc"])
            except ValueError:
                raise ResumeError(f"{where}: timestamp_utc {row['timestamp_utc']!r} is not ISO 8601 UTC") from None
            loaded[index] = Response(row["response"], frozenset(flags), clean_note(row["note"]),
                                     round(seconds, 3), row["timestamp_utc"])
        self._responses = loaded
        self._partial_sha = sha256_bytes(data)
        return len(loaded)

    def finalise(self) -> FinaliseResult:
        """Write the output (S11, every item in reading order) and its meta file, then
        remove the partial file. Refuses while items are unanswered (mark them skip with
        ``skip_remaining`` to end early) and never overwrites an existing output or meta."""
        self._require_open()
        missing = self.unanswered()
        if missing:
            raise IncompleteError(
                f"{len(missing)} of {self.n} images have no answer yet (first: image {missing[0] + 1}); "
                "answer them, or mark the remaining images as skip to end early"
            )
        if self._out_path.exists() or self.meta_path.exists():
            raise OutputExistsError(f"{self._out_path} or its meta file already exists; nothing was overwritten")
        data = render_csv(OUTPUT_COLUMNS, (self._row(i) for i in range(self.n)))
        digest = sha256_bytes(data)
        saved = self._stamp()
        meta = {"sha256": digest, "saved_utc": saved, "seed": self._seed, "n": self.n,
                "mode": self._mode, "tool_version": TOOL_VERSION}
        _publish_new(self._out_path, data)
        try:
            written = sha256_file(self._out_path)
            if written != digest:
                raise ReadingError(f"verification failed: {self._out_path} hashes to {written}, expected {digest}")
            _publish_new(self.meta_path, (json.dumps(meta, indent=2) + "\n").encode("utf-8"))
        except BaseException:
            # Roll back so that no output exists without its meta; the partial file still holds every answer.
            try:
                self._out_path.unlink()
            except OSError:
                pass
            raise
        self._finalised = True
        partial_removed = True
        try:
            self.partial_path.unlink(missing_ok=True)
        except OSError:
            partial_removed = False
        earliest = None
        if self._mode == "read":
            earliest = format_utc(parse_utc(saved) + timedelta(hours=RETEST_MIN_HOURS))
        return FinaliseResult(
            out_path=self._out_path, meta_path=self.meta_path, sha256=digest, saved_utc=saved,
            n=self.n, mode=self._mode,
            n_skip=sum(r.response == "skip" for r in self._responses.values()),
            n_flagged=sum(bool(r.flags) for r in self._responses.values()),
            partial_removed=partial_removed, partial_path=self.partial_path,
            earliest_retest_utc=earliest,
        )

    # ------------------------------------------------------------ what a front end may show
    def view(self) -> ViewState:
        """Display state: "k of N", the read_id and the reader's own draft; no filenames."""
        complete = self.complete
        return ViewState(
            position=self._position,
            n=self.n,
            read_id="" if complete else self._items[self._position].read_id,
            complete=complete,
            response=self._draft_response,
            flags=frozenset(self._draft_flags),
            note=self._draft_note,
            answered=len(self._responses),
            stepped_back=self._stepped_back,
            can_go_back=self.can_go_back,
            image_problem=None if complete else self.image_problem(),
            mode=self._mode,
            finalised=self._finalised,
        )

    def summary_lines(self) -> list[str]:
        """Start-of-session summary (list and output names, counts; no image names)."""
        lines = [
            f"Mode: {self._mode}",
            f"Reading list: {self._list_path.name} ({self.n} images)",
            f"Output: {self._out_path}",
            f"Order seed: {self._seed}" if self._seed is not None
            else "Order seed: not recorded (no list sidecar with a seed)",
        ]
        if self._resumed:
            where = "all answered" if self.complete else f"continuing at image {self._position + 1}"
            lines.append(f"Resuming: {len(self._responses)} of {self.n} answered; {where}")
        else:
            lines.append(f"Starting at image 1 of {self.n}")
        missing = self.missing_images()
        if missing:
            lines.append(f"WARNING: {len(missing)} image file(s) are missing; those images can only be skipped")
        return lines


# ---------------------------------------------------------------- practice runs and run planning
def check_practice_list(list_path: str | Path) -> None:
    """Refuse the study reading or retest list for a practice run (it would show study
    images before the blind read)."""
    name = Path(list_path).name.lower()
    if name in STUDY_LIST_NAMES:
        raise ListError(
            f"{Path(list_path).name!r} holds study images: a practice run must not show them before "
            "the blind read; use the practice list of dev-pool images (or no --list for a synthetic demo)"
        )


def make_practice_dir(base: str | Path | None = None) -> Path:
    """Create a new, empty temporary folder for a practice run."""
    return Path(tempfile.mkdtemp(prefix="ett_read_practice_", dir=None if base is None else str(base)))


def _demo_font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    try:
        return ImageFont.load_default(size=size)
    except (OSError, TypeError, AttributeError):
        return ImageFont.load_default()


def _demo_image(rng: np.random.Generator, side: int, with_tube: bool) -> Image.Image:
    """Synthetic grey picture with a dark midline band and, optionally, a bright stripe.
    It only resembles a radiograph enough to practise the keys."""
    yy, xx = np.mgrid[0:side, 0:side] / float(side)
    base = 0.30 + 0.30 * np.exp(-((xx - 0.5) ** 2) / 0.01)
    for centre in (0.30, 0.70):
        base -= 0.20 * np.exp(-(((xx - centre) / 0.15) ** 2 + ((yy - 0.58) / 0.28) ** 2))
    base[(np.abs(xx - 0.5) < 0.025) & (yy < 0.45)] -= 0.15
    if with_tube:
        base[(np.abs(xx - 0.5) < 0.006) & (yy < 0.38)] += 0.45
    pixels = np.clip((base + rng.normal(0.0, 0.03, size=base.shape)) * 255.0, 0, 255).astype(np.uint8)
    img = Image.fromarray(pixels)
    ImageDraw.Draw(img).text((8, 8), "SYNTHETIC DEMO - not a radiograph", fill=255,
                             font=_demo_font(max(12, side // 32)))
    return img


def write_demo_list(folder: str | Path, n: int = PRACTICE_N, seed: int | None = None, side: int = 512) -> Path:
    """Write ``n`` synthetic demo images, ``demo_list.csv`` (S10 columns, relative image
    paths) and its sidecar ``demo_list.meta.json`` into ``folder``; return the list path.
    Deterministic for a given seed (default ``derive_seed("reading")``)."""
    if n < 1:
        raise ValueError("n must be at least 1")
    folder = Path(folder)
    seed = derive_seed("reading") if seed is None else int(seed)
    rng = np.random.default_rng(seed)
    image_dir = folder / "demo_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    with_tube = rng.permutation(np.array([True] * (n // 2) + [False] * (n - n // 2)))
    read_ids: list[str] = []
    while len(read_ids) < n:
        candidate = f"{int(rng.integers(0, 2 ** 32)):08X}"
        if candidate not in read_ids:
            read_ids.append(candidate)
    rows = []
    for j in range(n):
        name = f"demo_{j + 1:02d}.png"
        _demo_image(rng, side, bool(with_tube[j])).save(image_dir / name, format="PNG")
        rows.append((read_ids[j], name, f"demo_images/{name}"))
    list_path = folder / "demo_list.csv"
    list_path.write_bytes(render_csv(LIST_COLUMNS, rows))
    sidecar = list_path.with_suffix(".meta.json")
    sidecar.write_text(json.dumps({"seed": seed, "n": n, "synthetic": True}, indent=2) + "\n", encoding="utf-8")
    return list_path


def plan_run(
    list_path: str | Path | None,
    out_path: str | Path | None,
    mode: str = "read",
    *,
    practice: bool = False,
    first_read: str | Path | None = None,
    now: datetime | None = None,
    practice_base: str | Path | None = None,
) -> RunPlan:
    """Resolve the command-line choices into a ``RunPlan``.

    Practice: refuses the study lists, creates a new temporary folder, uses the first
    ``PRACTICE_N`` items of ``list_path`` (or a synthetic demo list written into that
    folder when ``list_path`` is None) and writes the output there (named after
    ``out_path`` when given). Otherwise ``list_path`` and ``out_path`` are required; in
    retest mode ``first_read`` (the finalised first-read output) enables the 48-hour
    check, which raises ``RetestGapError`` when it is too early.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if first_read is not None and mode != "retest":
        raise ValueError("--first-read applies to --mode retest only")
    messages: list[str] = []
    if practice:
        if list_path is not None:      # refuse before any folder is created
            check_practice_list(list_path)
            _refuse_blinded(Path(list_path), "reading list")
            if not Path(list_path).is_file():
                raise ListError(f"reading list not found: {list_path}")
        folder = make_practice_dir(practice_base)
        if list_path is None:
            list_path = write_demo_list(folder)
            messages.append("Practice with synthetic demo images (not radiographs): for learning the keys only.")
        name = Path(out_path).name if out_path is not None else f"practice_{mode}.csv"
        out = folder / Path(name).with_suffix(".csv").name
        messages.append(f"PRACTICE run: answers go to the temporary folder {folder} and are not part of the study.")
        return RunPlan(Path(list_path), out, mode, PRACTICE_N, folder, tuple(messages))
    if list_path is None or out_path is None:
        raise ValueError("--list and --out are required unless --practice is given")
    if mode == "retest":
        if first_read is None:
            messages.append(
                "WARNING: no --first-read given, so the tool cannot check that 48 h have passed "
                "since the first read was saved (PROTOCOL section 2)."
            )
        else:
            earliest = check_retest_gap(first_read, now=now)
            messages.append(
                f"48-hour check passed: the first read was saved at "
                f"{format_utc(earliest - timedelta(hours=RETEST_MIN_HOURS))}; the retest may start from "
                f"{format_utc(earliest)}."
            )
    return RunPlan(Path(list_path), Path(out_path), mode, None, None, tuple(messages))
