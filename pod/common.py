"""Shared helpers for the GPU-pod scripts (``run_vlm.py``, ``smoke_test.py``, ``run_medsiglip.py``).

Data boundary (PROTOCOL.md section 9): the pod holds NIH ChestX-ray14 PNGs and prompt text only.
Its only item file is ``pod_items.csv`` (schema S6: ``item_id``, ``nih_filename``, ``prompt_set``,
``smoke``); ``read_pod_items`` refuses any other column, so an arm, a label, a uid or a patient id
cannot reach the pod by accident. Arms and labels are joined locally at scoring time.

This module holds everything that does not need a model: the pod item list, the task plan
(item x prompt x condition), image loading and the generated BLIND-grey PNG, the yes/no token-id
lists, the first-token and full-sequence signal arithmetic (through ``cxrdev.logits``), the S7
record builder and validator, the resume check, and the smoke-test coverage decision and report.
It imports the standard library, numpy, pillow and the pod-safe ``cxrdev`` modules (``constants``,
``prompts``, ``parse``, ``logits``, ``fileio``; none of them needs pandas or scipy). torch and
huggingface_hub are imported inside the few functions that need them.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
import statistics
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from cxrdev import constants
from cxrdev.fileio import read_jsonl, sha256_bytes, utc_now
from cxrdev.logits import first_token_signal, log_softmax, single_token_ids
from cxrdev.parse import ANSWERS, is_truncated
from cxrdev.prompts import prompt_sha256, prompt_text

# ----------------------------------------------------------------------------- schema S6 / S7
POD_ITEM_COLUMNS: tuple[str, ...] = ("item_id", "nih_filename", "prompt_set", "smoke")

#: Exact field order of one run record (PIPELINE_SPEC S7).
S7_FIELDS: tuple[str, ...] = (
    "run_id", "model_key", "model_id", "revision", "transformers_version", "torch_version",
    "dtype", "device", "item_id", "nih_filename", "image_sha256", "condition", "prompt_id",
    "prompt_sha256", "prompt_text", "visual_tokens", "seed", "max_new_tokens", "raw_output",
    "n_generated_tokens", "ended_with_eos", "parsed", "parser_version", "logp_yes", "logp_no",
    "logp_abstain", "z", "coverage", "signal", "yes_ids", "no_ids", "wall_s", "timestamp_utc",
)
_STR_FIELDS = frozenset({
    "run_id", "model_key", "model_id", "revision", "transformers_version", "torch_version", "dtype",
    "device", "item_id", "nih_filename", "image_sha256", "condition", "prompt_id", "prompt_sha256",
    "prompt_text", "raw_output", "parsed", "parser_version", "signal", "timestamp_utc",
})
_INT_FIELDS = frozenset({"visual_tokens", "seed", "max_new_tokens", "n_generated_tokens"})
_FLOAT_FIELDS = frozenset({"logp_yes", "logp_no", "logp_abstain", "z", "coverage", "wall_s"})
_ID_LIST_FIELDS = frozenset({"yes_ids", "no_ids"})

#: S9 detector score columns (MedSigLIP, written by run_medsiglip.py). ``dtype`` (the torch dtype the
#: scorer ran in, "float32" or "bfloat16") was added after PIPELINE_SPEC v1.0 so the report can quote it
#: (numbers.json key ``run.medsiglip.dtype``); readers treat it as optional.
S9_COLUMNS: tuple[str, ...] = (
    "item_id", "nih_filename", "s", "logit_with", "logit_without", "model_id", "revision", "dtype",
)

# ----------------------------------------------------------------------------- conditions and prompts
CONDITION_IMAGE = "image"
CONDITION_BLIND_TEXT = "BLIND-text"
CONDITION_BLIND_GREY = "BLIND-grey"
CONDITIONS: tuple[str, ...] = (CONDITION_IMAGE, CONDITION_BLIND_TEXT, CONDITION_BLIND_GREY, "BLIND-demo")
BLIND_TEXT_ID = "BLIND-TEXT"
BLIND_GREY_ID = "BLIND-GREY"
SMOKE_PROMPT = "P0"
#: prompt_set -> prompt ids (PIPELINE_SPEC S6: "full" = P0, P1, P2hi for A/B/U; "p0" = P0 for DEV).
PROMPT_SETS: dict[str, tuple[str, ...]] = {"full": tuple(constants.CORE_PROMPTS), "p0": tuple(constants.DEV_PROMPTS)}
#: BLIND-text and BLIND-grey run the CORE prompts (PROTOCOL.md section 4).
BLIND_PROMPTS: tuple[str, ...] = tuple(constants.CORE_PROMPTS)
DEVICE_KEY = "ett"

RESUME_KEY_FIELDS: tuple[str, ...] = ("item_id", "prompt_id", "condition")
SIGNALS: tuple[str, ...] = ("first_token", "full_sequence")
SIGNAL_CHOICES: tuple[str, ...] = ("auto",) + SIGNALS
#: Candidate completions scored by the full-sequence fallback (PROTOCOL.md section 3).
FULL_SEQUENCE_ANSWERS: tuple[str, str] = ("Yes", "No")
#: Token that closes an assistant turn in each model family's chat template.
END_OF_TURN_TOKENS: dict[str, str] = {"qwen3vl": "<|im_end|>", "qwen25vl": "<|im_end|>", "gemma3": "<end_of_turn>"}
QWEN_FAMILIES: frozenset[str] = frozenset({"qwen3vl", "qwen25vl"})

ITEM_ID_RE = re.compile(r"^[0-9a-f]{10}$")
NIH_FILENAME_RE = re.compile(r"^\d{8}_\d{3}\.png$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")

# ----------------------------------------------------------------------------- exit codes
EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2
EXIT_TOKEN_ASSERTION = 3
EXIT_RESUME_CONFLICT = 4


# ----------------------------------------------------------------------------- errors
class PodItemsError(ValueError):
    """``pod_items.csv`` is malformed or carries a column outside schema S6."""


class VisualTokenAssertionError(RuntimeError):
    """The number of visual tokens differs from the registered count (PROTOCOL.md section 3)."""


class ResumeConflictError(RuntimeError):
    """An existing JSONL file holds records from a different run configuration."""


class SmokeReportError(RuntimeError):
    """``smoke_report.json`` is missing, incomplete, or records a failed smoke test."""


# ----------------------------------------------------------------------------- pod items and tasks
@dataclass(frozen=True)
class PodItem:
    """One row of ``pod_items.csv`` (schema S6)."""

    item_id: str
    nih_filename: str
    prompt_set: str
    smoke: int


@dataclass(frozen=True)
class Task:
    """One unit of work: an item (or a blind row) under one prompt and one condition."""

    item_id: str
    nih_filename: str
    condition: str
    prompt_id: str

    @property
    def key(self) -> tuple[str, str, str]:
        """The resume key (item_id, prompt_id, condition)."""
        return (self.item_id, self.prompt_id, self.condition)


def read_pod_items(path: str | Path) -> list[PodItem]:
    """Read and validate ``pod_items.csv`` (schema S6), keeping the file's row order.

    The header must be exactly the four S6 columns (in any order); any other column, for example
    ``arm``, ``uid`` or a label column, raises PodItemsError, because nothing but opaque ids and
    NIH filenames may reach the pod. ``item_id`` must be 10-character lowercase hex and unique,
    ``nih_filename`` must look like ``00000001_000.png``, ``prompt_set`` must be "full" or "p0",
    ``smoke`` must be 0 or 1, and smoke items must be dev-pool items (``prompt_set`` "p0").
    """
    p = Path(path)
    with open(p, encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        header = [h.strip() for h in (reader.fieldnames or [])]
        if not header:
            raise PodItemsError(f"{p}: empty file or no header")
        extra = [h for h in header if h not in POD_ITEM_COLUMNS]
        missing = [c for c in POD_ITEM_COLUMNS if c not in header]
        if extra:
            raise PodItemsError(
                f"{p}: column(s) {extra} are not allowed on the pod; pod_items.csv carries only "
                f"{list(POD_ITEM_COLUMNS)} (PROTOCOL.md section 9)"
            )
        if missing:
            raise PodItemsError(f"{p}: missing column(s) {missing}")
        if len(set(header)) != len(header):
            raise PodItemsError(f"{p}: duplicate column names in {header}")
        rows = list(reader)
    items: list[PodItem] = []
    seen: set[str] = set()
    problems: list[str] = []
    for lineno, raw in enumerate(rows, start=2):
        if None in raw:  # more cells than header fields
            problems.append(f"line {lineno}: more cells than columns")
            continue
        row = {(k or "").strip(): (v or "").strip() for k, v in raw.items()}
        item_id, fname, pset, smoke = (row.get(c, "") for c in POD_ITEM_COLUMNS)
        row_problems: list[str] = []
        if not ITEM_ID_RE.match(item_id):
            row_problems.append(f"item_id {item_id!r} is not 10-character lowercase hex")
        elif item_id in seen:
            row_problems.append(f"duplicate item_id {item_id!r}")
        if not NIH_FILENAME_RE.match(fname):
            row_problems.append(f"nih_filename {fname!r} is not an NIH PNG name")
        if pset not in PROMPT_SETS:
            row_problems.append(f"prompt_set {pset!r} not in {sorted(PROMPT_SETS)}")
        if smoke not in ("0", "1"):
            row_problems.append(f"smoke {smoke!r} must be 0 or 1")
        elif smoke == "1" and pset != "p0":
            row_problems.append(f"smoke item {item_id!r} must be a dev-pool item (prompt_set p0)")
        seen.add(item_id)
        if row_problems:
            problems.extend(f"line {lineno}: {msg}" for msg in row_problems)
        else:
            items.append(PodItem(item_id, fname, pset, int(smoke)))
    if problems:
        shown = "; ".join(problems[:10])
        raise PodItemsError(f"{p}: {len(problems)} problem(s): {shown}")
    if not items:
        raise PodItemsError(f"{p}: no rows")
    return items


def plan_tasks(items: Sequence[PodItem], mode: str, limit: int | None = None) -> list[Task]:
    """Expand pod items into tasks, in file order.

    ``mode`` "items": every item under the prompts of its ``prompt_set`` ("full" -> P0, P1, P2hi;
    "p0" -> P0), condition "image". ``mode`` "smoke": the smoke items (``smoke == 1``) under P0,
    condition "image". ``mode`` "blind": BLIND-text and BLIND-grey rows under P0, P1 and P2hi (no
    item is used; ``items`` and ``limit`` are ignored). ``limit`` keeps the first N items (after
    the smoke filter) for quick tests.
    """
    if limit is not None and limit < 1:
        raise ValueError(f"limit must be a positive integer, got {limit}")
    if mode == "blind":
        tasks = [Task(BLIND_TEXT_ID, "", CONDITION_BLIND_TEXT, pid) for pid in BLIND_PROMPTS]
        tasks += [Task(BLIND_GREY_ID, "", CONDITION_BLIND_GREY, pid) for pid in BLIND_PROMPTS]
        return tasks
    if mode == "items":
        chosen = list(items)
    elif mode == "smoke":
        chosen = [it for it in items if it.smoke == 1]
    else:
        raise ValueError(f"unknown mode {mode!r}; expected 'items', 'smoke' or 'blind'")
    if limit is not None:
        chosen = chosen[:limit]
    tasks: list[Task] = []
    for it in chosen:
        prompt_ids = (SMOKE_PROMPT,) if mode == "smoke" else PROMPT_SETS[it.prompt_set]
        tasks.extend(Task(it.item_id, it.nih_filename, CONDITION_IMAGE, pid) for pid in prompt_ids)
    return tasks


def missing_images(tasks: Iterable[Task], images_dir: str | Path) -> list[str]:
    """NIH filenames needed by image tasks that do not exist under ``images_dir`` (sorted, unique)."""
    root = Path(images_dir)
    names = sorted({t.nih_filename for t in tasks if t.condition == CONDITION_IMAGE})
    return [n for n in names if not (root / n).is_file()]


# ----------------------------------------------------------------------------- images
def image_from_bytes(data: bytes) -> Image.Image:
    """Decode PNG (or other image) bytes and return ``convert("L").convert("RGB")`` of it.

    NIH PNGs are 8-bit greyscale, with a few stored as RGBA; converting to "L" first gives every
    image the same grey values before the three-channel copy the processors expect.
    """
    with Image.open(io.BytesIO(data)) as im:
        im.load()
        return im.convert("L").convert("RGB")


def load_image_file(path: str | Path) -> tuple[Image.Image, str]:
    """Load an image file as RGB (via "L") and return it with the SHA-256 of the file bytes."""
    data = Path(path).read_bytes()
    return image_from_bytes(data), sha256_bytes(data)


def grey_png_bytes(side: int = constants.IMAGE_SIDE, level: int = constants.GREY_LEVEL) -> bytes:
    """PNG bytes of the BLIND-grey image: ``side`` x ``side``, 8-bit greyscale, every pixel ``level``.

    The bytes are deterministic for a given Pillow and zlib; their SHA-256 is written to every
    BLIND-grey record, and ``write_grey_png`` saves a copy beside the run for audit.
    """
    if not (0 <= int(level) <= 255) or int(side) < 1:
        raise ValueError(f"invalid grey image: side={side}, level={level}")
    buf = io.BytesIO()
    Image.new("L", (int(side), int(side)), int(level)).save(buf, format="PNG")
    return buf.getvalue()


def write_grey_png(directory: str | Path, data: bytes | None = None) -> Path:
    """Write the BLIND-grey PNG bytes to ``<directory>/BLIND-GREY_<side>.png`` and return the path."""
    payload = grey_png_bytes() if data is None else data
    path = Path(directory) / f"{BLIND_GREY_ID}_{constants.IMAGE_SIDE}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.read_bytes() != payload:
        path.write_bytes(payload)
    return path


# ----------------------------------------------------------------------------- token ids and signals
@dataclass(frozen=True)
class SignalIds:
    """Single-token ids of the yes, no and (exploratory) abstain surface forms for one tokenizer."""

    yes_ids: tuple[int, ...]
    no_ids: tuple[int, ...]
    abstain_ids: tuple[int, ...]
    skipped: dict[str, list[str]] = field(default_factory=dict)


def signal_ids_from_encoder(encode: Callable[[str], Sequence[int]]) -> SignalIds:
    """Token ids for ``constants.YES_FORMS``, ``NO_FORMS`` and ``ABSTAIN_FIRST_FORMS``.

    ``encode`` maps text to ids without special tokens (``tok.encode(s, add_special_tokens=False)``).
    Only forms that encode to exactly one token count (``cxrdev.logits.single_token_ids``); the
    skipped forms are kept for the run log. Raises ValueError when no yes form or no no form is a
    single token, or when the two lists share an id. Abstain ids that coincide with a yes or no id
    are dropped (the abstain signal is exploratory and must not double-count).
    """
    def enc(s: str) -> list[int]:
        return [int(t) for t in encode(s)]

    yes, skipped_yes = single_token_ids(enc, constants.YES_FORMS)
    no, skipped_no = single_token_ids(enc, constants.NO_FORMS)
    abstain, skipped_abstain = single_token_ids(enc, constants.ABSTAIN_FIRST_FORMS)
    if not yes or not no:
        raise ValueError(f"no single-token yes or no form for this tokenizer (yes={yes}, no={no})")
    shared = sorted(set(yes) & set(no))
    if shared:
        raise ValueError(f"yes and no forms share token ids {shared}")
    abstain = [i for i in abstain if i not in yes and i not in no]
    return SignalIds(
        tuple(yes), tuple(no), tuple(abstain),
        {"yes": list(skipped_yes), "no": list(skipped_no), "abstain": list(skipped_abstain)},
    )


def first_token_fields(first_logits: np.ndarray, ids: SignalIds) -> dict[str, float]:
    """First-token signal from the raw logits of the first generated position.

    Applies ``cxrdev.logits.log_softmax`` (float64) to the raw logits (shape (V,) or (1, V)) and
    returns ``logp_yes``, ``logp_no``, ``logp_abstain`` (NaN without abstain ids), ``z`` and
    ``coverage`` as Python floats (``cxrdev.logits.first_token_signal``).
    """
    lp = log_softmax(np.asarray(first_logits, dtype=np.float64))
    sig = first_token_signal(lp, ids.yes_ids, ids.no_ids, ids.abstain_ids)
    return {k: float(sig[k]) for k in ("logp_yes", "logp_no", "logp_abstain", "z", "coverage")}


def sequence_logprob(step_logits: np.ndarray, target_ids: Sequence[int]) -> float:
    """Teacher-forced log-likelihood of ``target_ids`` from raw logits.

    ``step_logits`` has shape (k, V): row j holds the raw logits of the position that predicts
    ``target_ids[j]`` (that is, the position just before it). Returns
    ``sum_j log_softmax(step_logits[j])[target_ids[j]]`` in float64. Raises ValueError on a shape
    mismatch, an id outside the vocabulary, or NaN logits.
    """
    arr = np.asarray(step_logits, dtype=np.float64)
    targets = [int(t) for t in target_ids]
    if arr.ndim != 2 or arr.shape[0] != len(targets) or not targets:
        raise ValueError(f"step_logits must have shape (k, V) with k = {len(targets)} > 0, got {arr.shape}")
    if np.isnan(arr).any():
        raise ValueError("step_logits contains NaN")
    vocab = arr.shape[1]
    if any(t < 0 or t >= vocab for t in targets):
        raise ValueError(f"target ids {targets} out of range for a vocabulary of {vocab}")
    lp = log_softmax(arr)
    return float(sum(lp[j, t] for j, t in enumerate(targets)))


def full_sequence_fields(logp_yes: float, logp_no: float) -> dict[str, float]:
    """Signal fields from the full-sequence log-likelihoods of the "Yes" and "No" completions.

    ``z = logp_yes - logp_no``; ``coverage = exp(logp_yes) + exp(logp_no)`` (the probability of the
    two complete answers). Raises ValueError if either input is not finite.
    """
    a, b = float(logp_yes), float(logp_no)
    if not (math.isfinite(a) and math.isfinite(b)):
        raise ValueError(f"full-sequence log-likelihoods must be finite, got {a}, {b}")
    return {"logp_yes": a, "logp_no": b, "z": a - b, "coverage": math.exp(a) + math.exp(b)}


def generation_outcome(new_ids: Sequence[int], eos_ids: Iterable[int]) -> tuple[int, bool]:
    """Number of generated tokens and whether generation ended with an end-of-sequence token.

    Counts tokens up to and including the first token in ``eos_ids`` (anything after it, such as
    padding, is ignored); without one, every token counts and ``ended_with_eos`` is False.
    """
    eos = {int(e) for e in eos_ids}
    ids = [int(t) for t in new_ids]
    for j, tok in enumerate(ids):
        if tok in eos:
            return j + 1, True
    return len(ids), False


# ----------------------------------------------------------------------------- S7 records
def make_record(values: Mapping[str, Any]) -> dict[str, Any]:
    """Build an S7 record with the fields in schema order, coercing numeric and list types.

    Raises ValueError when a field is missing or an unknown field is present.
    """
    missing = [f for f in S7_FIELDS if f not in values]
    extra = sorted(set(values) - set(S7_FIELDS))
    if missing or extra:
        raise ValueError(f"S7 record: missing {missing}, unexpected {extra}")
    rec: dict[str, Any] = {}
    for f in S7_FIELDS:
        v = values[f]
        if f in _INT_FIELDS:
            v = int(v)
        elif f in _FLOAT_FIELDS:
            v = float(v)
        elif f in _ID_LIST_FIELDS:
            v = [int(x) for x in v]
        elif f == "ended_with_eos":
            if not isinstance(v, (bool, np.bool_)):
                raise ValueError(f"ended_with_eos must be a bool, got {type(v).__name__}")
            v = bool(v)
        rec[f] = v
    validate_record(rec)
    return rec


def validate_record(rec: Mapping[str, Any]) -> None:
    """Check one S7 record: exact fields, types, allowed values and condition-specific rules.

    Raises ValueError listing every problem found.
    """
    problems: list[str] = []
    keys = list(rec)
    if keys != list(S7_FIELDS):
        problems.append(f"fields differ from S7 (missing {sorted(set(S7_FIELDS) - set(keys))}, "
                        f"unexpected {sorted(set(keys) - set(S7_FIELDS))}, or out of order)")
        raise ValueError("; ".join(problems))
    for f in _STR_FIELDS:
        if not isinstance(rec[f], str):
            problems.append(f"{f} must be str")
    for f in _INT_FIELDS:
        if isinstance(rec[f], bool) or not isinstance(rec[f], int):
            problems.append(f"{f} must be int")
    for f in _FLOAT_FIELDS:
        if isinstance(rec[f], bool) or not isinstance(rec[f], (int, float)):
            problems.append(f"{f} must be float")
    for f in _ID_LIST_FIELDS:
        v = rec[f]
        if not isinstance(v, list) or not v or not all(isinstance(x, int) and not isinstance(x, bool) for x in v):
            problems.append(f"{f} must be a non-empty list of int")
    if not isinstance(rec["ended_with_eos"], bool):
        problems.append("ended_with_eos must be bool")
    if problems:
        raise ValueError("S7 record: " + "; ".join(problems))
    cond = rec["condition"]
    if cond not in CONDITIONS:
        problems.append(f"condition {cond!r} not in {CONDITIONS}")
    if rec["parsed"] not in ANSWERS:
        problems.append(f"parsed {rec['parsed']!r} not in {ANSWERS}")
    if rec["signal"] not in SIGNALS:
        problems.append(f"signal {rec['signal']!r} not in {SIGNALS}")
    if rec["prompt_sha256"] != prompt_sha256(rec["prompt_text"]):
        problems.append("prompt_sha256 does not match prompt_text")
    for f in ("logp_yes", "logp_no", "z", "wall_s"):
        if not math.isfinite(rec[f]):
            problems.append(f"{f} must be finite")
    if not (0.0 <= rec["coverage"] <= 1.0 + 1e-6):
        problems.append(f"coverage {rec['coverage']} outside [0, 1]")
    if rec["n_generated_tokens"] < 0 or rec["max_new_tokens"] < 1:
        problems.append("token counts out of range")
    if cond == CONDITION_BLIND_TEXT:
        if rec["item_id"] != BLIND_TEXT_ID or rec["nih_filename"] or rec["image_sha256"] or rec["visual_tokens"] != 0:
            problems.append("BLIND-text rows need item_id BLIND-TEXT, empty nih_filename and image_sha256, visual_tokens 0")
    elif cond == CONDITION_BLIND_GREY:
        if rec["item_id"] != BLIND_GREY_ID or rec["nih_filename"] or not SHA256_RE.match(rec["image_sha256"]):
            problems.append("BLIND-grey rows need item_id BLIND-GREY, empty nih_filename and the grey PNG sha256")
        if rec["visual_tokens"] <= 0:
            problems.append("BLIND-grey rows need a positive visual_tokens")
    elif cond == CONDITION_IMAGE:
        if not ITEM_ID_RE.match(rec["item_id"]) or not NIH_FILENAME_RE.match(rec["nih_filename"]):
            problems.append("image rows need an opaque item_id and an NIH filename")
        if not SHA256_RE.match(rec["image_sha256"]) or rec["visual_tokens"] <= 0:
            problems.append("image rows need the PNG sha256 and a positive visual_tokens")
    if problems:
        raise ValueError("S7 record: " + "; ".join(problems))


def record_is_truncated(rec: Mapping[str, Any]) -> bool:
    """``cxrdev.parse.is_truncated`` applied to an S7 record."""
    return is_truncated(int(rec["n_generated_tokens"]), int(rec["max_new_tokens"]), bool(rec["ended_with_eos"]))


# ----------------------------------------------------------------------------- resume
def existing_records(path: str | Path) -> list[dict]:
    """Records already in a JSONL file (empty when the file does not exist)."""
    p = Path(path)
    return read_jsonl(p) if p.exists() else []


def check_resume(records: Sequence[Mapping[str, Any]], expected: Mapping[str, Any]) -> set[tuple]:
    """Validate existing records against the current run and return their resume keys.

    Every record must agree with ``expected`` on each of its fields (for example ``run_id``,
    ``model_id``, ``revision``, ``signal``), and its ``prompt_sha256`` must match the current
    text of its ``prompt_id``. Duplicate keys (item_id, prompt_id, condition) are a conflict too.
    Raises ResumeConflictError naming the first conflicts, so a run never mixes configurations
    in one file; start a new ``--out`` file instead.
    """
    conflicts: list[str] = []
    keys: set[tuple] = set()
    current_hash: dict[str, str | None] = {}
    for i, rec in enumerate(records, start=1):
        missing = [f for f in RESUME_KEY_FIELDS if f not in rec]
        if missing:
            conflicts.append(f"record {i} lacks {missing}")
            continue
        key = tuple(rec[f] for f in RESUME_KEY_FIELDS)
        if key in keys:
            conflicts.append(f"record {i} duplicates key {key}")
        keys.add(key)
        for fname, want in expected.items():
            if rec.get(fname) != want:
                conflicts.append(f"record {i} {key}: {fname}={rec.get(fname)!r}, current run has {want!r}")
        pid = rec.get("prompt_id")
        if pid not in current_hash:
            try:
                current_hash[pid] = prompt_sha256(prompt_text(pid, DEVICE_KEY))
            except (KeyError, TypeError):
                current_hash[pid] = None
        if current_hash[pid] is None or rec.get("prompt_sha256") != current_hash[pid]:
            conflicts.append(f"record {i} {key}: prompt_sha256 differs from the current {pid} text")
        if len(conflicts) >= 20:
            break
    if conflicts:
        shown = "\n  ".join(conflicts[:8])
        raise ResumeConflictError(
            f"{len(conflicts)} conflict(s) with the existing records (write to a new --out file):\n  {shown}"
        )
    return keys


# ----------------------------------------------------------------------------- smoke test
def coverage_decision(coverages: Sequence[float], threshold: float = constants.COVERAGE_MIN_MEDIAN) -> tuple[float, str]:
    """Median first-token coverage and the registered decision.

    Returns ``(median, "first_token")`` when the median is at least ``threshold`` (0.90) and
    ``(median, "full_sequence")`` otherwise (PROTOCOL.md section 3). Raises ValueError for an empty
    list or a non-finite value.
    """
    vals = [float(c) for c in coverages]
    if not vals:
        raise ValueError("no coverage values")
    if not all(math.isfinite(v) for v in vals):
        raise ValueError("coverage values must be finite")
    med = float(statistics.median(vals))
    return med, ("first_token" if med >= threshold else "full_sequence")


def smoke_section(records: Sequence[Mapping[str, Any]], threshold: float = constants.COVERAGE_MIN_MEDIAN) -> dict[str, Any]:
    """Summary of one model's smoke records (first-token signal under P0 on the smoke items).

    Keys: ``n_items``, ``visual_tokens`` (the single count, or the sorted list if several),
    ``median_coverage``, ``min_coverage``, ``max_coverage``, ``coverage_threshold``, ``decision``,
    ``s_per_item_mean``, ``s_per_item_median``, ``parsed_counts``, ``n_truncated``. Dev-pool items
    only, so these aggregates may be read before the blind verification.
    """
    if not records:
        raise ValueError("no smoke records")
    covs = [float(r["coverage"]) for r in records]
    median, decision = coverage_decision(covs, threshold)
    vts = sorted({int(r["visual_tokens"]) for r in records})
    walls = [float(r["wall_s"]) for r in records]
    return {
        "n_items": len(records),
        "prompt_id": SMOKE_PROMPT,
        "visual_tokens": vts[0] if len(vts) == 1 else vts,
        "median_coverage": median,
        "min_coverage": min(covs),
        "max_coverage": max(covs),
        "coverage_threshold": float(threshold),
        "decision": decision,
        "s_per_item_mean": float(statistics.fmean(walls)),
        "s_per_item_median": float(statistics.median(walls)),
        "parsed_counts": {a: sum(1 for r in records if r["parsed"] == a) for a in ANSWERS},
        "n_truncated": sum(1 for r in records if record_is_truncated(r)),
    }


def atomic_write_text(path: str | Path, text: str) -> None:
    """Write UTF-8 text to a temporary file beside ``path``, fsync, then replace ``path``."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, p)


def read_json(path: str | Path) -> Any:
    """Load a JSON file (UTF-8)."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: str | Path, obj: Any) -> None:
    """Write JSON atomically (indent 2, UTF-8, trailing newline)."""
    atomic_write_text(path, json.dumps(obj, indent=2, ensure_ascii=False, default=_json_default) + "\n")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (set, frozenset, tuple)):
        return list(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def update_smoke_report(path: str | Path, model_key: str, section: Mapping[str, Any]) -> dict[str, Any]:
    """Insert or replace one model's section in ``smoke_report.json`` and return the whole report.

    The report is ``{"coverage_threshold": 0.9, "updated_utc": ..., "models": {model_key: section}}``;
    sections of other models are kept.
    """
    p = Path(path)
    report: dict[str, Any] = {}
    if p.exists():
        loaded = read_json(p)
        if isinstance(loaded, dict):
            report = loaded
    models = report.get("models") if isinstance(report.get("models"), dict) else {}
    models[model_key] = dict(section)
    report = {
        "coverage_threshold": float(constants.COVERAGE_MIN_MEDIAN),
        "updated_utc": utc_now(),
        "models": models,
    }
    write_json(p, report)
    return report


def read_smoke_decision(path: str | Path, model_key: str) -> str:
    """The signal decided by the smoke test for ``model_key`` ("first_token" or "full_sequence").

    Raises SmokeReportError when the report or the model's section is missing, when the
    visual-token assertion did not pass, or when the decision is not a known signal.
    """
    p = Path(path)
    if not p.exists():
        raise SmokeReportError(
            f"{p} not found: run pod/smoke_test.py first, or pass --signal first_token|full_sequence"
        )
    report = read_json(p)
    section = (report.get("models") or {}).get(model_key) if isinstance(report, dict) else None
    if not isinstance(section, dict):
        raise SmokeReportError(f"{p} has no section for {model_key}: run pod/smoke_test.py for this model")
    if section.get("visual_token_assertion") != "pass":
        raise SmokeReportError(f"{p}: the smoke test of {model_key} did not pass ({section.get('error', 'no detail')})")
    decision = section.get("decision")
    if decision not in SIGNALS:
        raise SmokeReportError(f"{p}: {model_key} has no valid decision ({decision!r})")
    return decision


def resolve_signal(requested: str, smoke_report: str | Path | None, model_key: str) -> tuple[str, str]:
    """The signal to use and where it came from.

    "first_token" and "full_sequence" are used as given (source "cli"); "auto" takes the smoke
    decision for the model from ``smoke_report`` (source "smoke_report").
    """
    if requested in SIGNALS:
        return requested, "cli"
    if requested != "auto":
        raise ValueError(f"unknown signal {requested!r}; expected one of {SIGNAL_CHOICES}")
    if smoke_report is None:
        raise SmokeReportError("--signal auto needs a smoke report")
    return read_smoke_decision(smoke_report, model_key), "smoke_report"


# ----------------------------------------------------------------------------- CSV output (no pandas on the pod)
def write_csv_rows(path: str | Path, columns: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> str:
    """Write rows as UTF-8 CSV with ``\\n`` line endings, atomically; return the SHA-256 of the bytes.

    Floats are written with ``repr`` (shortest round-trip form); every row must have exactly
    ``columns``.
    """
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(list(columns))
    for row in rows:
        if set(row) != set(columns):
            raise ValueError(f"row keys {sorted(row)} differ from columns {list(columns)}")
        writer.writerow([repr(float(row[c])) if isinstance(row[c], (float, np.floating)) else row[c] for c in columns])
    data = buf.getvalue()
    atomic_write_text(path, data)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------------- environment (lazy torch / hub)
def resolve_model_revision(repo_id: str, revision: str | None = None, *, local_files_only: bool = False) -> str:
    """The commit hash that ``revision`` (default: the default branch) of a Hub model repo points to.

    A 40-character hash is returned unchanged. Otherwise ``huggingface_hub.resolve_revision`` is
    used (offline it reads the ``refs/`` mapping that a previous download cached); when that is
    unavailable, the local snapshot folder name is used. Loading every file of a model with the
    returned hash keeps the weights, config and processor on one commit, which is the
    ``revision`` written to each record.
    """
    if revision and COMMIT_RE.match(revision):
        return revision
    import huggingface_hub

    offline = local_files_only or os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in {"1", "true", "yes", "on"}
    resolver = getattr(huggingface_hub, "resolve_revision", None)
    if resolver is not None:
        try:
            resolved = resolver(repo_id, revision=revision, local_files_only=offline)
            commit = getattr(resolved, "resolved", None) or str(resolved)
            if COMMIT_RE.match(commit):
                return commit
        except Exception:  # noqa: BLE001 - fall back to the local snapshot below
            pass
    path = huggingface_hub.snapshot_download(repo_id, revision=revision, local_files_only=True)
    commit = Path(path).name
    if not COMMIT_RE.match(commit):
        raise RuntimeError(f"could not resolve a commit hash for {repo_id} (got {commit!r})")
    return commit


def library_versions() -> dict[str, str]:
    """Versions of torch and transformers (imported here, lazily)."""
    import torch
    import transformers

    return {"torch_version": str(torch.__version__), "transformers_version": str(transformers.__version__)}


def torch_dtype(name: str) -> Any:
    """``torch.<name>`` for "bfloat16", "float16" or "float32"."""
    import torch

    if name not in {"bfloat16", "float16", "float32"}:
        raise ValueError(f"unsupported dtype {name!r}")
    return getattr(torch, name)


def default_device() -> str:
    """"cuda:0" when a CUDA GPU is visible, else "cpu"."""
    import torch

    return "cuda:0" if torch.cuda.is_available() else "cpu"


def normalise_device(device: str | None) -> str:
    """Normalise a device string ("cuda" -> "cuda:0"); None -> ``default_device()``."""
    if device is None:
        return default_device()
    d = str(device).strip()
    return "cuda:0" if d == "cuda" else d


def reset_cuda_peak() -> None:
    """Reset the CUDA peak-memory counter (no-op without CUDA)."""
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def cuda_peak_gb(device: str) -> float | None:
    """Peak CUDA memory allocated on ``device`` since the last reset, in GiB (None on CPU)."""
    import torch

    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        return None
    return round(torch.cuda.max_memory_allocated(torch.device(device)) / 2**30, 3)


def gpu_name(device: str) -> str | None:
    """The CUDA device name (None on CPU)."""
    import torch

    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        return None
    return str(torch.cuda.get_device_name(torch.device(device)))


def free_accelerator_memory() -> None:
    """Collect garbage and release cached CUDA memory (between two models)."""
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:  # pragma: no cover - torch is always present where models run
        pass


def meta_path(out_path: str | Path) -> Path:
    """Sidecar file ``<stem>.meta.json`` beside an output file."""
    p = Path(out_path)
    return p.with_name(p.stem + ".meta.json")


def append_session_meta(out_path: str | Path, session: Mapping[str, Any]) -> Path:
    """Append one session description to the ``<stem>.meta.json`` sidecar of an output file.

    The sidecar holds run settings only (model, revision, versions, token ids, generation
    configuration, timing), never model outputs.
    """
    path = meta_path(out_path)
    meta: dict[str, Any] = {"sessions": []}
    if path.exists():
        loaded = read_json(path)
        if isinstance(loaded, dict) and isinstance(loaded.get("sessions"), list):
            meta = loaded
    meta["sessions"].append(dict(session))
    write_json(path, meta)
    return path
