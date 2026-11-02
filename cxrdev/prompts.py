"""Prompt loading, device substitution and hashing (PROTOCOL.md section 4).

The prompt texts live in ``cxrdev/prompts.json`` (fixed input, written by the lead). A prompt
is its template with the ``{device}`` placeholder replaced by a device phrase, for example
P0 with the default device "ett":

    Is an endotracheal tube present in this chest X-ray? Answer with a single word: Yes or No.

P2hi and P2lo are Kalai et al.'s confidence-target sentence with t substituted as a
percentage, followed by the P1 question and answer line.

This module imports only the standard library: it is copied to the GPU pod.
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

PROMPT_IDS: tuple[str, ...] = ("P0", "P1", "P2hi", "P2lo")
DEFAULT_PROMPTS_PATH: Path = Path(__file__).resolve().with_name("prompts.json")
PLACEHOLDER = "{device}"


def _validate(data: Any, source: str) -> None:
    """Raise ValueError when a loaded prompts file does not have the expected structure."""
    if not isinstance(data, dict):
        raise ValueError(f"{source}: top level must be a JSON object")
    devices = data.get("devices")
    prompts = data.get("prompts")
    if not isinstance(devices, dict) or not devices:
        raise ValueError(f"{source}: 'devices' must be a non-empty object")
    if not isinstance(prompts, dict) or not prompts:
        raise ValueError(f"{source}: 'prompts' must be a non-empty object")
    for key, phrase in devices.items():
        if not isinstance(phrase, str) or not phrase.strip():
            raise ValueError(f"{source}: device {key!r} must map to a non-empty string")
    for pid, entry in prompts.items():
        if not isinstance(entry, dict):
            raise ValueError(f"{source}: prompt {pid!r} must be an object")
        template = entry.get("template")
        if not isinstance(template, str) or template.count(PLACEHOLDER) != 1:
            raise ValueError(f"{source}: prompt {pid!r} needs a 'template' string with exactly one {PLACEHOLDER}")
        if "t" not in entry:
            raise ValueError(f"{source}: prompt {pid!r} has no 't' field (use null for no penalty)")
        t = entry["t"]
        if t is not None and (isinstance(t, bool) or not isinstance(t, (int, float)) or not 0.0 < float(t) < 1.0):
            raise ValueError(f"{source}: prompt {pid!r} has t={t!r}; expected null or a number in (0, 1)")


def load_prompts(path: str | Path | None = None) -> dict:
    """Load and validate a prompts file (default ``cxrdev/prompts.json``).

    Returns a fresh dict on every call (callers may modify it freely). Raises ValueError
    when the file lacks ``devices`` / ``prompts`` or a template lacks exactly one
    ``{device}`` placeholder.
    """
    p = DEFAULT_PROMPTS_PATH if path is None else Path(path)
    with open(p, encoding="utf-8") as fh:
        data = json.load(fh)
    _validate(data, str(p))
    return data


@lru_cache(maxsize=1)
def _default_prompts() -> dict:
    """The default prompts file, parsed once per process (read-only use inside this module)."""
    return load_prompts()


def _prompt_entry(prompt_id: str) -> dict:
    prompts = _default_prompts()["prompts"]
    try:
        return prompts[prompt_id]
    except KeyError:
        raise KeyError(f"unknown prompt id {prompt_id!r}; known ids: {sorted(prompts)}") from None


def prompt_text(prompt_id: str, device_key: str = "ett") -> str:
    """The prompt text: the template of ``prompt_id`` with ``{device}`` replaced by ``devices[device_key]``.

    Raises KeyError for an unknown prompt id or device key.
    """
    entry = _prompt_entry(prompt_id)
    devices = _default_prompts()["devices"]
    try:
        phrase = devices[device_key]
    except KeyError:
        raise KeyError(f"unknown device key {device_key!r}; known keys: {sorted(devices)}") from None
    return entry["template"].replace(PLACEHOLDER, phrase)


def prompt_sha256(text: str) -> str:
    """Hex SHA-256 of the UTF-8 bytes of ``text`` (logged as ``prompt_sha256`` in every run record)."""
    if not isinstance(text, str):
        raise TypeError(f"text must be str, not {type(text).__name__}")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def penalty_t(prompt_id: str) -> float | None:
    """The confidence target t of a prompt (0.9 for P2hi, 0.5 for P2lo, None for P0 and P1).

    Raises KeyError for an unknown prompt id.
    """
    t = _prompt_entry(prompt_id)["t"]
    return None if t is None else float(t)
