"""Two-stage answer parser for greedy model outputs (PROTOCOL.md section 3, "Parsing").

Pipeline: ``normalise`` the raw output (curly quotes to straight quotes, markdown emphasis and
backticks removed, whitespace collapsed); stage 1 reads a leading "yes" or "no" (anything
non-alphanumeric may precede it, and it must end at a word boundary, so "Nope" and "None"
do not count); stage 2 searches anywhere in the output for the abstain lexicon; otherwise the
output is "unparsable". Unparsable and truncated outputs are excluded from the metric
denominators and reported as their own rates (PROTOCOL.md section 3, pre-registered rule).

The version string is written to every run record; change ``PARSER_VERSION`` with any change
to the rules below. This module imports only the standard library: it runs on the GPU pod.
"""
from __future__ import annotations

import math
import re

PARSER_VERSION = "2"
ANSWERS: tuple[str, ...] = ("yes", "no", "abstain", "unparsable")

STAGE1_RE = r"^\W*(yes|no)\b"
ABSTAIN_RE = (
    r"(cannot tell|can't tell|don't know|do not know|not sure|unsure|unknown|"
    r"cannot determine|can't determine|unable to determine|not possible to (tell|determine))"
)

_STAGE1 = re.compile(STAGE1_RE, re.IGNORECASE)
_ABSTAIN = re.compile(ABSTAIN_RE, re.IGNORECASE)
_WHITESPACE = re.compile(r"\s+")
_TRANSLATE = str.maketrans(
    {
        "‘": "'",  # left single quotation mark
        "’": "'",  # right single quotation mark (the usual curly apostrophe)
        "“": '"',  # left double quotation mark
        "”": '"',  # right double quotation mark
        "*": None,
        "_": None,
        "`": None,
    }
)


def normalise(text: str) -> str:
    """Normalise a raw output before parsing.

    Maps U+2018/U+2019 to ``'`` and U+201C/U+201D to ``"``, removes ``*``, ``_`` and backticks,
    collapses every whitespace run (newlines included) to one space, and strips both ends.
    ``None`` and a float NaN (an empty cell read back from a table) normalise to ``""``;
    any other non-string raises TypeError.
    """
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return ""
    if not isinstance(text, str):
        raise TypeError(f"text must be str, not {type(text).__name__}")
    return _WHITESPACE.sub(" ", text.translate(_TRANSLATE)).strip()


def parse_answer(text: str) -> str:
    """Parse one output into "yes", "no", "abstain" or "unparsable".

    Stage 1: ``re.match(r"^\\W*(yes|no)\\b", s, re.I)`` on the normalised text gives "yes" or
    "no" (so "No, I cannot tell." is "no"). Stage 2: otherwise ``re.search(ABSTAIN_RE, s, re.I)``
    gives "abstain". Stage 3: otherwise "unparsable" (for example "The answer is no." or "Nope").
    """
    s = normalise(text)
    match = _STAGE1.match(s)
    if match:
        return match.group(1).lower()
    if _ABSTAIN.search(s):
        return "abstain"
    return "unparsable"


def is_truncated(n_generated_tokens: int, max_new_tokens: int, ended_with_eos: bool) -> bool:
    """True when generation hit the token budget without an end-of-sequence token.

    Equals ``(not ended_with_eos) and n_generated_tokens >= max_new_tokens``. A string for
    ``ended_with_eos`` (for example "False" read back from a CSV) raises TypeError rather than
    being silently treated as true.
    """
    if ended_with_eos is None or isinstance(ended_with_eos, (str, bytes)):
        raise TypeError(f"ended_with_eos must be a bool, not {type(ended_with_eos).__name__}")
    return (not bool(ended_with_eos)) and int(n_generated_tokens) >= int(max_new_tokens)
