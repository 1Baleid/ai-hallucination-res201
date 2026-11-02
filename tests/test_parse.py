"""Tests for cxrdev.parse (PROTOCOL.md section 3, "Parsing")."""
from __future__ import annotations

import pytest

from cxrdev import parse


def test_parser_version():
    assert parse.PARSER_VERSION == "2"
    assert parse.ANSWERS == ("yes", "no", "abstain", "unparsable")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # stage 1: leading yes / no
        ("Yes", "yes"),
        ("No.", "no"),
        ("**No**", "no"),
        ("YES.", "yes"),
        (" yes", "yes"),
        ("Yes, there is an endotracheal tube.", "yes"),
        ("\nNo", "no"),
        ("\n\n  No", "no"),
        ("No, I cannot tell.", "no"),
        ("_Yes_", "yes"),
        ("`No`", "no"),
        ("- Yes", "yes"),
        ('"Yes"', "yes"),
        ("“No”", "no"),
        ("yes/no", "yes"),
        ("No-", "no"),
        # stage 2: abstain lexicon anywhere
        ("I cannot tell.", "abstain"),
        ("I can't tell", "abstain"),
        ("I don’t know", "abstain"),
        ("I don‘t know", "abstain"),
        ("I don't know.", "abstain"),
        ("I do not know", "abstain"),
        ("I'm not sure.", "abstain"),
        ("I am unsure", "abstain"),
        ("Unable to determine", "abstain"),
        ("I cannot determine", "abstain"),
        ("I can’t determine this", "abstain"),
        ("Cannot tell.", "abstain"),
        ("Unsure", "abstain"),
        ("Not sure", "abstain"),
        ("Unknown.", "abstain"),
        ("It is not possible to tell from this image.", "abstain"),
        ("It is not possible to determine.", "abstain"),
        ("**I don't know**", "abstain"),
        ("I\ndon't\nknow", "abstain"),
        ("The answer is: I DON'T KNOW", "abstain"),
        # stage 3: unparsable
        ("The answer is no.", "unparsable"),
        ("There is no endotracheal tube", "unparsable"),
        ("Nope", "unparsable"),
        ("None", "unparsable"),
        ("Yesterday", "unparsable"),
        ("Noted.", "unparsable"),
        ("", "unparsable"),
        ("   \n\t ", "unparsable"),
        ("Answer: Yes", "unparsable"),
        ("I think so", "unparsable"),
    ],
)
def test_parse_answer(raw, expected):
    assert parse.parse_answer(raw) == expected


def test_stage1_wins_over_abstain_lexicon():
    assert parse.parse_answer("Yes, but I am not sure.") == "yes"
    assert parse.parse_answer("No. Unknown device.") == "no"


def test_parse_missing_values():
    assert parse.parse_answer(None) == "unparsable"
    assert parse.parse_answer(float("nan")) == "unparsable"
    with pytest.raises(TypeError):
        parse.parse_answer(1)
    with pytest.raises(TypeError):
        parse.parse_answer(b"Yes")


def test_normalise():
    assert parse.normalise("I don’t know") == "I don't know"
    assert parse.normalise("‘a’ “b”") == "'a' \"b\""
    assert parse.normalise("**Yes**") == "Yes"
    assert parse.normalise("__No__") == "No"
    assert parse.normalise("```yes```") == "yes"
    assert parse.normalise("  a \n\n b\t\tc  ") == "a b c"
    assert parse.normalise("a b") == "a b"  # non-breaking space is whitespace
    assert parse.normalise("") == ""
    assert parse.normalise(None) == ""
    # straight quotes and other punctuation survive
    assert parse.normalise("It's \"fine\".") == "It's \"fine\"."


def test_abstain_regex_is_the_registered_lexicon():
    assert parse.ABSTAIN_RE == (
        r"(cannot tell|can't tell|don't know|do not know|not sure|unsure|unknown|cannot determine|"
        r"can't determine|unable to determine|not possible to (tell|determine))"
    )
    assert parse.STAGE1_RE == r"^\W*(yes|no)\b"


@pytest.mark.parametrize(
    ("n_gen", "max_new", "eos", "expected"),
    [
        (16, 16, False, True),   # hit the budget without EOS
        (17, 16, False, True),   # beyond the budget (defensive)
        (16, 16, True, False),   # EOS as the last allowed token
        (3, 16, True, False),    # short answer with EOS
        (3, 16, False, False),   # stopped early without EOS: not truncation by the budget
        (0, 16, False, False),
        (15, 16, False, False),
    ],
)
def test_is_truncated_truth_table(n_gen, max_new, eos, expected):
    assert parse.is_truncated(n_gen, max_new, eos) is expected


def test_is_truncated_types():
    import numpy as np

    assert parse.is_truncated(np.int64(16), 16, np.bool_(False)) is True
    assert parse.is_truncated(16, 16, 1) is False
    with pytest.raises(TypeError):
        parse.is_truncated(16, 16, "False")
    with pytest.raises(TypeError):
        parse.is_truncated(16, 16, None)
