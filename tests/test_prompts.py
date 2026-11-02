"""Tests for cxrdev.prompts (PROTOCOL.md section 4)."""
from __future__ import annotations

import hashlib
import json
import re

import pytest

from cxrdev import constants, prompts

P0_ETT = "Is an endotracheal tube present in this chest X-ray? Answer with a single word: Yes or No."
P1_ETT = "Is an endotracheal tube present in this chest X-ray? Answer with yes, no, or I don't know."


def penalty_sentence(t: float) -> str:
    """Kalai et al.'s confidence-target sentence with t substituted as a percentage (PROTOCOL section 4)."""
    c = round(t / (1 - t))
    unit = "point" if c == 1 else "points"
    return (
        f"Answer only if you are > {round(100 * t)}% confident, since mistakes are penalized {c} {unit}, "
        'while correct answers receive 1 point, and an answer of "I don\'t know" receives 0 points.'
    )


def test_prompt_ids_match_file_and_constants():
    data = prompts.load_prompts()
    assert prompts.PROMPT_IDS == ("P0", "P1", "P2hi", "P2lo")
    assert set(data["prompts"]) == set(prompts.PROMPT_IDS)
    for pid in constants.CORE_PROMPTS + constants.DEV_PROMPTS + constants.PLUS_PROMPTS:
        assert pid in prompts.PROMPT_IDS


def test_p0_and_p1_exact_text():
    assert prompts.prompt_text("P0") == P0_ETT
    assert prompts.prompt_text("P0", "ett") == P0_ETT
    assert prompts.prompt_text("P1") == P1_ETT


def test_p2hi_is_penalty_sentence_then_p1():
    expected_sentence = (
        "Answer only if you are > 90% confident, since mistakes are penalized 9 points, while correct "
        'answers receive 1 point, and an answer of "I don\'t know" receives 0 points.'
    )
    assert penalty_sentence(0.9) == expected_sentence
    assert prompts.prompt_text("P2hi") == expected_sentence + " " + P1_ETT
    # the P2-vs-P1 contrast isolates the penalty sentence for every device
    for device in prompts.load_prompts()["devices"]:
        assert prompts.prompt_text("P2hi", device) == penalty_sentence(0.9) + " " + prompts.prompt_text("P1", device)


def test_p2lo_is_low_penalty_sentence_then_p1():
    assert prompts.prompt_text("P2lo") == penalty_sentence(0.5) + " " + P1_ETT
    assert "penalized 1 point," in prompts.prompt_text("P2lo")


def test_penalty_t_matches_sentence():
    assert prompts.penalty_t("P0") is None
    assert prompts.penalty_t("P1") is None
    assert prompts.penalty_t("P2hi") == pytest.approx(0.9)
    assert prompts.penalty_t("P2lo") == pytest.approx(0.5)
    for pid in ("P2hi", "P2lo"):
        t = prompts.penalty_t(pid)
        assert prompts.prompt_text(pid).startswith(penalty_sentence(t))


@pytest.mark.parametrize(
    ("device", "expected_p0"),
    [
        ("ett", P0_ETT),
        ("ngt", "Is a nasogastric or other enteric feeding tube present in this chest X-ray? "
                "Answer with a single word: Yes or No."),
        ("cvc", "Is a central venous catheter, including a PICC line, present in this chest X-ray? "
                "Answer with a single word: Yes or No."),
        ("swan", "Is a Swan-Ganz catheter present in this chest X-ray? Answer with a single word: Yes or No."),
    ],
)
def test_device_substitution_p0(device, expected_p0):
    assert prompts.prompt_text("P0", device) == expected_p0


@pytest.mark.parametrize("pid", ["P0", "P1", "P2hi", "P2lo"])
@pytest.mark.parametrize("device", ["ett", "ngt", "cvc", "swan"])
def test_every_prompt_reads_grammatically(pid, device):
    text = prompts.prompt_text(pid, device)
    assert "{" not in text and "}" not in text
    for bad in ("  ", ",,", " ,", " ?", ",?", ", ?", " .", ".."):
        assert bad not in text, (bad, text)
    assert text.count("?") == 1
    assert text == text.strip()
    # article agrees with the first sound of the device phrase
    m = re.search(r"\bIs (an?) (\w)", text)
    assert m is not None
    article, first = m.groups()
    assert article == ("an" if first.lower() in "aeiou" else "a")
    assert "present in this chest X-ray?" in text


def test_sha256_is_utf8_and_stable():
    assert prompts.prompt_sha256("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert prompts.prompt_sha256("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    assert prompts.prompt_sha256("café ’") == hashlib.sha256("café ’".encode("utf-8")).hexdigest()
    # pinned: any edit of the frozen prompt files changes these hashes and fails here
    assert prompts.prompt_sha256(prompts.prompt_text("P0")) == (
        "972e9fbdaf3b7ce5c47892b04d55e01518b014285e1d4a678d380e62d896abf0"
    )
    assert prompts.prompt_sha256(prompts.prompt_text("P2hi")) == (
        "0561672cfd2f308742ff1c8aa41a0587283d2595f376b364339d7b42ee5429e5"
    )
    with pytest.raises(TypeError):
        prompts.prompt_sha256(b"abc")


def test_unknown_ids_raise_keyerror():
    with pytest.raises(KeyError):
        prompts.prompt_text("P3")
    with pytest.raises(KeyError):
        prompts.prompt_text("p0")
    with pytest.raises(KeyError):
        prompts.penalty_t("P9")
    with pytest.raises(KeyError):
        prompts.prompt_text("P0", "chest_tube")


def test_load_prompts_returns_fresh_copy():
    a = prompts.load_prompts()
    a["prompts"]["P0"]["template"] = "changed {device}"
    assert prompts.load_prompts()["prompts"]["P0"]["template"] != "changed {device}"
    assert prompts.prompt_text("P0") == P0_ETT


def test_load_prompts_custom_path_and_validation(tmp_path):
    good = {"devices": {"ett": "an endotracheal tube"},
            "prompts": {"PX": {"template": "Is {device} visible?", "t": None}}}
    path = tmp_path / "custom_prompts.json"
    path.write_text(json.dumps(good), encoding="utf-8")
    assert prompts.load_prompts(path) == good
    assert prompts.load_prompts(str(path)) == good

    bad_cases = [
        {"devices": {}, "prompts": good["prompts"]},
        {"devices": good["devices"], "prompts": {"PX": {"template": "no placeholder", "t": None}}},
        {"devices": good["devices"], "prompts": {"PX": {"template": "{device} {device}", "t": None}}},
        {"devices": good["devices"], "prompts": {"PX": {"template": "Is {device}?", "t": 1.5}}},
        {"devices": good["devices"], "prompts": {"PX": {"template": "Is {device}?"}}},
        ["not", "an", "object"],
    ]
    for i, bad in enumerate(bad_cases):
        p = tmp_path / f"bad_{i}.json"
        p.write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises(ValueError):
            prompts.load_prompts(p)
