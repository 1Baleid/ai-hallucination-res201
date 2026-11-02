"""Tests for cxrdev.logits (PROTOCOL.md section 3, "Cached first-token signal")."""
from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from cxrdev import constants, logits


# ----------------------------------------------------------------------------- log_softmax / logsumexp
def test_log_softmax_normalises_and_matches_direct_formula():
    x = np.array([2.0, -1.0, 0.5, 3.25, -7.0])
    lp = logits.log_softmax(x)
    assert lp.dtype == np.float64
    assert np.exp(lp).sum() == pytest.approx(1.0, abs=1e-12)
    direct = np.log(np.exp(x) / np.exp(x).sum())
    np.testing.assert_allclose(lp, direct, rtol=0, atol=1e-12)


def test_log_softmax_stable_for_large_logits():
    base = np.array([0.0, -1.0, -5.0, -20000.0])
    x = base + 1e4
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any overflow RuntimeWarning fails the test
        with np.errstate(over="raise"):  # (underflow of exp(-20000) to 0 is benign)
            lp = logits.log_softmax(x)
            lp_neg = logits.log_softmax(base - 1e4)
    assert np.all(np.isfinite(lp))
    assert np.exp(lp).sum() == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_allclose(lp, logits.log_softmax(base), atol=1e-12)
    np.testing.assert_allclose(lp_neg, logits.log_softmax(base), atol=1e-12)
    # float32 / float16 inputs are promoted to float64
    assert logits.log_softmax(x.astype(np.float32)).dtype == np.float64


def test_log_softmax_rows_and_masked_tokens():
    x = np.array([[1.0, 2.0, 3.0], [1e4, 1e4, -np.inf]])
    lp = logits.log_softmax(x)
    np.testing.assert_allclose(np.exp(lp).sum(axis=1), [1.0, 1.0], atol=1e-12)
    assert lp[1, 2] == -np.inf
    assert lp[1, 0] == pytest.approx(math.log(0.5))
    all_masked = logits.log_softmax(np.array([-np.inf, -np.inf]))
    assert np.all(np.isnan(all_masked))
    with pytest.raises(ValueError):
        logits.log_softmax(np.array([]))


def test_logsumexp():
    assert logits.logsumexp(np.log([0.2, 0.3])) == pytest.approx(math.log(0.5))
    assert logits.logsumexp(np.array([1e4, 1e4])) == pytest.approx(1e4 + math.log(2.0))
    assert logits.logsumexp(np.array([])) == -math.inf
    assert logits.logsumexp(np.array([-np.inf, -np.inf])) == -math.inf
    assert logits.logsumexp(np.array([0.0, np.inf])) == math.inf
    assert math.isnan(logits.logsumexp(np.array([0.0, np.nan])))
    assert logits.logsumexp(np.array([[0.0, 0.0], [0.0, 0.0]])) == pytest.approx(math.log(4.0))
    assert isinstance(logits.logsumexp(np.array([1.0])), float)


# ----------------------------------------------------------------------------- single_token_ids
class FakeTokenizer:
    """Encodes some surface forms to one token, others to two or zero tokens."""

    table = {
        "Yes": [9454],
        "yes": [9693],
        " Yes": [7414],
        " yes": [9693],        # shares the id of "yes" (duplicate is removed)
        "YES": [14328, 1570],  # two tokens: skipped
        " YES": [],            # zero tokens: skipped
        "No": [2822],
        "no": [2152],
        " No": [2308],
        " no": [912],
        "NO": [9173],
        " NO": [5664, 46],     # two tokens: skipped
    }

    def encode(self, text: str) -> list[int]:
        return list(self.table[text])


def test_single_token_ids_with_fake_encoder():
    tok = FakeTokenizer()
    yes_ids, yes_skipped = logits.single_token_ids(tok.encode, constants.YES_FORMS)
    assert yes_ids == [9454, 9693, 7414]          # order of first appearance, duplicate removed
    assert yes_skipped == ["YES", " YES"]
    no_ids, no_skipped = logits.single_token_ids(tok.encode, constants.NO_FORMS)
    assert no_ids == [2822, 2152, 2308, 912, 9173]
    assert no_skipped == [" NO"]
    assert all(type(i) is int for i in yes_ids + no_ids)


def test_single_token_ids_accepts_array_output_and_repeated_forms():
    ids, skipped = logits.single_token_ids(lambda s: np.array([len(s)] * (1 if s != "ab" else 2)),
                                           ["a", "bb", "a", "ab", "ab", "cc"])
    assert ids == [1, 2]
    assert skipped == ["ab"]
    with pytest.raises(TypeError):
        logits.single_token_ids(lambda s: [1], "Yes")


# ----------------------------------------------------------------------------- first_token_signal
def _probs_vector() -> np.ndarray:
    p = np.array([0.05, 0.30, 0.10, 0.20, 0.15, 0.04, 0.06, 0.10])
    assert p.sum() == pytest.approx(1.0)
    return p


def test_first_token_signal_hand_made_vector():
    p = _probs_vector()
    lp = np.log(p)
    sig = logits.first_token_signal(lp, yes_ids=[1, 3], no_ids=[2, 6], abstain_ids=[4])
    assert set(sig) == {"logp_yes", "logp_no", "logp_abstain", "z", "coverage"}
    assert sig["logp_yes"] == pytest.approx(math.log(0.30 + 0.20))
    assert sig["logp_no"] == pytest.approx(math.log(0.10 + 0.06))
    assert sig["logp_abstain"] == pytest.approx(math.log(0.15))
    assert sig["z"] == pytest.approx(math.log(0.50 / 0.16))
    assert sig["coverage"] == pytest.approx(0.66)
    assert all(isinstance(v, float) for v in sig.values())
    # renormalised P(yes) over the yes and no forms
    assert logits.p_yes_normalised(sig["z"]) == pytest.approx(0.50 / 0.66)


def test_first_token_signal_from_raw_logits_via_log_softmax():
    rng = np.random.default_rng(constants.derive_seed("model"))
    raw = rng.normal(0.0, 3.0, size=1000) + 1e4  # large offset: must not matter
    lp = logits.log_softmax(raw)
    yes, no = [10, 20, 30], [40, 50]
    sig = logits.first_token_signal(lp[None, :], yes, no)  # (1, V) accepted
    p = np.exp(raw - raw.max())
    p /= p.sum()
    assert sig["coverage"] == pytest.approx(p[yes].sum() + p[no].sum(), rel=1e-9)
    assert sig["z"] == pytest.approx(math.log(p[yes].sum() / p[no].sum()), rel=1e-9)


def test_first_token_signal_abstain_nan_and_duplicates():
    lp = np.log(_probs_vector())
    sig = logits.first_token_signal(lp, [1], [2])
    assert math.isnan(sig["logp_abstain"])
    sig_dup = logits.first_token_signal(lp, [1, 1, 3, 3], [2, 2])
    assert sig_dup["logp_yes"] == pytest.approx(math.log(0.50))  # duplicates counted once
    assert sig_dup["coverage"] <= 1.0
    sig_np = logits.first_token_signal(lp, np.array([1, 3]), (2,), abstain_ids=np.array([], dtype=int))
    assert sig_np["z"] == pytest.approx(math.log(0.50 / 0.10))
    assert math.isnan(sig_np["logp_abstain"])


def test_first_token_signal_validation():
    lp = np.log(_probs_vector())
    with pytest.raises(ValueError):
        logits.first_token_signal(lp, [], [2])                 # no yes ids
    with pytest.raises(ValueError):
        logits.first_token_signal(lp, [1], [])                 # no no ids
    with pytest.raises(ValueError):
        logits.first_token_signal(lp, [1, 2], [2])             # shared id
    with pytest.raises(ValueError):
        logits.first_token_signal(lp, [-1], [2])               # negative id would wrap around
    with pytest.raises(ValueError):
        logits.first_token_signal(lp, [8], [2])                # beyond the vocabulary
    with pytest.raises(ValueError):
        logits.first_token_signal(lp + 3.0, [1], [2])          # raw logits, not log-probabilities
    with pytest.raises(ValueError):
        logits.first_token_signal(np.where(np.arange(8) == 0, np.nan, lp), [1], [2])
    with pytest.raises(ValueError):
        logits.first_token_signal(np.stack([lp, lp]), [1], [2])  # a batch of 2 is ambiguous
    with pytest.raises(TypeError):
        logits.first_token_signal(lp, [1.5], [2])


# ----------------------------------------------------------------------------- p_yes_normalised
def test_p_yes_normalised():
    assert logits.p_yes_normalised(0.0) == 0.5
    assert logits.p_yes_normalised(math.log(3.0)) == pytest.approx(0.75)
    assert logits.p_yes_normalised(-math.log(3.0)) == pytest.approx(0.25)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert logits.p_yes_normalised(1000.0) == 1.0
        assert logits.p_yes_normalised(-1000.0) == 0.0
        assert logits.p_yes_normalised(math.inf) == 1.0
        assert logits.p_yes_normalised(-math.inf) == 0.0
    assert math.isnan(logits.p_yes_normalised(float("nan")))
    assert isinstance(logits.p_yes_normalised(1.0), float)
    arr = logits.p_yes_normalised(np.array([-2.0, 0.0, 2.0]))
    np.testing.assert_allclose(arr, 1.0 / (1.0 + np.exp(-np.array([-2.0, 0.0, 2.0]))))
