"""Cached first-token signal helpers (PROTOCOL.md section 3, "Cached first-token signal").

At the first generated position the runner takes the raw logits over the vocabulary (never
the processed scores), converts them with ``log_softmax`` and calls ``first_token_signal``:

    z = logsumexp(logprobs[yes_ids]) - logsumexp(logprobs[no_ids])
    coverage = P(yes forms) + P(no forms)

where the id lists come from ``single_token_ids(encode, constants.YES_FORMS / NO_FORMS)``.
``p_yes_normalised(z)`` is P(yes) renormalised over the yes and no forms. A bf16 torch tensor
has no numpy dtype: convert it with ``t.float().cpu().numpy()`` before calling these helpers.

This module imports only the standard library and numpy: it runs on the GPU pod.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

import numpy as np

#: Tolerance on |logsumexp(logprobs)| accepted by ``first_token_signal`` (float32 log-softmax
#: over a 150k vocabulary is normalised to about 1e-5; raw logits are off by far more).
NORMALISATION_TOL = 1e-3


def log_softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable log-softmax over the last axis, returned as float64.

    The maximum is subtracted before exponentiating, so logits around 1e4 do not overflow.
    ``-inf`` entries (masked tokens) get log-probability ``-inf``; a row that is entirely
    ``-inf`` has no distribution and comes back as NaN; NaN inputs propagate.
    """
    x = np.asarray(logits, dtype=np.float64)
    if x.ndim == 0 or x.shape[-1] == 0:
        raise ValueError("logits must have at least one entry along the last axis")
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.max(x, axis=-1, keepdims=True)
        m = np.where(np.isfinite(m), m, 0.0)
        shifted = x - m
        lse = np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
        return shifted - lse


def logsumexp(x: np.ndarray) -> float:
    """log(sum(exp(x))) over all elements, computed stably in float64.

    Returns ``-inf`` for an empty input or an all ``-inf`` input, ``+inf`` if any element is
    ``+inf``, and NaN if any element is NaN.
    """
    a = np.asarray(x, dtype=np.float64).ravel()
    if a.size == 0:
        return float("-inf")
    if np.isnan(a).any():
        return float("nan")
    m = float(np.max(a))
    if not np.isfinite(m):
        return m
    return m + float(np.log(np.sum(np.exp(a - m))))


def single_token_ids(encode: Callable[[str], list[int]], forms: Sequence[str]) -> tuple[list[int], list[str]]:
    """Token ids of the surface forms that encode to exactly one token.

    ``encode`` maps a string to its token ids without special tokens (for a Hugging Face
    tokenizer: ``lambda s: tok.encode(s, add_special_tokens=False)``). Returns ``(ids, skipped)``:
    ``ids`` are the unique single-token ids in order of first appearance (two forms that share
    an id contribute it once) and ``skipped`` are the forms that encode to zero or several
    tokens, in input order without repeats. Both lists are logged with the run.
    """
    if isinstance(forms, str):
        raise TypeError("forms must be a sequence of strings, not a single string")
    ids: list[int] = []
    skipped: list[str] = []
    for form in forms:
        toks = [int(t) for t in encode(form)]
        if len(toks) == 1:
            if toks[0] not in ids:
                ids.append(toks[0])
        elif form not in skipped:
            skipped.append(form)
    return ids, skipped


def _id_array(ids: Iterable[int], vocab_size: int, name: str) -> np.ndarray:
    """Unique, validated integer ids (negative ids would silently wrap around in numpy)."""
    arr = np.asarray(list(ids))
    if arr.size == 0:
        return np.zeros(0, dtype=np.int64)
    if arr.dtype.kind not in "iu":
        if arr.dtype.kind == "f" and np.all(np.isfinite(arr)) and np.all(arr == np.round(arr)):
            arr = arr.astype(np.int64)
        else:
            raise TypeError(f"{name} must be integers")
    arr = np.unique(arr.astype(np.int64))
    if arr[0] < 0 or arr[-1] >= vocab_size:
        raise ValueError(f"{name} out of range for a vocabulary of {vocab_size}: {arr.tolist()}")
    return arr


def first_token_signal(
    logprobs: np.ndarray,
    yes_ids: Iterable[int],
    no_ids: Iterable[int],
    abstain_ids: Iterable[int] = (),
) -> dict:
    """First-token yes/no signal from one vector of log-probabilities.

    ``logprobs`` is the log-softmax of the raw first-step logits, shape (V,) or (1, V).
    Returns floats ``logp_yes`` and ``logp_no`` (logsumexp over the ids), ``logp_abstain``
    (NaN if no abstain ids; exploratory only), ``z = logp_yes - logp_no`` and
    ``coverage = exp(logp_yes) + exp(logp_no)``.

    Duplicate ids are counted once. Raises ValueError when ``yes_ids`` or ``no_ids`` is empty,
    when the two lists share an id, when an id is outside the vocabulary, when ``logprobs``
    contains NaN, or when it is not normalised (|logsumexp| > ``NORMALISATION_TOL``, the usual
    sign that raw logits were passed instead of ``log_softmax(logits)``).
    """
    lp = np.asarray(logprobs, dtype=np.float64)
    if lp.ndim == 2 and lp.shape[0] == 1:
        lp = lp[0]
    if lp.ndim != 1 or lp.size == 0:
        raise ValueError(f"logprobs must be a non-empty vector of shape (V,) or (1, V), got {np.shape(logprobs)}")
    if np.isnan(lp).any():
        raise ValueError("logprobs contains NaN")
    total = logsumexp(lp)
    if not abs(total) <= NORMALISATION_TOL:
        raise ValueError(f"logprobs are not normalised (logsumexp = {total:.6g}); pass log_softmax(logits)")
    vocab = lp.size
    yes = _id_array(yes_ids, vocab, "yes_ids")
    no = _id_array(no_ids, vocab, "no_ids")
    abstain = _id_array(abstain_ids, vocab, "abstain_ids")
    if yes.size == 0 or no.size == 0:
        raise ValueError("yes_ids and no_ids must both be non-empty (no single-token surface form found)")
    shared = np.intersect1d(yes, no)
    if shared.size:
        raise ValueError(f"yes_ids and no_ids share token ids {shared.tolist()}")
    logp_yes = logsumexp(lp[yes])
    logp_no = logsumexp(lp[no])
    logp_abstain = logsumexp(lp[abstain]) if abstain.size else float("nan")
    return {
        "logp_yes": logp_yes,
        "logp_no": logp_no,
        "logp_abstain": logp_abstain,
        "z": logp_yes - logp_no,
        "coverage": float(np.exp(logp_yes) + np.exp(logp_no)),
    }


def p_yes_normalised(z: float | np.ndarray) -> float | np.ndarray:
    """Logistic sigmoid 1 / (1 + exp(-z)): P(yes) renormalised over the yes and no forms.

    Stable for large |z| (no overflow), maps +inf to 1.0 and -inf to 0.0, propagates NaN.
    A scalar input returns a float; an array input returns an array of the same shape.
    """
    arr = np.asarray(z, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        e = np.exp(-np.abs(arr))  # in [0, 1]: never overflows; NaN stays NaN
        out = np.where(arr >= 0, 1.0 / (1.0 + e), e / (1.0 + e))
    if out.ndim == 0:
        return float(out)
    return out
