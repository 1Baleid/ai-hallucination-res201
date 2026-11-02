"""Run one vision-language model over the pod item list and write S7 JSONL records.

Usage, from the repository root on the pod::

    python pod/run_vlm.py --model-key qwen3vl2b --items items/pod_items.csv --images-dir images \
        --out runs/<run_id>/qwen3vl2b.jsonl --run-id <run_id> [--signal auto|first_token|full_sequence]
    python pod/run_vlm.py --model-key qwen3vl2b --blind --out runs/<run_id>/qwen3vl2b.jsonl --run-id <run_id>
    python pod/run_vlm.py --model-key qwen3vl2b --smoke-only --items ... --images-dir ... \
        --out runs/<run_id>/smoke/qwen3vl2b.smoke.jsonl --run-id <run_id>-smoke \
        --smoke-report runs/<run_id>/smoke_report.json

Modes (PIPELINE_SPEC section 5, PROTOCOL.md sections 3 and 4):

* items (default): every item under the prompts of its ``prompt_set`` ("full" = P0, P1, P2hi on
  the image; "p0" = P0 only), condition "image".
* ``--blind``: the BLIND-text rows (prompt without an image) and the BLIND-grey rows (a generated
  1024 x 1024 PNG at grey level 128) under P0, P1 and P2hi. Under greedy decoding each is one
  deterministic generation per model and prompt.
* ``--smoke-only``: the smoke items (``smoke == 1``, dev pool) under P0 with the first-token signal;
  writes the model's section of ``smoke_report.json`` (visual tokens, median coverage, decision,
  s/item, peak GPU memory). The smoke JSONL is started afresh (an older one is kept as ``.prev``).

Decoding: the model's own chat template with no system prompt, the image first and then the prompt
text in one user turn; greedy (``do_sample=False``, ``num_beams=1``, ``repetition_penalty=1.0``),
``max_new_tokens=16``, the sampling fields of the model's generation config cleared, and
``output_logits=True`` with ``return_dict_in_generate=True``. The signal comes from the raw
first-step logits (never the processed scores): log-softmax in float64, then
``cxrdev.logits.first_token_signal`` over the single-token yes and no surface forms. With the
full-sequence signal (``--signal full_sequence``, or ``--signal auto`` when the smoke test's median
coverage was below 0.90), ``logp_yes`` and ``logp_no`` are the teacher-forced log-likelihoods of
the completions "Yes" and "No" closed by the end-of-turn token, appended to the generation prompt
(two forward passes per record); ``logp_abstain`` stays the first-token value (exploratory).

Every record is checked before it is written: the visual-token count is asserted per model
(``image_grid_thw`` for Qwen: 1,024 at 1024 x 1024 with min = max pixels = 1,048,576; 256 image
tokens for MedGemma), and a failure stops the run with exit code 3. Runs resume from the keys
(item_id, prompt_id, condition) already in ``--out``; a file written with another configuration
(run id, model, revision, signal, versions, prompts) is refused with exit code 4.

Progress lines show counts and timing only, never answers or scores, except in smoke mode (dev-pool
items), so the logs can be watched before the blind verification (PROTOCOL.md section 2).
"""
from __future__ import annotations

import argparse
import platform
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from cxrdev import constants  # noqa: E402
from cxrdev.fileio import append_jsonl, sha256_bytes, utc_now  # noqa: E402
from cxrdev.parse import PARSER_VERSION, parse_answer  # noqa: E402
from cxrdev.prompts import prompt_sha256, prompt_text  # noqa: E402
from pod import common  # noqa: E402
from pod.common import (  # noqa: E402
    CONDITION_BLIND_GREY,
    CONDITION_BLIND_TEXT,
    CONDITION_IMAGE,
    DEVICE_KEY,
    END_OF_TURN_TOKENS,
    FULL_SEQUENCE_ANSWERS,
    QWEN_FAMILIES,
    PodItem,
    PodItemsError,
    ResumeConflictError,
    SmokeReportError,
    Task,
    VisualTokenAssertionError,
)

#: Generation-config fields that only matter for sampling; cleared on the model's config so that
#: greedy decoding inherits nothing from a checkpoint's sampling defaults.
SAMPLING_FIELDS: tuple[str, ...] = (
    "temperature", "top_k", "top_p", "min_p", "top_h", "typical_p", "epsilon_cutoff", "eta_cutoff",
)
#: Per-token inputs extended with zeros when completion tokens are appended (text tokens).
PER_TOKEN_ZERO_KEYS: tuple[str, ...] = ("mm_token_type_ids", "token_type_ids")
GREY_CACHE_KEY = "__BLIND-GREY__"


@dataclass
class GenerationResult:
    """The greedy generation of one record."""

    new_ids: list[int]          # generated ids up to and including the first end-of-sequence token
    first_logits: np.ndarray    # raw logits of the first generated position, shape (V,), float32
    raw_output: str             # decoded new tokens without special tokens
    n_generated_tokens: int
    ended_with_eos: bool


def _size_value(size: Any, key: str) -> Any:
    """Read ``key`` from an image-processor size (a dict or a transformers SizeDict)."""
    if isinstance(size, Mapping):
        return size.get(key)
    return getattr(size, key, None)


def configure_processor(processor: Any, spec: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the registered image settings to a processor and return them for the run log.

    Qwen families: ``image_processor.size = {"shortest_edge": min_pixels, "longest_edge": max_pixels}``
    (the transformers 5 form of ``min_pixels`` / ``max_pixels``), read back to confirm. Models with a
    fixed ``image_size`` (MedGemma, 896 px): the processor's size must already be that square and
    pan-and-scan must be off, otherwise the token count would change. Raises ValueError otherwise.
    """
    ip = getattr(processor, "image_processor", None)
    settings: dict[str, Any] = {
        "processor_class": type(processor).__name__,
        "image_processor_class": type(ip).__name__ if ip is not None else None,
    }
    if spec.get("min_pixels") is not None or spec.get("max_pixels") is not None:
        if ip is None:
            raise ValueError("the processor has no image_processor to configure")
        lo, hi = int(spec["min_pixels"]), int(spec["max_pixels"])
        ip.size = {"shortest_edge": lo, "longest_edge": hi}
        got = (_size_value(ip.size, "shortest_edge"), _size_value(ip.size, "longest_edge"))
        if got != (lo, hi):
            raise ValueError(f"image processor size did not take min/max pixels ({lo}, {hi}); got {got}")
        settings.update(
            min_pixels=lo, max_pixels=hi,
            patch_size=getattr(ip, "patch_size", None), merge_size=getattr(ip, "merge_size", None),
        )
    if spec.get("image_size") is not None:
        if ip is None:
            raise ValueError("the processor has no image_processor to check")
        side = int(spec["image_size"])
        got = (_size_value(ip.size, "height"), _size_value(ip.size, "width"))
        if got != (side, side):
            raise ValueError(f"image processor size is {got}; the registered input is {side} x {side}")
        if getattr(ip, "do_pan_and_scan", None):
            raise ValueError("pan-and-scan must be off (it adds image tokens)")
        settings.update(image_size=side)
    return settings


def make_generation_config(model: Any, *, max_new_tokens: int, eos_ids: Sequence[int], pad_id: int) -> Any:
    """Greedy generation config with raw logits; clears the model's sampling defaults first.

    ``model.generation_config`` loses its sampling fields (temperature, top-k, top-p, ...) and gets
    ``do_sample=False``, so transformers cannot fill them back in from the checkpoint; the returned
    ``GenerationConfig`` sets ``do_sample=False``, ``num_beams=1``, ``repetition_penalty=1.0``,
    ``max_new_tokens``, ``output_logits=True``, ``output_scores=False``,
    ``return_dict_in_generate=True`` and the end-of-sequence and pad ids.
    """
    from transformers import GenerationConfig

    base = getattr(model, "generation_config", None)
    if base is not None:
        for name in SAMPLING_FIELDS:
            if hasattr(base, name):
                setattr(base, name, None)
        base.do_sample = False
        if hasattr(base, "num_beams"):
            base.num_beams = 1
        if hasattr(base, "repetition_penalty"):
            base.repetition_penalty = 1.0
    return GenerationConfig(
        do_sample=False,
        num_beams=1,
        repetition_penalty=1.0,
        max_new_tokens=int(max_new_tokens),
        output_logits=True,
        output_scores=False,
        return_dict_in_generate=True,
        eos_token_id=[int(e) for e in eos_ids],
        pad_token_id=int(pad_id),
    )


class VLMBackend:
    """A loaded vision-language model and its processor, with the operations the runner needs.

    ``model`` and ``processor`` are a transformers ``AutoModelForImageTextToText`` and
    ``AutoProcessor`` pair (``load_backend``), or test doubles with the same methods:
    ``processor.apply_chat_template``, ``processor.tokenizer`` (``encode``, ``decode``,
    ``convert_tokens_to_ids``, ``eos_token_id``), ``model.generate``, ``model(...)``,
    ``model.config.image_token_id`` and ``model.generation_config``.
    """

    def __init__(
        self,
        model: Any,
        processor: Any,
        model_key: str,
        *,
        revision: str,
        dtype_name: str,
        device: str,
        transformers_version: str,
        torch_version: str,
        model_id: str | None = None,
        max_new_tokens: int = constants.MAX_NEW_TOKENS,
        seed: int | None = None,
    ) -> None:
        if model_key not in constants.MODELS:
            raise KeyError(f"unknown model key {model_key!r}; known: {sorted(constants.MODELS)}")
        self.model_key = model_key
        self.spec = dict(constants.MODELS[model_key])
        self.family = str(self.spec["family"])
        self.model = model
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.model_id = model_id or str(self.spec["hf_id"])
        self.revision = str(revision)
        self.dtype_name = str(dtype_name)
        self.device = str(device)
        self.transformers_version = str(transformers_version)
        self.torch_version = str(torch_version)
        self.max_new_tokens = int(max_new_tokens)
        self.seed = constants.derive_seed("model") if seed is None else int(seed)
        self.processor_settings = configure_processor(processor, self.spec)
        self.signal_ids = common.signal_ids_from_encoder(self.encode)
        self.eot_id = self._end_of_turn_id()
        self.eos_ids = self._eos_ids()
        self.pad_id = self._pad_id()
        self.generation_config = make_generation_config(
            model, max_new_tokens=self.max_new_tokens, eos_ids=self.eos_ids, pad_id=self.pad_id
        )
        self.image_token_id = int(model.config.image_token_id)
        ip = getattr(processor, "image_processor", None)
        self.merge_size = int(getattr(ip, "merge_size", 1) or 1) if self.family in QWEN_FAMILIES else 1
        self.completions: dict[str, list[int]] = {
            answer: self.encode(answer) + [self.eot_id] for answer in FULL_SEQUENCE_ANSWERS
        }

    # ------------------------------------------------------------------ tokenizer helpers
    def encode(self, text: str) -> list[int]:
        """Token ids of ``text`` without special tokens."""
        return [int(t) for t in self.tokenizer.encode(text, add_special_tokens=False)]

    def _end_of_turn_id(self) -> int:
        token = END_OF_TURN_TOKENS.get(self.family)
        if token is None:
            raise ValueError(f"no end-of-turn token registered for family {self.family!r}")
        tid = self.tokenizer.convert_tokens_to_ids(token)
        unk = getattr(self.tokenizer, "unk_token_id", None)
        if not isinstance(tid, int) or isinstance(tid, bool) or tid < 0 or (unk is not None and tid == unk):
            raise ValueError(f"the tokenizer has no id for the end-of-turn token {token!r} (got {tid!r})")
        return int(tid)

    def _eos_ids(self) -> tuple[int, ...]:
        ids: set[int] = {self.eot_id}
        gen_eos = getattr(getattr(self.model, "generation_config", None), "eos_token_id", None)
        if isinstance(gen_eos, int):
            ids.add(gen_eos)
        elif gen_eos is not None:
            ids.update(int(e) for e in gen_eos)
        tok_eos = getattr(self.tokenizer, "eos_token_id", None)
        if isinstance(tok_eos, int):
            ids.add(tok_eos)
        return tuple(sorted(ids))

    def _pad_id(self) -> int:
        for candidate in (
            getattr(getattr(self.model, "generation_config", None), "pad_token_id", None),
            getattr(self.tokenizer, "pad_token_id", None),
        ):
            if isinstance(candidate, int) and not isinstance(candidate, bool):
                return candidate
        return self.eos_ids[0]

    # ------------------------------------------------------------------ descriptions
    def identity(self) -> dict[str, Any]:
        """The record fields that describe the model and decoding (shared by every record)."""
        return {
            "model_key": self.model_key,
            "model_id": self.model_id,
            "revision": self.revision,
            "transformers_version": self.transformers_version,
            "torch_version": self.torch_version,
            "dtype": self.dtype_name,
            "device": self.device,
            "seed": self.seed,
            "max_new_tokens": self.max_new_tokens,
            "yes_ids": list(self.signal_ids.yes_ids),
            "no_ids": list(self.signal_ids.no_ids),
        }

    def settings(self) -> dict[str, Any]:
        """Run settings for the ``.meta.json`` sidecar (no model outputs)."""
        gen = self.generation_config
        gen_dict = gen.to_dict() if hasattr(gen, "to_dict") else dict(vars(gen))
        try:
            template = self.render_prompt(prompt_text("P0", DEVICE_KEY), with_image=True)
        except Exception as exc:  # noqa: BLE001 - the log entry is informative only
            template = f"<not rendered: {exc!r}>"
        return {
            **self.identity(),
            "family": self.family,
            "processor": self.processor_settings,
            "expected_visual_tokens": self.spec.get("expected_visual_tokens"),
            "abstain_ids": list(self.signal_ids.abstain_ids),
            "skipped_forms": self.signal_ids.skipped,
            "end_of_turn_id": self.eot_id,
            "eos_ids": list(self.eos_ids),
            "pad_id": self.pad_id,
            "full_sequence_completions": {k: list(v) for k, v in self.completions.items()},
            "generation_config": {k: gen_dict[k] for k in sorted(gen_dict)},
            "rendered_p0_template": template,
            "gpu_name": common.gpu_name(self.device),
        }

    # ------------------------------------------------------------------ inputs
    @staticmethod
    def _messages(prompt: str, image: Any, with_image: bool) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = []
        if with_image:
            content.append({"type": "image", "image": image} if image is not None else {"type": "image"})
        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]

    def render_prompt(self, prompt: str, *, with_image: bool) -> str:
        """The chat-template string for one user turn (image placeholder first), no system prompt."""
        return self.processor.apply_chat_template(
            self._messages(prompt, None, with_image), add_generation_prompt=True, tokenize=False
        )

    def build_inputs(self, prompt: str, image: Any | None) -> Any:
        """Tokenised model inputs for one user turn: the image (if any), then the prompt text.

        Uses ``processor.apply_chat_template(..., tokenize=True, return_dict=True)``, which expands
        the image placeholder and avoids a second BOS token for templates that write one.
        """
        return self.processor.apply_chat_template(
            self._messages(prompt, image, image is not None),
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

    def count_visual_tokens(self, inputs: Mapping[str, Any], *, has_image: bool, where: str = "") -> int:
        """Count the visual tokens of one input and assert the registered number.

        Qwen families: tokens = prod(image_grid_thw) / merge_size^2, which must equal the image-pad
        tokens in ``input_ids`` and ``expected_visual_tokens`` (1,024 for the hero). Gemma 3
        (MedGemma): the image soft tokens in ``input_ids`` must equal ``mm_tokens_per_image`` and
        the registered 256. A text-only input must hold no image token (0 is returned). Raises
        VisualTokenAssertionError otherwise.
        """
        ids = inputs["input_ids"]
        n_image_tokens = int((ids == self.image_token_id).sum())
        ctx = f"{self.model_key}" + (f" [{where}]" if where else "")
        if not has_image:
            if n_image_tokens:
                raise VisualTokenAssertionError(f"{ctx}: {n_image_tokens} image tokens in a text-only prompt")
            return 0
        expected = self.spec.get("expected_visual_tokens")
        if self.family in QWEN_FAMILIES:
            grid = inputs.get("image_grid_thw")
            if grid is None:
                raise VisualTokenAssertionError(f"{ctx}: the processor returned no image_grid_thw")
            rows = [[int(x) for x in row] for row in grid.tolist()]
            if len(rows) != 1 or len(rows[0]) != 3:
                raise VisualTokenAssertionError(f"{ctx}: expected one image grid, got {rows}")
            t, h, w = rows[0]
            tokens = (t * h * w) // (self.merge_size ** 2)
            detail = f"image_grid_thw {rows[0]}, merge {self.merge_size}"
            if tokens != n_image_tokens:
                raise VisualTokenAssertionError(
                    f"{ctx}: {detail} gives {tokens} tokens but input_ids hold {n_image_tokens} image tokens"
                )
        else:
            tokens = n_image_tokens
            detail = f"{n_image_tokens} image tokens in input_ids"
            per_image = getattr(getattr(self.model, "config", None), "mm_tokens_per_image", None)
            if per_image is not None and tokens != int(per_image):
                raise VisualTokenAssertionError(f"{ctx}: {detail}, the model expects {per_image} per image")
        if tokens <= 0:
            raise VisualTokenAssertionError(f"{ctx}: no visual tokens ({detail})")
        if expected is not None and tokens != int(expected):
            hint = (
                f" (check min_pixels = max_pixels = {self.spec.get('min_pixels')} and the transformers version)"
                if self.family in QWEN_FAMILIES else " (check the image size and that pan-and-scan is off)"
            )
            raise VisualTokenAssertionError(f"{ctx}: {tokens} visual tokens ({detail}), expected {expected}{hint}")
        return tokens

    def prepare(self, inputs: Mapping[str, Any]) -> dict[str, Any]:
        """Move tensors to the model device; floating tensors (pixel values) take the model dtype."""
        import torch

        dtype = common.torch_dtype(self.dtype_name)
        out: dict[str, Any] = {}
        for key in list(inputs.keys()):
            value = inputs[key]
            if isinstance(value, torch.Tensor):
                value = value.to(self.device)
                if torch.is_floating_point(value):
                    value = value.to(dtype)
            out[key] = value
        return out

    # ------------------------------------------------------------------ model calls
    def generate(self, model_inputs: Mapping[str, Any]) -> GenerationResult:
        """Greedy generation; returns the new tokens, the decoded text and the raw first-step logits."""
        import torch

        torch.manual_seed(self.seed)
        with torch.inference_mode():
            out = self.model.generate(**model_inputs, generation_config=self.generation_config)
        prompt_len = int(model_inputs["input_ids"].shape[1])
        new_ids = [int(t) for t in out.sequences[0, prompt_len:].tolist()]
        raw = getattr(out, "logits", None)
        if not raw:
            raise RuntimeError("generate() returned no raw logits (needs output_logits=True and return_dict_in_generate=True)")
        first = raw[0]
        if first.dim() == 2:
            first = first[0]
        first_np = first.detach().to(torch.float32).cpu().numpy()
        n, ended = common.generation_outcome(new_ids, self.eos_ids)
        kept = new_ids[:n]
        text = self.tokenizer.decode(kept, skip_special_tokens=True)
        return GenerationResult(kept, first_np, str(text), n, ended)

    def completion_logprob(self, model_inputs: Mapping[str, Any], completion: Sequence[int]) -> float:
        """Teacher-forced log-likelihood of ``completion`` appended to the generation prompt (one pass)."""
        import torch

        ids = model_inputs["input_ids"]
        length, k = int(ids.shape[1]), len(completion)
        if k == 0:
            raise ValueError("empty completion")
        comp = torch.tensor([list(completion)], dtype=ids.dtype, device=ids.device)
        batch: dict[str, Any] = {}
        for key, value in model_inputs.items():
            if key == "input_ids":
                batch[key] = torch.cat([ids, comp], dim=1)
            elif isinstance(value, torch.Tensor) and value.dim() == 2 and tuple(value.shape) == tuple(ids.shape):
                if key == "attention_mask":
                    fill = torch.ones((1, k), dtype=value.dtype, device=value.device)
                elif key in PER_TOKEN_ZERO_KEYS:
                    fill = torch.zeros((1, k), dtype=value.dtype, device=value.device)
                else:
                    raise ValueError(f"cannot extend the per-token input {key!r} for teacher forcing")
                batch[key] = torch.cat([value, fill], dim=1)
            else:
                batch[key] = value
        with torch.inference_mode():
            out = self.model(**batch, use_cache=False, logits_to_keep=k + 1)
        logits = out.logits[0]
        if logits.shape[0] == k + 1:
            step = logits[:k]
        elif logits.shape[0] == length + k:
            step = logits[length - 1: length - 1 + k]
        else:
            raise RuntimeError(f"unexpected logits shape {tuple(out.logits.shape)} for {length} + {k} tokens")
        return common.sequence_logprob(step.detach().to(torch.float32).cpu().numpy(), completion)

    def score_full_sequence(self, model_inputs: Mapping[str, Any]) -> dict[str, float]:
        """Full-sequence signal: log P("Yes" + end of turn) and log P("No" + end of turn), z, coverage."""
        yes, no = FULL_SEQUENCE_ANSWERS
        return common.full_sequence_fields(
            self.completion_logprob(model_inputs, self.completions[yes]),
            self.completion_logprob(model_inputs, self.completions[no]),
        )


def load_backend(
    model_key: str,
    *,
    device: str | None = None,
    dtype: str | None = None,
    revision: str | None = None,
    local_files_only: bool = False,
    max_new_tokens: int = constants.MAX_NEW_TOKENS,
) -> VLMBackend:
    """Load a registered model and its processor (bf16 by default) at one resolved commit."""
    from transformers import AutoModelForImageTextToText, AutoProcessor

    spec = constants.MODELS[model_key]
    hf_id = str(spec["hf_id"])
    dtype_name = dtype or str(spec["dtype"])
    dev = common.normalise_device(device)
    rev = common.resolve_model_revision(hf_id, revision, local_files_only=local_files_only)
    processor = AutoProcessor.from_pretrained(hf_id, revision=rev, local_files_only=local_files_only)
    model = AutoModelForImageTextToText.from_pretrained(
        hf_id, revision=rev, dtype=common.torch_dtype(dtype_name), device_map=dev, local_files_only=local_files_only
    )
    model.eval()
    return VLMBackend(
        model, processor, model_key, model_id=hf_id, revision=rev, dtype_name=dtype_name, device=dev,
        max_new_tokens=max_new_tokens, **common.library_versions(),
    )


# ----------------------------------------------------------------------------- records
def process_task(
    backend: VLMBackend,
    task: Task,
    *,
    signal: str,
    run_id: str,
    images_dir: str | Path | None,
    image_cache: dict[str, tuple[Any, str]],
) -> dict[str, Any]:
    """Run one task and return its S7 record (validated, not yet written)."""
    t0 = time.perf_counter()
    if task.condition == CONDITION_IMAGE:
        if images_dir is None:
            raise ValueError("image tasks need --images-dir")
        cached = image_cache.get(task.nih_filename)
        if cached is None:
            for key in [k for k in image_cache if k != GREY_CACHE_KEY]:
                del image_cache[key]  # one NIH image at a time (its prompts are consecutive)
            cached = common.load_image_file(Path(images_dir) / task.nih_filename)
            image_cache[task.nih_filename] = cached
        image, image_sha = cached
    elif task.condition == CONDITION_BLIND_GREY:
        cached = image_cache.get(GREY_CACHE_KEY)
        if cached is None:
            data = common.grey_png_bytes()
            cached = (common.image_from_bytes(data), sha256_bytes(data))
            image_cache[GREY_CACHE_KEY] = cached
        image, image_sha = cached
    elif task.condition == CONDITION_BLIND_TEXT:
        image, image_sha = None, ""
    else:
        raise ValueError(f"condition {task.condition!r} is not run by this script")
    text = prompt_text(task.prompt_id, DEVICE_KEY)
    inputs = backend.build_inputs(text, image)
    visual_tokens = backend.count_visual_tokens(
        inputs, has_image=image is not None, where=f"{task.item_id} {task.prompt_id} {task.condition}"
    )
    model_inputs = backend.prepare(inputs)
    gen = backend.generate(model_inputs)
    first = common.first_token_fields(gen.first_logits, backend.signal_ids)
    if signal == "first_token":
        sig = first
    elif signal == "full_sequence":
        full = backend.score_full_sequence(model_inputs)
        sig = {**full, "logp_abstain": first["logp_abstain"]}
    else:
        raise ValueError(f"unknown signal {signal!r}")
    wall = time.perf_counter() - t0
    values = {
        **backend.identity(),
        "run_id": run_id,
        "item_id": task.item_id,
        "nih_filename": task.nih_filename,
        "image_sha256": image_sha,
        "condition": task.condition,
        "prompt_id": task.prompt_id,
        "prompt_sha256": prompt_sha256(text),
        "prompt_text": text,
        "visual_tokens": visual_tokens,
        "raw_output": gen.raw_output,
        "n_generated_tokens": gen.n_generated_tokens,
        "ended_with_eos": gen.ended_with_eos,
        "parsed": parse_answer(gen.raw_output),
        "parser_version": PARSER_VERSION,
        "logp_yes": sig["logp_yes"],
        "logp_no": sig["logp_no"],
        "logp_abstain": sig["logp_abstain"],
        "z": sig["z"],
        "coverage": sig["coverage"],
        "signal": signal,
        "wall_s": round(wall, 4),
        "timestamp_utc": utc_now(),
    }
    return common.make_record(values)


def run_tasks(
    backend: VLMBackend,
    tasks: Sequence[Task],
    *,
    out_path: str | Path,
    run_id: str,
    signal: str,
    images_dir: str | Path | None = None,
    show_outputs: bool = False,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Run every task not yet in ``out_path`` and append one S7 record per task.

    Existing records must match this run (``common.check_resume``). ``show_outputs`` prints the
    parsed answer and coverage per record (smoke mode only; dev-pool items). Returns a summary
    with ``n_tasks``, ``n_done_before``, ``n_written``, ``seconds`` and ``s_per_record_mean``.
    """
    if signal not in common.SIGNALS:
        raise ValueError(f"unknown signal {signal!r}")
    out = Path(out_path)
    done = common.check_resume(common.existing_records(out), {**backend.identity(), "run_id": run_id, "signal": signal})
    pending = [t for t in tasks if t.key not in done]
    if any(t.condition == CONDITION_IMAGE for t in pending):
        if images_dir is None:
            raise ValueError("image tasks need --images-dir")
        missing = common.missing_images(pending, images_dir)
        if missing:
            raise FileNotFoundError(f"{len(missing)} image(s) missing under {images_dir}, e.g. {missing[:5]}")
    if any(t.condition == CONDITION_BLIND_GREY for t in pending):
        common.write_grey_png(out.parent)
    log(f"{backend.model_key}: {len(tasks)} tasks, {len(tasks) - len(pending)} already done, {len(pending)} to run "
        f"(signal {signal})")
    image_cache: dict[str, tuple[Any, str]] = {}
    started = time.perf_counter()
    walls: list[float] = []
    for i, task in enumerate(pending, start=1):
        rec = process_task(backend, task, signal=signal, run_id=run_id, images_dir=images_dir, image_cache=image_cache)
        append_jsonl(out, rec)
        walls.append(rec["wall_s"])
        if show_outputs:
            log(f"[{i:>4}/{len(pending)}] {task.item_id} {task.prompt_id} parsed={rec['parsed']} "
                f"coverage={rec['coverage']:.3f} z={rec['z']:+.2f} tokens={rec['visual_tokens']} {rec['wall_s']:.2f}s")
        elif i == 1 or i % 25 == 0 or i == len(pending):
            elapsed = time.perf_counter() - started
            eta = elapsed / i * (len(pending) - i)
            log(f"[{i:>5}/{len(pending)}] {elapsed / i:.2f} s/record, elapsed {elapsed / 60:.1f} min, "
                f"eta {eta / 60:.1f} min")
    seconds = time.perf_counter() - started
    return {
        "n_tasks": len(tasks),
        "n_done_before": len(tasks) - len(pending),
        "n_written": len(pending),
        "seconds": round(seconds, 3),
        "s_per_record_mean": round(float(np.mean(walls)), 4) if walls else None,
    }


def _rotate(path: Path) -> Path | None:
    """Move an existing file to ``<stem>.prev<suffix>`` (replacing an older one); return the new path."""
    if not path.exists():
        return None
    prev = path.with_name(path.stem + ".prev" + path.suffix)
    prev.unlink(missing_ok=True)
    path.replace(prev)
    return prev


def run_smoke(
    backend: VLMBackend,
    items: Sequence[PodItem],
    *,
    images_dir: str | Path,
    out_path: str | Path,
    report_path: str | Path,
    run_id: str,
    limit: int | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Smoke test of one model: the smoke items under P0 with the first-token signal.

    Starts ``out_path`` afresh, writes the records, then the model's section of
    ``smoke_report.json``: visual tokens, median coverage and the decision ("first_token" when the
    median is at least 0.90, else "full_sequence"), s/item and peak GPU memory. A failed
    visual-token assertion is written to the report as "fail" and re-raised.
    """
    tasks = common.plan_tasks(items, "smoke", limit)
    if not tasks:
        raise PodItemsError("no smoke items (smoke == 1) in pod_items.csv")
    out = Path(out_path)
    rotated = _rotate(out)
    base: dict[str, Any] = {
        "run_id": run_id,
        "model_id": backend.model_id,
        "revision": backend.revision,
        "transformers_version": backend.transformers_version,
        "torch_version": backend.torch_version,
        "dtype": backend.dtype_name,
        "device": backend.device,
        "gpu_name": common.gpu_name(backend.device),
        "expected_visual_tokens": backend.spec.get("expected_visual_tokens"),
        "smoke_jsonl": out.name,
        "previous_smoke_jsonl": rotated.name if rotated else None,
        "timestamp_utc": utc_now(),
    }
    try:
        summary = run_tasks(
            backend, tasks, out_path=out, run_id=run_id, signal="first_token",
            images_dir=images_dir, show_outputs=True, log=log,
        )
    except VisualTokenAssertionError as exc:
        common.update_smoke_report(
            report_path, backend.model_key,
            {**base, "visual_token_assertion": "fail", "error": str(exc), "decision": None},
        )
        raise
    records = [r for r in common.existing_records(out) if r.get("model_key") == backend.model_key]
    section: dict[str, Any] = {
        **base,
        "visual_token_assertion": "pass",
        **common.smoke_section(records),
        "peak_gpu_mem_gb": common.cuda_peak_gb(backend.device),
        "seconds_total": summary["seconds"],
        "yes_ids": list(backend.signal_ids.yes_ids),
        "no_ids": list(backend.signal_ids.no_ids),
        "abstain_ids": list(backend.signal_ids.abstain_ids),
        "skipped_forms": backend.signal_ids.skipped,
    }
    if section["n_items"] != constants.N_SMOKE:
        section["note"] = f"{section['n_items']} smoke items run; the protocol registers {constants.N_SMOKE}"
    common.update_smoke_report(report_path, backend.model_key, section)
    log(f"{backend.model_key}: smoke median coverage {section['median_coverage']:.3f} -> decision "
        f"{section['decision']}; {section['visual_tokens']} visual tokens; {section['s_per_item_mean']:.2f} s/item")
    if section["decision"] == "full_sequence":
        log(f"{backend.model_key}: median coverage below {constants.COVERAGE_MIN_MEDIAN}: the runs use the "
            "full-sequence signal; record the switch in PROTOCOL.md section 11")
    return section


# ----------------------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    """The command-line interface (PIPELINE_SPEC section 5 plus device, dtype and revision options)."""
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-key", required=True, choices=sorted(constants.MODELS))
    p.add_argument("--items", help="pod_items.csv (schema S6); not needed with --blind")
    p.add_argument("--images-dir", help="directory holding the NIH PNGs; not needed with --blind")
    p.add_argument("--out", required=True, help="JSONL output (one S7 record per line; appended, resumable)")
    p.add_argument("--run-id", required=True)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--smoke-only", action="store_true", help="run the smoke items under P0 and write smoke_report.json")
    mode.add_argument("--blind", action="store_true", help="run the BLIND-text and BLIND-grey rows only")
    p.add_argument("--limit", type=int, default=None, help="first N items only (quick tests)")
    p.add_argument("--signal", choices=common.SIGNAL_CHOICES, default="auto",
                   help="auto = the smoke-test decision for this model (default)")
    p.add_argument("--smoke-report", default=None, help="default: smoke_report.json beside --out")
    p.add_argument("--device", default=None, help="default: cuda:0 if available, else cpu")
    p.add_argument("--dtype", default=None, choices=("bfloat16", "float16", "float32"),
                   help="default: the registered dtype (bfloat16)")
    p.add_argument("--revision", default=None, help="Hub revision; default: the default branch, resolved to a commit")
    p.add_argument("--max-new-tokens", type=int, default=constants.MAX_NEW_TOKENS)
    p.add_argument("--local-files-only", action="store_true", help="never contact the Hub (models pre-downloaded)")
    return p


def _cli(args: argparse.Namespace, factory: Callable[..., VLMBackend]) -> int:
    mode = "smoke" if args.smoke_only else ("blind" if args.blind else "items")
    if mode != "blind" and (not args.items or not args.images_dir):
        print("error: --items and --images-dir are required unless --blind is given", file=sys.stderr)
        return common.EXIT_USAGE
    if args.limit is not None and args.limit < 1:
        print("error: --limit must be positive", file=sys.stderr)
        return common.EXIT_USAGE
    spec = constants.MODELS[args.model_key]
    dtype_name = args.dtype or str(spec["dtype"])
    out = Path(args.out)
    report = Path(args.smoke_report) if args.smoke_report else out.parent / "smoke_report.json"
    items = common.read_pod_items(args.items) if mode != "blind" else []
    tasks = common.plan_tasks(items, mode, args.limit)
    if not tasks:
        raise PodItemsError("no tasks to run (no smoke items?)")
    if mode == "smoke":
        signal, source = "first_token", "smoke"
    else:
        signal, source = common.resolve_signal(args.signal, report, args.model_key)
    if mode != "blind":
        missing = common.missing_images(tasks, args.images_dir)
        if missing:
            raise FileNotFoundError(f"{len(missing)} image(s) missing under {args.images_dir}, e.g. {missing[:5]}")
    if mode != "smoke":  # check the static fields before spending time on loading the model
        static = {"run_id": args.run_id, "model_key": args.model_key, "model_id": str(spec["hf_id"]),
                  "dtype": dtype_name, "max_new_tokens": args.max_new_tokens, "signal": signal}
        done = common.check_resume(common.existing_records(out), static)
        if all(t.key in done for t in tasks):
            print(f"{args.model_key}: all {len(tasks)} {mode} tasks already in {out}; nothing to do")
            return common.EXIT_OK
    started = utc_now()
    backend = factory(
        args.model_key, device=args.device, dtype=dtype_name, revision=args.revision,
        local_files_only=args.local_files_only, max_new_tokens=args.max_new_tokens,
    )
    if mode == "smoke":
        section = run_smoke(backend, items, images_dir=args.images_dir, out_path=out, report_path=report,
                            run_id=args.run_id, limit=args.limit)
        summary: dict[str, Any] = {k: section[k] for k in ("n_items", "median_coverage", "decision", "s_per_item_mean")}
    else:
        summary = run_tasks(backend, tasks, out_path=out, run_id=args.run_id, signal=signal, images_dir=args.images_dir)
    common.append_session_meta(out, {
        "started_utc": started,
        "finished_utc": utc_now(),
        "mode": mode,
        "run_id": args.run_id,
        "signal": signal,
        "signal_source": source,
        "smoke_report": str(report) if source != "cli" else None,
        "items_file": str(args.items) if args.items else None,
        "limit": args.limit,
        "summary": summary,
        "backend": backend.settings(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    })
    print(f"{args.model_key}: {mode} finished: {summary}")
    return common.EXIT_OK


def main(argv: Sequence[str] | None = None, *, backend_factory: Callable[..., VLMBackend] | None = None) -> int:
    """Entry point; returns the exit code (3 = visual-token assertion, 4 = resume conflict, 2 = input error)."""
    args = build_parser().parse_args(argv)
    try:
        return _cli(args, backend_factory or load_backend)
    except VisualTokenAssertionError as exc:
        print(f"VISUAL-TOKEN ASSERTION FAILED: {exc}", file=sys.stderr)
        return common.EXIT_TOKEN_ASSERTION
    except ResumeConflictError as exc:
        print(f"RESUME CONFLICT: {exc}", file=sys.stderr)
        return common.EXIT_RESUME_CONFLICT
    except (PodItemsError, SmokeReportError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return common.EXIT_USAGE


if __name__ == "__main__":
    sys.exit(main())
