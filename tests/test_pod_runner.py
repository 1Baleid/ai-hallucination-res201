"""Tests for the GPU-pod runner (pod/common.py, pod/run_vlm.py, pod/smoke_test.py, pod/run_medsiglip.py,
pod/queue.sh, pod/setup_pod.sh).

Synthetic data only: random-noise PNGs with NIH-style names and opaque ids; no RANZCR data.
Three layers:

* fakes (always run): a toy tokenizer, a processor that returns controlled token layouts, and a
  bigram "model" whose raw logits are a fixed table, so every S7 field, resume, the BLIND rows,
  truncation, the visual-token assertion, the coverage decision and the full-sequence arithmetic
  can be checked against values computed independently here;
* the real Qwen3-VL-2B processor files (skipped cleanly when they are not in the local Hugging Face
  cache): yes/no single-token ids, the chat template without a system prompt, image_grid_thw for
  min = max pixels = 1,048,576, and a tiny randomly initialised Qwen3-VL model built from a config
  (no weights downloaded) that runs the real transformers generate / forward paths;
* the shell scripts: syntax check, and queue.sh end to end with a stub interpreter (skipped when no
  usable bash is found).

An opt-in test (``-m cpu_smoke``) runs the real Qwen3-VL-2B weights on CPU if they are cached.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
import subprocess
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from cxrdev import constants  # noqa: E402
from cxrdev.fileio import append_jsonl, read_jsonl, sha256_bytes  # noqa: E402
from cxrdev.parse import PARSER_VERSION, is_truncated  # noqa: E402
from cxrdev.prompts import prompt_sha256, prompt_text  # noqa: E402
from pod import common, run_medsiglip, run_vlm, smoke_test  # noqa: E402
from pod.common import (  # noqa: E402
    POD_ITEM_COLUMNS,
    S7_FIELDS,
    PodItem,
    PodItemsError,
    ResumeConflictError,
    SmokeReportError,
    Task,
    VisualTokenAssertionError,
)

REPO = Path(__file__).resolve().parents[1]
QWEN_ID = "Qwen/Qwen3-VL-2B-Instruct"
V = 64                       # fake vocabulary size
BOS, EOS, EOT, IMG, GEN, PAD = 1, 2, 3, 5, 6, 0
YES_IDS = [10, 11, 12, 13, 14]       # " YES" is multi-token in the fake tokenizer
NO_IDS = [20, 21, 22, 23, 24, 25]
ABSTAIN_IDS = [30, 31, 32, 33, 34, 35, 36, 37]
TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


# ============================================================================= fakes
class FakeTokenizer:
    """Toy tokenizer: the surface forms are single tokens (except " YES"); other text is one id per character."""

    def __init__(self, family: str = "qwen3vl") -> None:
        self.pieces = {
            "Yes": 10, "yes": 11, " Yes": 12, " yes": 13, "YES": 14,
            "No": 20, "no": 21, " No": 22, " no": 23, "NO": 24, " NO": 25,
            "I": 30, " I": 31, "Cannot": 32, "cannot": 33, " cannot": 34, "Unknown": 35, "unknown": 36,
            " unknown": 37,
        }
        self.special = {"<bos>": BOS, "<eos>": EOS, common.END_OF_TURN_TOKENS[family]: EOT, "<img>": IMG,
                        "<gen>": GEN, "<pad>": PAD}
        self.id_to_text = {v: k for k, v in self.pieces.items()}
        self.eos_token_id = EOS
        self.pad_token_id = PAD
        self.unk_token_id = 63

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        if text in self.pieces:
            return [self.pieces[text]]
        return [40 + (ord(c) % 20) for c in text]

    def decode(self, ids, skip_special_tokens=True):
        out = []
        for i in ids:
            i = int(i)
            if i in self.special.values():
                if not skip_special_tokens:
                    out.append(f"<{i}>")
            else:
                out.append(self.id_to_text.get(i, "x"))
        return "".join(out)

    def convert_tokens_to_ids(self, token):
        return self.special.get(token, self.unk_token_id)


class FakeProcessor:
    """Builds input ids [BOS, (IMG x n), text ids..., GEN]; records what it was given."""

    def __init__(self, family="qwen3vl", *, grid=(1, 64, 64), gemma_tokens=256, extra_image_tokens=0,
                 text_only_image_tokens=0, size=None):
        self.family = family
        self.qwen = family in common.QWEN_FAMILIES
        self.tokenizer = FakeTokenizer(family)
        self.grid = tuple(grid)
        self.gemma_tokens = gemma_tokens
        self.extra_image_tokens = extra_image_tokens
        self.text_only_image_tokens = text_only_image_tokens
        default_size = ({"shortest_edge": 56 * 56, "longest_edge": 28 * 28 * 1280} if self.qwen
                        else {"height": 896, "width": 896})
        self.image_processor = SimpleNamespace(size=dict(size or default_size), merge_size=2, patch_size=16,
                                               do_pan_and_scan=False)
        self.calls: list[dict] = []

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=False, return_dict=False,
                            return_tensors=None, **kwargs):
        assert add_generation_prompt is True
        assert len(messages) == 1 and messages[0]["role"] == "user"  # one user turn, no system prompt
        content = messages[0]["content"]
        if not tokenize:
            return "<user>" + "".join("<img>" if c["type"] == "image" else c["text"] for c in content) + "<eot><gen>"
        assert return_dict is True and return_tensors == "pt"
        images = [c.get("image") for c in content if c["type"] == "image"]
        texts = [c["text"] for c in content if c["type"] == "text"]
        self.calls.append({"order": [c["type"] for c in content], "images": images, "texts": texts})
        ids = [BOS]
        if images:
            n = (int(np.prod(self.grid)) // 4 if self.qwen else self.gemma_tokens) + self.extra_image_tokens
            ids += [IMG] * n
        else:
            ids += [IMG] * self.text_only_image_tokens
        ids += self.tokenizer.encode(texts[0])[:12] + [GEN]
        out = {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones(1, len(ids), dtype=torch.long)}
        type_ids = torch.tensor([[1 if i == IMG else 0 for i in ids]])
        if self.qwen:
            out["mm_token_type_ids"] = type_ids
            if images:
                out["image_grid_thw"] = torch.tensor([list(self.grid)])
                out["pixel_values"] = torch.full((4, 6), float(np.asarray(images[0]).mean()) / 255.0)
        else:
            out["token_type_ids"] = type_ids
            if images:
                out["pixel_values"] = torch.full((1, 3, 4, 4), float(np.asarray(images[0]).mean()) / 255.0)
        return out


def make_table(first: dict[int, float], *, after: dict[int, dict[int, float]] | None = None, base=-3.0) -> np.ndarray:
    """Bigram logit table: row t = raw logits after token t; row GEN is the first generated position.

    By default every row except GEN strongly predicts the end of turn (EOT), so generation stops
    after the first token; ``after`` overrides rows to build longer outputs.
    """
    table = np.full((V, V), base, dtype=np.float32)
    table[:, EOT] = 6.0
    table[GEN, EOT] = base
    for tok, val in first.items():
        table[GEN, tok] = val
    for prev, row in (after or {}).items():
        for tok, val in row.items():
            table[prev, tok] = val
    return table


class BigramModel:
    """A fake causal LM: the raw logits for the next token depend only on the previous token."""

    def __init__(self, table: np.ndarray, *, mm_tokens_per_image=None):
        self.table = torch.tensor(table, dtype=torch.float32)
        cfg = {"image_token_id": IMG}
        if mm_tokens_per_image is not None:
            cfg["mm_tokens_per_image"] = mm_tokens_per_image
        self.config = SimpleNamespace(**cfg)
        self.generation_config = transformers.GenerationConfig(
            do_sample=True, temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.05,
            eos_token_id=[EOS], pad_token_id=PAD,
        )
        self.generate_calls: list[dict] = []
        self.forward_calls: list[dict] = []

    def generate(self, input_ids=None, generation_config=None, **kwargs):
        self.generate_calls.append({"config": generation_config, "keys": sorted(kwargs),
                                    "dtypes": {k: v.dtype for k, v in kwargs.items() if hasattr(v, "dtype")}})
        seq = input_ids[0].tolist()
        eos = set(generation_config.eos_token_id)
        logits, scores = [], []
        for _ in range(generation_config.max_new_tokens):
            raw = self.table[seq[-1]].clone().unsqueeze(0)
            logits.append(raw)
            scores.append(torch.zeros_like(raw))  # "processed" scores that must never be used
            nxt = int(torch.argmax(raw[0]))
            seq.append(nxt)
            if nxt in eos:
                break
        return SimpleNamespace(sequences=torch.tensor([seq]), logits=tuple(logits), scores=tuple(scores))

    def __call__(self, input_ids=None, attention_mask=None, use_cache=None, logits_to_keep=0, **kwargs):
        self.forward_calls.append({"ids": input_ids[0].tolist(), "mask": attention_mask[0].tolist(),
                                   "use_cache": use_cache, "keep": logits_to_keep,
                                   "extra": {k: (v[0].tolist() if hasattr(v, "dim") and v.dim() == 2 and
                                                 v.shape == input_ids.shape else None) for k, v in kwargs.items()}})
        out = self.table[input_ids[0]]
        if logits_to_keep:
            out = out[-logits_to_keep:]
        return SimpleNamespace(logits=out.unsqueeze(0))


def make_backend(model_key="qwen3vl2b", *, table=None, processor=None, dtype="float32", revision="a" * 40,
                 model=None):
    family = constants.MODELS[model_key]["family"]
    table = make_table({10: 5.0, 20: 3.0, 30: 1.0}) if table is None else table
    model = model or BigramModel(table, mm_tokens_per_image=256 if family == "gemma3" else None)
    proc = processor or FakeProcessor(family)
    return run_vlm.VLMBackend(model, proc, model_key, revision=revision, dtype_name=dtype, device="cpu",
                              transformers_version="5.22.1", torch_version="2.15.0+cpu")


def fake_factory(**backend_kwargs):
    calls = []

    def factory(model_key, **kwargs):
        calls.append((model_key, kwargs))
        chosen = dict(backend_kwargs.get(model_key, backend_kwargs.get("*", {})))
        chosen.setdefault("dtype", kwargs.get("dtype") or "float32")
        return make_backend(model_key, **chosen)

    factory.calls = calls
    return factory


# ----------------------------------------------------------------------------- numpy reference helpers
def ref_log_softmax(row) -> np.ndarray:
    x = np.asarray(row, dtype=np.float64)
    m = x.max()
    return x - (m + math.log(np.exp(x - m).sum()))


def ref_lse(vals) -> float:
    v = np.asarray(vals, dtype=np.float64)
    m = v.max()
    return float(m + math.log(np.exp(v - m).sum()))


# ----------------------------------------------------------------------------- synthetic files
def item_id(i: int) -> str:
    return f"{i:010x}"


def nih_name(i: int) -> str:
    return f"{i:08d}_000.png"


def write_png(path: Path, seed: int, side: int = 64, mode: str = "L") -> bytes:
    rng = np.random.default_rng(seed)
    if mode == "L":
        im = Image.fromarray(rng.integers(0, 256, (side, side), dtype=np.uint8))
    else:
        im = Image.fromarray(rng.integers(0, 256, (side, side, 4), dtype=np.uint8), mode="RGBA")
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path)
    return path.read_bytes()


def write_items(path: Path, rows, header=POD_ITEM_COLUMNS) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(list(header))
        w.writerows(rows)
    return path


@pytest.fixture
def workspace(tmp_path):
    """Two test items (prompt_set full), three dev items (p0; two smoke) and their PNGs."""
    images = tmp_path / "images"
    rows = []
    for i, (pset, smoke) in enumerate([("full", 0), ("full", 0), ("p0", 1), ("p0", 1), ("p0", 0)], start=1):
        write_png(images / nih_name(i), seed=i)
        rows.append([item_id(i), nih_name(i), pset, smoke])
    items = write_items(tmp_path / "items" / "pod_items.csv", rows)
    return SimpleNamespace(root=tmp_path, images=images, items=items, out=tmp_path / "runs" / "r1" / "qwen3vl2b.jsonl")


# ============================================================================= pod items and tasks
def test_read_pod_items_valid_and_order(workspace):
    items = common.read_pod_items(workspace.items)
    assert [it.item_id for it in items] == [item_id(i) for i in range(1, 6)]
    assert items[2] == PodItem(item_id(3), nih_name(3), "p0", 1)


@pytest.mark.parametrize("extra", ["arm", "uid", "ETT - Normal", "nih_patient_id"])
def test_read_pod_items_refuses_any_extra_column(tmp_path, extra):
    path = write_items(tmp_path / "p.csv", [[item_id(1), nih_name(1), "full", 0, "x"]], header=POD_ITEM_COLUMNS + (extra,))
    with pytest.raises(PodItemsError, match="not allowed on the pod"):
        common.read_pod_items(path)


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (["ABCDEF0123", nih_name(1), "full", 0], "lowercase hex"),
        (["1.2.826.0.1", nih_name(1), "full", 0], "lowercase hex"),
        ([item_id(1), "image.jpg", "full", 0], "NIH PNG"),
        ([item_id(1), nih_name(1), "all", 0], "prompt_set"),
        ([item_id(1), nih_name(1), "full", 2], "smoke"),
        ([item_id(1), nih_name(1), "full", 1], "dev-pool"),
    ],
)
def test_read_pod_items_rejects_bad_values(tmp_path, row, message):
    with pytest.raises(PodItemsError, match=message):
        common.read_pod_items(write_items(tmp_path / "p.csv", [row]))


def test_read_pod_items_duplicates_missing_columns_bom_and_empty(tmp_path):
    dup = write_items(tmp_path / "d.csv", [[item_id(1), nih_name(1), "full", 0], [item_id(1), nih_name(2), "p0", 0]])
    with pytest.raises(PodItemsError, match="duplicate item_id"):
        common.read_pod_items(dup)
    miss = write_items(tmp_path / "m.csv", [[item_id(1), nih_name(1), "full"]], header=POD_ITEM_COLUMNS[:3])
    with pytest.raises(PodItemsError, match="missing column"):
        common.read_pod_items(miss)
    empty = write_items(tmp_path / "e.csv", [])
    with pytest.raises(PodItemsError, match="no rows"):
        common.read_pod_items(empty)
    bom = tmp_path / "b.csv"
    bom.write_bytes(b"\xef\xbb\xbfitem_id,nih_filename,prompt_set,smoke\n" + f"{item_id(7)},{nih_name(7)},p0,1\n".encode())
    assert common.read_pod_items(bom) == [PodItem(item_id(7), nih_name(7), "p0", 1)]
    ragged = tmp_path / "r.csv"
    ragged.write_text(f"item_id,nih_filename,prompt_set,smoke\n{item_id(1)},{nih_name(1)},full,0,extra\n", encoding="utf-8")
    with pytest.raises(PodItemsError, match="more cells"):
        common.read_pod_items(ragged)


def test_plan_tasks_modes_and_limit(workspace):
    items = common.read_pod_items(workspace.items)
    tasks = common.plan_tasks(items, "items")
    assert len(tasks) == 2 * 3 + 3 * 1
    assert [t.prompt_id for t in tasks[:4]] == ["P0", "P1", "P2hi", "P0"]
    assert all(t.condition == "image" for t in tasks)
    assert [t.prompt_id for t in tasks if t.item_id == item_id(5)] == ["P0"]
    smoke = common.plan_tasks(items, "smoke")
    assert [(t.item_id, t.prompt_id) for t in smoke] == [(item_id(3), "P0"), (item_id(4), "P0")]
    assert len(common.plan_tasks(items, "items", limit=1)) == 3
    assert len(common.plan_tasks(items, "smoke", limit=1)) == 1
    blind = common.plan_tasks([], "blind")
    assert [(t.item_id, t.condition, t.prompt_id) for t in blind] == [
        ("BLIND-TEXT", "BLIND-text", "P0"), ("BLIND-TEXT", "BLIND-text", "P1"), ("BLIND-TEXT", "BLIND-text", "P2hi"),
        ("BLIND-GREY", "BLIND-grey", "P0"), ("BLIND-GREY", "BLIND-grey", "P1"), ("BLIND-GREY", "BLIND-grey", "P2hi"),
    ]
    assert all(t.nih_filename == "" for t in blind)
    assert tasks[0].key == (item_id(1), "P0", "image")
    with pytest.raises(ValueError):
        common.plan_tasks(items, "other")
    with pytest.raises(ValueError):
        common.plan_tasks(items, "items", limit=0)
    assert common.missing_images(tasks, workspace.images) == []
    (workspace.images / nih_name(2)).unlink()
    assert common.missing_images(tasks, workspace.images) == [nih_name(2)]


# ============================================================================= images
def test_load_image_file_converts_via_grey_and_hashes_file_bytes(tmp_path):
    data = write_png(tmp_path / nih_name(1), seed=3, side=32, mode="RGBA")
    image, sha = common.load_image_file(tmp_path / nih_name(1))
    assert sha == hashlib.sha256(data).hexdigest()
    assert image.mode == "RGB" and image.size == (32, 32)
    arr = np.asarray(image)
    assert np.array_equal(arr[..., 0], arr[..., 1]) and np.array_equal(arr[..., 1], arr[..., 2])
    with Image.open(io.BytesIO(data)) as im:
        expected = np.asarray(im.convert("L"))
    assert np.array_equal(arr[..., 0], expected)


def test_grey_png_is_deterministic_1024_mid_grey(tmp_path):
    a, b = common.grey_png_bytes(), common.grey_png_bytes()
    assert a == b
    image = common.image_from_bytes(a)
    arr = np.asarray(image)
    assert image.size == (constants.IMAGE_SIDE, constants.IMAGE_SIDE) and image.mode == "RGB"
    assert arr.min() == arr.max() == constants.GREY_LEVEL
    path = common.write_grey_png(tmp_path)
    assert path.name == "BLIND-GREY_1024.png" and path.read_bytes() == a
    with pytest.raises(ValueError):
        common.grey_png_bytes(level=300)


# ============================================================================= signals and arithmetic
def test_signal_ids_from_fake_encoder_skips_multi_token_forms():
    ids = common.signal_ids_from_encoder(FakeTokenizer().encode)
    assert list(ids.yes_ids) == YES_IDS and list(ids.no_ids) == NO_IDS and list(ids.abstain_ids) == ABSTAIN_IDS
    assert ids.skipped["yes"] == [" YES"] and ids.skipped["no"] == []
    with pytest.raises(ValueError, match="no single-token"):
        common.signal_ids_from_encoder(lambda s: [1, 2])
    with pytest.raises(ValueError, match="share"):
        common.signal_ids_from_encoder(lambda s: [7])


def test_first_token_fields_from_raw_logits_match_reference():
    rng = np.random.default_rng(0)
    raw = rng.normal(0, 3, V).astype(np.float32) + 100.0  # raw logits, far from normalised
    ids = common.signal_ids_from_encoder(FakeTokenizer().encode)
    got = common.first_token_fields(raw, ids)
    lp = ref_log_softmax(raw)
    assert got["logp_yes"] == pytest.approx(ref_lse(lp[YES_IDS]), abs=1e-9)
    assert got["logp_no"] == pytest.approx(ref_lse(lp[NO_IDS]), abs=1e-9)
    assert got["logp_abstain"] == pytest.approx(ref_lse(lp[ABSTAIN_IDS]), abs=1e-9)
    assert got["z"] == pytest.approx(got["logp_yes"] - got["logp_no"], abs=1e-12)
    assert got["coverage"] == pytest.approx(np.exp(lp[YES_IDS]).sum() + np.exp(lp[NO_IDS]).sum(), abs=1e-9)
    assert all(isinstance(v, float) for v in got.values())
    # shape (1, V) is accepted too
    assert common.first_token_fields(raw[None, :], ids)["z"] == pytest.approx(got["z"], abs=1e-12)


def test_sequence_logprob_and_full_sequence_fields():
    rng = np.random.default_rng(1)
    steps = rng.normal(0, 2, (3, V))
    targets = [10, 3, 7]
    expected = sum(ref_log_softmax(steps[j])[t] for j, t in enumerate(targets))
    assert common.sequence_logprob(steps, targets) == pytest.approx(expected, abs=1e-10)
    with pytest.raises(ValueError):
        common.sequence_logprob(steps, [10, 3])
    with pytest.raises(ValueError):
        common.sequence_logprob(steps, [10, 3, V])
    bad = steps.copy()
    bad[1, 4] = np.nan
    with pytest.raises(ValueError):
        common.sequence_logprob(bad, targets)
    f = common.full_sequence_fields(-0.5, -2.0)
    assert f["z"] == pytest.approx(1.5) and f["coverage"] == pytest.approx(math.exp(-0.5) + math.exp(-2.0))
    with pytest.raises(ValueError):
        common.full_sequence_fields(float("-inf"), -1.0)


@pytest.mark.parametrize(
    ("ids", "expected"),
    [([10, 3], (2, True)), ([10] * 16, (16, False)), ([10, 2, 0, 0], (2, True)), ([], (0, False)), ([3], (1, True))],
)
def test_generation_outcome(ids, expected):
    assert common.generation_outcome(ids, [2, 3]) == expected


# ============================================================================= backend construction
def test_backend_clears_sampling_fields_and_builds_greedy_config():
    backend = make_backend()
    model_cfg = backend.model.generation_config
    for name in ("temperature", "top_k", "top_p"):
        assert getattr(model_cfg, name) is None
    assert model_cfg.do_sample is False and model_cfg.repetition_penalty == 1.0
    gc = backend.generation_config
    assert isinstance(gc, transformers.GenerationConfig)
    assert gc.do_sample is False and gc.num_beams == 1 and gc.repetition_penalty == 1.0
    assert gc.max_new_tokens == constants.MAX_NEW_TOKENS == 16
    assert gc.output_logits is True and gc.return_dict_in_generate is True and not gc.output_scores
    assert sorted(gc.eos_token_id) == [EOS, EOT] and gc.pad_token_id == PAD
    assert backend.eot_id == EOT and backend.eos_ids == (EOS, EOT)
    assert backend.completions == {"Yes": [10, EOT], "No": [20, EOT]}
    assert backend.seed == constants.derive_seed("model")
    assert backend.processor.image_processor.size == {"shortest_edge": 1024 * 1024, "longest_edge": 1024 * 1024}
    settings = backend.settings()
    assert settings["processor"]["min_pixels"] == settings["processor"]["max_pixels"] == 1048576
    assert settings["rendered_p0_template"].startswith("<user><img>Is an endotracheal tube")
    json.dumps(settings, default=str)


def test_backend_rejects_missing_end_of_turn_token_and_bad_gemma_size():
    proc = FakeProcessor("qwen3vl")
    proc.tokenizer.special.pop("<|im_end|>")
    with pytest.raises(ValueError, match="end-of-turn"):
        make_backend(processor=proc)
    with pytest.raises(ValueError, match="896"):
        make_backend("medgemma15", processor=FakeProcessor("gemma3", size={"height": 448, "width": 448}))
    pan = FakeProcessor("gemma3")
    pan.image_processor.do_pan_and_scan = True
    with pytest.raises(ValueError, match="pan-and-scan"):
        make_backend("medgemma15", processor=pan)
    with pytest.raises(KeyError):
        make_backend("not-a-model")


# ============================================================================= records
def test_image_record_has_every_s7_field_with_reference_values(workspace):
    table = make_table({10: 5.0, 20: 3.0, 30: 1.0, 11: 2.0})
    backend = make_backend(table=table)
    task = Task(item_id(1), nih_name(1), "image", "P1")
    rec = run_vlm.process_task(backend, task, signal="first_token", run_id="r1", images_dir=workspace.images, image_cache={})
    assert list(rec) == list(S7_FIELDS)
    lp = ref_log_softmax(table[GEN])
    text = prompt_text("P1", "ett")
    expected = {
        "run_id": "r1", "model_key": "qwen3vl2b", "model_id": QWEN_ID, "revision": "a" * 40,
        "transformers_version": "5.22.1", "torch_version": "2.15.0+cpu", "dtype": "float32", "device": "cpu",
        "item_id": item_id(1), "nih_filename": nih_name(1),
        "image_sha256": sha256_bytes((workspace.images / nih_name(1)).read_bytes()),
        "condition": "image", "prompt_id": "P1", "prompt_sha256": prompt_sha256(text), "prompt_text": text,
        "visual_tokens": 1024, "seed": constants.derive_seed("model"), "max_new_tokens": 16,
        "raw_output": "Yes", "n_generated_tokens": 2, "ended_with_eos": True, "parsed": "yes",
        "parser_version": PARSER_VERSION, "signal": "first_token", "yes_ids": YES_IDS, "no_ids": NO_IDS,
    }
    for key, value in expected.items():
        assert rec[key] == value, key
    assert rec["logp_yes"] == pytest.approx(ref_lse(lp[YES_IDS]), abs=1e-9)
    assert rec["logp_no"] == pytest.approx(ref_lse(lp[NO_IDS]), abs=1e-9)
    assert rec["logp_abstain"] == pytest.approx(ref_lse(lp[ABSTAIN_IDS]), abs=1e-9)
    assert rec["z"] == pytest.approx(rec["logp_yes"] - rec["logp_no"], abs=1e-12)
    assert rec["coverage"] == pytest.approx(math.exp(rec["logp_yes"]) + math.exp(rec["logp_no"]), abs=1e-12)
    assert rec["wall_s"] >= 0 and TS_RE.match(rec["timestamp_utc"])
    assert isinstance(rec["ended_with_eos"], bool) and isinstance(rec["visual_tokens"], int)
    # the processor got the image first, then the prompt text, as one user turn
    call = backend.processor.calls[-1]
    assert call["order"] == ["image", "text"] and call["texts"] == [text]
    np.testing.assert_array_equal(np.asarray(call["images"][0])[..., 0],
                                  np.asarray(Image.open(workspace.images / nih_name(1)).convert("L")))
    # generate received every processor output and the greedy config
    gen_call = backend.model.generate_calls[-1]
    assert gen_call["keys"] == ["attention_mask", "image_grid_thw", "mm_token_type_ids", "pixel_values"]
    assert gen_call["config"] is backend.generation_config
    # the JSONL round trip keeps every value
    path = workspace.root / "rt.jsonl"
    append_jsonl(path, rec)
    assert read_jsonl(path) == [rec]


def test_record_uses_raw_logits_not_processed_scores(workspace):
    table = make_table({10: 5.0, 20: 3.0})
    backend = make_backend(table=table)
    rec = run_vlm.process_task(backend, Task(item_id(1), nih_name(1), "image", "P0"), signal="first_token",
                               run_id="r", images_dir=workspace.images, image_cache={})
    # the fake's processed scores are all zero, which would give z = log(5/6) and coverage = 11/64
    lp = ref_log_softmax(table[GEN])
    assert rec["z"] == pytest.approx(ref_lse(lp[YES_IDS]) - ref_lse(lp[NO_IDS]), abs=1e-9)
    assert rec["z"] > 1.0 and rec["coverage"] > 0.9


def test_truncated_generation_is_flagged_and_unparsable(workspace):
    table = make_table({10: 5.0}, after={10: {10: 9.0}})  # "Yes" forever, never the end of turn
    backend = make_backend(table=table)
    rec = run_vlm.process_task(backend, Task(item_id(1), nih_name(1), "image", "P0"), signal="first_token",
                               run_id="r", images_dir=workspace.images, image_cache={})
    assert rec["n_generated_tokens"] == 16 and rec["ended_with_eos"] is False
    assert rec["raw_output"] == "Yes" * 16 and rec["parsed"] == "unparsable"
    assert is_truncated(rec["n_generated_tokens"], rec["max_new_tokens"], rec["ended_with_eos"])
    assert common.record_is_truncated(rec)


def test_eos_on_last_allowed_token_is_not_truncation(workspace):
    # 15 distinct tokens chained greedily, then the end of turn as the 16th (last allowed) token
    chain = [20, 22, 23, 25, 12, 13, 31, 34, 37, 40, 41, 42, 43, 44, 45]
    after = {prev: {nxt: 9.0} for prev, nxt in zip(chain, chain[1:])}
    table = make_table({chain[0]: 5.0}, after=after)
    backend = make_backend(table=table)
    rec = run_vlm.process_task(backend, Task(item_id(1), nih_name(1), "image", "P0"), signal="first_token",
                               run_id="r", images_dir=workspace.images, image_cache={})
    assert rec["n_generated_tokens"] == 16 and rec["ended_with_eos"] is True
    assert not common.record_is_truncated(rec)
    assert rec["raw_output"] == "No No no NO Yes yes I cannot unknownxxxxxx" and rec["parsed"] == "no"


def test_full_sequence_signal_uses_two_teacher_forced_passes(workspace):
    table = make_table({10: 2.0, 20: 2.5, 30: 1.0}, after={10: {EOT: 4.0}, 20: {EOT: 1.0, 7: 2.0}})
    backend = make_backend(table=table)
    rec = run_vlm.process_task(backend, Task(item_id(1), nih_name(1), "image", "P0"), signal="full_sequence",
                               run_id="r", images_dir=workspace.images, image_cache={})
    lp_gen = ref_log_softmax(table[GEN])
    exp_yes = lp_gen[10] + ref_log_softmax(table[10])[EOT]
    exp_no = lp_gen[20] + ref_log_softmax(table[20])[EOT]
    assert rec["signal"] == "full_sequence"
    assert rec["logp_yes"] == pytest.approx(exp_yes, abs=1e-9)
    assert rec["logp_no"] == pytest.approx(exp_no, abs=1e-9)
    assert rec["z"] == pytest.approx(exp_yes - exp_no, abs=1e-9)
    assert rec["coverage"] == pytest.approx(math.exp(exp_yes) + math.exp(exp_no), abs=1e-9)
    assert rec["logp_abstain"] == pytest.approx(ref_lse(lp_gen[ABSTAIN_IDS]), abs=1e-9)  # first-token value
    calls = backend.model.forward_calls
    assert len(calls) == 2
    prompt_ids = backend.processor.calls[-1]
    for call, completion in zip(calls, ([10, EOT], [20, EOT])):
        assert call["ids"][-2:] == completion and call["ids"][-3] == GEN
        assert call["mask"] == [1] * len(call["ids"])
        assert call["extra"]["mm_token_type_ids"][-2:] == [0, 0]
        assert call["use_cache"] is False and call["keep"] == 3
    assert prompt_ids["order"] == ["image", "text"]
    # the first-token z differs from the full-sequence z for this table
    first = common.first_token_fields(table[GEN], backend.signal_ids)
    assert abs(first["z"] - rec["z"]) > 0.1


class FullLogitsModel(BigramModel):
    """A model that ignores ``logits_to_keep`` and returns logits for every position."""

    def __call__(self, input_ids=None, attention_mask=None, use_cache=None, logits_to_keep=0, **kwargs):
        return super().__call__(input_ids=input_ids, attention_mask=attention_mask, use_cache=use_cache,
                                logits_to_keep=0, **kwargs)


def test_completion_logprob_handles_models_that_ignore_logits_to_keep(workspace):
    backend = make_backend(model=FullLogitsModel(make_table({10: 2.0}, after={10: {EOT: 3.0}})))
    inputs = backend.prepare(backend.build_inputs("question", None))
    expected = ref_log_softmax(backend.model.table[GEN].numpy())[10] + ref_log_softmax(backend.model.table[10].numpy())[EOT]
    assert backend.completion_logprob(inputs, [10, EOT]) == pytest.approx(expected, abs=1e-6)
    bad = dict(inputs)
    bad["weird_per_token"] = torch.zeros_like(inputs["input_ids"])
    with pytest.raises(ValueError, match="weird_per_token"):
        backend.completion_logprob(bad, [10, EOT])
    with pytest.raises(ValueError):
        backend.completion_logprob(inputs, [])


def test_prepare_casts_floating_inputs_to_the_model_dtype(workspace):
    backend = make_backend(dtype="bfloat16")
    image, _ = common.load_image_file(workspace.images / nih_name(1))
    prepared = backend.prepare(backend.build_inputs("q", image))
    assert prepared["pixel_values"].dtype == torch.bfloat16
    assert prepared["input_ids"].dtype == torch.long and prepared["image_grid_thw"].dtype == torch.long


# ============================================================================= visual-token assertion
def test_visual_token_assertion_qwen(workspace):
    image, _ = common.load_image_file(workspace.images / nih_name(1))
    ok = make_backend()
    assert ok.count_visual_tokens(ok.build_inputs("q", image), has_image=True) == 1024
    assert ok.count_visual_tokens(ok.build_inputs("q", None), has_image=False) == 0
    small = make_backend(processor=FakeProcessor("qwen3vl", grid=(1, 32, 32)))
    with pytest.raises(VisualTokenAssertionError, match=r"256 visual tokens.*expected 1024"):
        small.count_visual_tokens(small.build_inputs("q", image), has_image=True)
    inconsistent = make_backend(processor=FakeProcessor("qwen3vl", extra_image_tokens=3))
    with pytest.raises(VisualTokenAssertionError, match="input_ids hold 1027"):
        inconsistent.count_visual_tokens(inconsistent.build_inputs("q", image), has_image=True)
    leaky = make_backend(processor=FakeProcessor("qwen3vl", text_only_image_tokens=2))
    with pytest.raises(VisualTokenAssertionError, match="text-only"):
        leaky.count_visual_tokens(leaky.build_inputs("q", None), has_image=False)


def test_visual_token_assertion_gemma(workspace):
    image, _ = common.load_image_file(workspace.images / nih_name(1))
    ok = make_backend("medgemma15")
    assert ok.count_visual_tokens(ok.build_inputs("q", image), has_image=True) == 256
    crops = make_backend("medgemma15", processor=FakeProcessor("gemma3", gemma_tokens=512))
    with pytest.raises(VisualTokenAssertionError, match="512"):
        crops.count_visual_tokens(crops.build_inputs("q", image), has_image=True)
    rec = run_vlm.process_task(ok, Task(item_id(1), nih_name(1), "image", "P2hi"), signal="full_sequence",
                               run_id="r", images_dir=workspace.images, image_cache={})
    assert rec["visual_tokens"] == 256 and rec["model_id"] == "google/medgemma-1.5-4b-it"
    assert all(c["extra"]["token_type_ids"][-2:] == [0, 0] for c in ok.model.forward_calls)


def test_token_assertion_stops_the_run_before_writing(workspace):
    backend = make_backend(processor=FakeProcessor("qwen3vl", grid=(1, 32, 32)))
    tasks = common.plan_tasks(common.read_pod_items(workspace.items), "items")
    with pytest.raises(VisualTokenAssertionError):
        run_vlm.run_tasks(backend, tasks, out_path=workspace.out, run_id="r1", signal="first_token",
                          images_dir=workspace.images, log=lambda s: None)
    assert not workspace.out.exists()


# ============================================================================= blind rows
def test_blind_rows(workspace):
    backend = make_backend()
    tasks = common.plan_tasks([], "blind")
    summary = run_vlm.run_tasks(backend, tasks, out_path=workspace.out, run_id="r1", signal="first_token",
                                log=lambda s: None)
    assert summary["n_written"] == 6
    recs = read_jsonl(workspace.out)
    grey = common.grey_png_bytes()
    text_rows = [r for r in recs if r["condition"] == "BLIND-text"]
    grey_rows = [r for r in recs if r["condition"] == "BLIND-grey"]
    assert [r["prompt_id"] for r in text_rows] == ["P0", "P1", "P2hi"] == [r["prompt_id"] for r in grey_rows]
    for r in text_rows:
        assert (r["item_id"], r["nih_filename"], r["image_sha256"], r["visual_tokens"]) == ("BLIND-TEXT", "", "", 0)
    for r in grey_rows:
        assert (r["item_id"], r["nih_filename"], r["visual_tokens"]) == ("BLIND-GREY", "", 1024)
        assert r["image_sha256"] == hashlib.sha256(grey).hexdigest()
    calls = backend.processor.calls
    assert [c["order"] for c in calls[:3]] == [["text"]] * 3
    assert all(c["order"] == ["image", "text"] for c in calls[3:])
    arr = np.asarray(calls[3]["images"][0])
    assert arr.shape == (1024, 1024, 3) and arr.min() == arr.max() == 128
    assert (workspace.out.parent / "BLIND-GREY_1024.png").read_bytes() == grey
    # blind rows and item rows share one file and resume independently
    items_tasks = common.plan_tasks(common.read_pod_items(workspace.items), "items")
    s2 = run_vlm.run_tasks(backend, items_tasks, out_path=workspace.out, run_id="r1", signal="first_token",
                           images_dir=workspace.images, log=lambda s: None)
    assert s2["n_written"] == len(items_tasks)
    assert len(read_jsonl(workspace.out)) == 6 + len(items_tasks)


# ============================================================================= resume
def test_resume_skips_done_keys_and_repairs_a_partial_line(workspace):
    backend = make_backend()
    tasks = common.plan_tasks(common.read_pod_items(workspace.items), "items")
    first = run_vlm.run_tasks(backend, tasks[:4], out_path=workspace.out, run_id="r1", signal="first_token",
                              images_dir=workspace.images, log=lambda s: None)
    assert first["n_written"] == 4
    with open(workspace.out, "ab") as fh:  # a crash in the middle of a write
        fh.write(b'{"run_id": "r1", "item_id": "trunc')
    with pytest.warns(RuntimeWarning, match="malformed"):
        second = run_vlm.run_tasks(backend, tasks, out_path=workspace.out, run_id="r1", signal="first_token",
                                   images_dir=workspace.images, log=lambda s: None)
    assert (second["n_done_before"], second["n_written"]) == (4, len(tasks) - 4)
    with pytest.warns(RuntimeWarning):
        recs = read_jsonl(workspace.out)
    keys = [(r["item_id"], r["prompt_id"], r["condition"]) for r in recs]
    assert len(keys) == len(set(keys)) == len(tasks)
    with pytest.warns(RuntimeWarning):
        third = run_vlm.run_tasks(backend, tasks, out_path=workspace.out, run_id="r1", signal="first_token",
                                  images_dir=workspace.images, log=lambda s: None)
    assert third["n_written"] == 0


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"run_id": "r2"}, "run_id"),
        ({"signal": "full_sequence"}, "signal"),
        ({"revision": "b" * 40}, "revision"),
        ({"model_id": "other/model"}, "model_id"),
        ({"transformers_version": "4.57.0"}, "transformers_version"),
        ({"prompt_sha256": "0" * 64}, "prompt_sha256"),
    ],
)
def test_resume_refuses_records_from_another_configuration(workspace, change, message):
    backend = make_backend()
    tasks = common.plan_tasks(common.read_pod_items(workspace.items), "items")
    run_vlm.run_tasks(backend, tasks[:2], out_path=workspace.out, run_id="r1", signal="first_token",
                      images_dir=workspace.images, log=lambda s: None)
    recs = read_jsonl(workspace.out)
    recs[1].update(change)
    workspace.out.write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    with pytest.raises(ResumeConflictError, match=message):
        run_vlm.run_tasks(backend, tasks, out_path=workspace.out, run_id="r1", signal="first_token",
                          images_dir=workspace.images, log=lambda s: None)


def test_check_resume_flags_duplicates_and_unknown_prompts():
    rec = {"item_id": "x", "prompt_id": "P0", "condition": "image", "prompt_sha256": prompt_sha256(prompt_text("P0"))}
    assert common.check_resume([rec], {}) == {("x", "P0", "image")}
    with pytest.raises(ResumeConflictError, match="duplicates"):
        common.check_resume([rec, dict(rec)], {})
    with pytest.raises(ResumeConflictError, match="prompt_sha256"):
        common.check_resume([{**rec, "prompt_id": "P9"}], {})
    with pytest.raises(ResumeConflictError, match="lacks"):
        common.check_resume([{"item_id": "x"}], {})


# ============================================================================= record validation
def _valid_record(**overrides):
    text = prompt_text("P0")
    rec = {
        "run_id": "r", "model_key": "qwen3vl2b", "model_id": QWEN_ID, "revision": "a" * 40,
        "transformers_version": "5.22.1", "torch_version": "2.15", "dtype": "bfloat16", "device": "cuda:0",
        "item_id": item_id(1), "nih_filename": nih_name(1), "image_sha256": "c" * 64, "condition": "image",
        "prompt_id": "P0", "prompt_sha256": prompt_sha256(text), "prompt_text": text, "visual_tokens": 1024,
        "seed": 1, "max_new_tokens": 16, "raw_output": "Yes", "n_generated_tokens": 2, "ended_with_eos": True,
        "parsed": "yes", "parser_version": "2", "logp_yes": -0.1, "logp_no": -3.0, "logp_abstain": float("nan"),
        "z": 2.9, "coverage": 0.95, "signal": "first_token", "yes_ids": [1], "no_ids": [2], "wall_s": 0.3,
        "timestamp_utc": "2026-11-02T10:00:00Z",
    }
    rec.update(overrides)
    return rec


def test_make_record_orders_coerces_and_validates():
    shuffled = dict(reversed(list(_valid_record(visual_tokens=np.int64(1024), wall_s=np.float32(0.5)).items())))
    rec = common.make_record(shuffled)
    assert list(rec) == list(S7_FIELDS) and type(rec["visual_tokens"]) is int and type(rec["wall_s"]) is float
    assert math.isnan(rec["logp_abstain"])
    with pytest.raises(ValueError, match="missing"):
        common.make_record({k: v for k, v in _valid_record().items() if k != "seed"})
    with pytest.raises(ValueError, match="unexpected"):
        common.make_record(_valid_record(arm="A"))
    with pytest.raises(ValueError, match="bool"):
        common.make_record(_valid_record(ended_with_eos="True"))


@pytest.mark.parametrize(
    "overrides",
    [
        {"condition": "image2"},
        {"parsed": "maybe"},
        {"signal": "other"},
        {"coverage": 1.5},
        {"z": float("nan")},
        {"prompt_text": "edited"},
        {"yes_ids": []},
        {"condition": "BLIND-text", "item_id": "BLIND-TEXT", "nih_filename": "", "image_sha256": "",
         "visual_tokens": 3},
        {"condition": "BLIND-grey", "item_id": "BLIND-GREY", "nih_filename": "", "image_sha256": ""},
        {"item_id": "1.2.840.113619"},
        {"visual_tokens": 0},
    ],
)
def test_validate_record_rejects(overrides):
    with pytest.raises(ValueError):
        common.validate_record(_valid_record(**overrides))


def test_validate_record_accepts_blind_rows():
    common.validate_record(_valid_record(condition="BLIND-text", item_id="BLIND-TEXT", nih_filename="",
                                         image_sha256="", visual_tokens=0))
    common.validate_record(_valid_record(condition="BLIND-grey", item_id="BLIND-GREY", nih_filename=""))


# ============================================================================= coverage decision and smoke report
@pytest.mark.parametrize(
    ("covs", "median", "decision"),
    [
        ([0.95] * 20, 0.95, "first_token"),
        ([0.90] * 10 + [0.99] * 10, 0.945, "first_token"),
        ([0.5] * 10 + [0.90] * 11, 0.90, "first_token"),
        ([0.8999] * 11 + [0.99] * 9, 0.8999, "full_sequence"),
        ([0.2, 0.95], 0.575, "full_sequence"),
    ],
)
def test_coverage_decision(covs, median, decision):
    got_median, got_decision = common.coverage_decision(covs)
    assert got_median == pytest.approx(median) and got_decision == decision


def test_coverage_decision_rejects_empty_and_nan():
    with pytest.raises(ValueError):
        common.coverage_decision([])
    with pytest.raises(ValueError):
        common.coverage_decision([0.9, float("nan")])


def test_smoke_report_merge_and_decision_reading(tmp_path):
    report = tmp_path / "smoke_report.json"
    common.update_smoke_report(report, "qwen3vl2b", {"visual_token_assertion": "pass", "decision": "first_token"})
    common.update_smoke_report(report, "medgemma15", {"visual_token_assertion": "pass", "decision": "full_sequence"})
    data = json.loads(report.read_text(encoding="utf-8"))
    assert set(data["models"]) == {"qwen3vl2b", "medgemma15"} and data["coverage_threshold"] == 0.9
    assert common.read_smoke_decision(report, "qwen3vl2b") == "first_token"
    assert common.resolve_signal("auto", report, "medgemma15") == ("full_sequence", "smoke_report")
    assert common.resolve_signal("first_token", None, "medgemma15") == ("first_token", "cli")
    with pytest.raises(SmokeReportError, match="no section"):
        common.read_smoke_decision(report, "chexone")
    with pytest.raises(SmokeReportError, match="not found"):
        common.read_smoke_decision(tmp_path / "none.json", "qwen3vl2b")
    common.update_smoke_report(report, "qwen3vl2b", {"visual_token_assertion": "fail", "error": "grid", "decision": None})
    with pytest.raises(SmokeReportError, match="did not pass"):
        common.read_smoke_decision(report, "qwen3vl2b")
    common.update_smoke_report(report, "qwen3vl2b", {"visual_token_assertion": "pass", "decision": "maybe"})
    with pytest.raises(SmokeReportError, match="no valid decision"):
        common.read_smoke_decision(report, "qwen3vl2b")
    with pytest.raises(ValueError):
        common.resolve_signal("sometimes", report, "qwen3vl2b")


def test_run_smoke_writes_report_with_the_registered_fields(workspace):
    report = workspace.root / "runs" / "r1" / "smoke_report.json"
    out = workspace.root / "runs" / "r1" / "smoke" / "qwen3vl2b.smoke.jsonl"
    out.parent.mkdir(parents=True)
    out.write_text("old\n", encoding="utf-8")
    backend = make_backend(table=make_table({10: 9.0, 20: 8.0}))
    lines: list[str] = []
    section = run_vlm.run_smoke(backend, common.read_pod_items(workspace.items), images_dir=workspace.images,
                                out_path=out, report_path=report, run_id="r1-smoke", log=lines.append)
    assert (out.parent / "qwen3vl2b.smoke.prev.jsonl").read_text(encoding="utf-8") == "old\n"
    recs = read_jsonl(out)
    assert [r["item_id"] for r in recs] == [item_id(3), item_id(4)] and {r["prompt_id"] for r in recs} == {"P0"}
    assert all(r["signal"] == "first_token" and r["run_id"] == "r1-smoke" for r in recs)
    for key in ("visual_tokens", "median_coverage", "decision", "s_per_item_mean", "peak_gpu_mem_gb", "n_items",
                "revision", "yes_ids", "no_ids", "skipped_forms", "visual_token_assertion"):
        assert key in section, key
    assert section["visual_tokens"] == 1024 and section["n_items"] == 2 and section["peak_gpu_mem_gb"] is None
    assert section["decision"] == "first_token" and section["median_coverage"] > 0.9
    assert "note" in section  # 2 smoke items instead of the registered 20
    assert json.loads(report.read_text(encoding="utf-8"))["models"]["qwen3vl2b"]["decision"] == "first_token"
    assert any("parsed=yes" in line for line in lines)  # smoke items are dev-pool items: outputs may be shown


def test_run_smoke_low_coverage_decides_full_sequence(workspace):
    table = make_table({10: 1.0, 20: 0.5, 40: 6.0}, after={40: {EOT: 9.0}})  # first token mostly "other"
    backend = make_backend(table=table)
    report = workspace.root / "smoke_report.json"
    section = run_vlm.run_smoke(backend, common.read_pod_items(workspace.items), images_dir=workspace.images,
                                out_path=workspace.root / "s.jsonl", report_path=report, run_id="s", log=lambda s: None)
    assert section["median_coverage"] < 0.9 and section["decision"] == "full_sequence"
    assert common.read_smoke_decision(report, "qwen3vl2b") == "full_sequence"


def test_run_smoke_records_a_token_failure(workspace):
    backend = make_backend(processor=FakeProcessor("qwen3vl", grid=(1, 16, 16)))
    report = workspace.root / "smoke_report.json"
    with pytest.raises(VisualTokenAssertionError):
        run_vlm.run_smoke(backend, common.read_pod_items(workspace.items), images_dir=workspace.images,
                          out_path=workspace.root / "s.jsonl", report_path=report, run_id="s", log=lambda s: None)
    sec = json.loads(report.read_text(encoding="utf-8"))["models"]["qwen3vl2b"]
    assert sec["visual_token_assertion"] == "fail" and "expected 1024" in sec["error"]


def test_progress_lines_never_show_outputs_outside_smoke(workspace):
    backend = make_backend()
    lines: list[str] = []
    tasks = common.plan_tasks(common.read_pod_items(workspace.items), "items")
    run_vlm.run_tasks(backend, tasks, out_path=workspace.out, run_id="r1", signal="first_token",
                      images_dir=workspace.images, log=lines.append)
    joined = "\n".join(lines)
    assert "parsed" not in joined and "coverage=" not in joined and "z=" not in joined and "Yes" not in joined
    assert "s/record" in joined


# ============================================================================= run_vlm CLI
def _cli_args(ws, *extra, out=None, run_id="r1", model="qwen3vl2b"):
    return ["--model-key", model, "--items", str(ws.items), "--images-dir", str(ws.images),
            "--out", str(out or ws.out), "--run-id", run_id, *extra]


def test_cli_smoke_then_auto_signal_then_blind(workspace):
    factory = fake_factory(**{"*": {"table": make_table({10: 1.0, 20: 0.5, 40: 6.0}, after={40: {EOT: 9.0}})}})
    run_dir = workspace.out.parent
    smoke_out = run_dir / "smoke" / "qwen3vl2b.smoke.jsonl"
    report = run_dir / "smoke_report.json"
    assert run_vlm.main(_cli_args(workspace, "--smoke-only", "--smoke-report", str(report), out=smoke_out,
                                  run_id="r1-smoke"), backend_factory=factory) == 0
    assert json.loads(report.read_text(encoding="utf-8"))["models"]["qwen3vl2b"]["decision"] == "full_sequence"
    assert run_vlm.main(_cli_args(workspace), backend_factory=factory) == 0  # --signal auto by default
    recs = read_jsonl(workspace.out)
    assert len(recs) == 9 and {r["signal"] for r in recs} == {"full_sequence"}
    assert run_vlm.main(["--model-key", "qwen3vl2b", "--blind", "--out", str(workspace.out), "--run-id", "r1"],
                        backend_factory=factory) == 0
    recs = read_jsonl(workspace.out)
    assert len(recs) == 15 and {r["condition"] for r in recs} == {"image", "BLIND-text", "BLIND-grey"}
    meta = json.loads(common.meta_path(workspace.out).read_text(encoding="utf-8"))
    assert [s["mode"] for s in meta["sessions"]] == ["items", "blind"]
    assert meta["sessions"][0]["signal_source"] == "smoke_report"
    backend_meta = meta["sessions"][0]["backend"]
    assert backend_meta["generation_config"]["do_sample"] is False
    assert backend_meta["generation_config"]["output_logits"] is True
    assert backend_meta["full_sequence_completions"] == {"Yes": [10, EOT], "No": [20, EOT]}
    assert all(kwargs["dtype"] == "bfloat16" for _, kwargs in factory.calls)  # registered dtype by default


def test_cli_nothing_to_do_does_not_load_the_model(workspace):
    factory = fake_factory()
    args = _cli_args(workspace, "--signal", "first_token")
    assert run_vlm.main(args, backend_factory=factory) == 0
    n_loads = len(factory.calls)

    def refuse(*a, **k):
        raise AssertionError("the model must not be loaded")

    assert run_vlm.main(args, backend_factory=refuse) == 0
    assert len(factory.calls) == n_loads


def test_cli_exit_codes(workspace, capsys):
    factory = fake_factory()
    # auto signal without a smoke report
    assert run_vlm.main(_cli_args(workspace), backend_factory=factory) == 2
    assert "smoke_test.py" in capsys.readouterr().err
    # token assertion
    bad = fake_factory(**{"*": {"processor": FakeProcessor("qwen3vl", grid=(1, 8, 8))}})
    assert run_vlm.main(_cli_args(workspace, "--signal", "first_token"), backend_factory=bad) == 3
    assert "VISUAL-TOKEN ASSERTION FAILED" in capsys.readouterr().err
    assert not workspace.out.exists()
    # resume conflict (same file, another run id)
    assert run_vlm.main(_cli_args(workspace, "--signal", "first_token", "--limit", "1"), backend_factory=factory) == 0
    assert run_vlm.main(_cli_args(workspace, "--signal", "first_token", run_id="r2"), backend_factory=factory) == 4
    assert "RESUME CONFLICT" in capsys.readouterr().err
    # forbidden column in the item file
    bad_items = write_items(workspace.root / "bad.csv", [[item_id(1), nih_name(1), "full", 0, "B"]],
                            header=POD_ITEM_COLUMNS + ("arm",))
    args = _cli_args(workspace, "--signal", "first_token", out=workspace.root / "o.jsonl")
    args[args.index("--items") + 1] = str(bad_items)
    assert run_vlm.main(args, backend_factory=factory) == 2
    # missing image
    (workspace.images / nih_name(5)).unlink()
    assert run_vlm.main(_cli_args(workspace, "--signal", "first_token", out=workspace.root / "o2.jsonl"),
                        backend_factory=factory) == 2
    assert "missing" in capsys.readouterr().err
    # items mode without --items
    assert run_vlm.main(["--model-key", "qwen3vl2b", "--out", str(workspace.root / "o3.jsonl"), "--run-id", "x"],
                        backend_factory=factory) == 2
    with pytest.raises(SystemExit):
        run_vlm.main(_cli_args(workspace, "--blind", "--smoke-only"), backend_factory=factory)


# ============================================================================= smoke_test.py
def test_smoke_test_runs_every_model_and_fails_on_a_token_assertion(workspace):
    factory = fake_factory(medgemma15={"processor": FakeProcessor("gemma3", gemma_tokens=300)})
    run_dir = workspace.root / "runs" / "r9"
    code = smoke_test.main(["--items", str(workspace.items), "--images-dir", str(workspace.images),
                            "--run-dir", str(run_dir), "--run-id", "r9"], backend_factory=factory)
    assert code == 3
    assert [k for k, _ in factory.calls] == ["qwen3vl2b", "medgemma15"]
    report = json.loads((run_dir / "smoke_report.json").read_text(encoding="utf-8"))["models"]
    assert report["qwen3vl2b"]["visual_token_assertion"] == "pass"
    assert report["medgemma15"]["visual_token_assertion"] == "fail"
    assert len(read_jsonl(run_dir / "smoke" / "qwen3vl2b.smoke.jsonl")) == 2
    assert {r["run_id"] for r in read_jsonl(run_dir / "smoke" / "qwen3vl2b.smoke.jsonl")} == {"r9-smoke"}


def test_smoke_test_passes_and_reports_load_errors(workspace):
    run_dir = workspace.root / "runs" / "ok"
    args = ["--items", str(workspace.items), "--images-dir", str(workspace.images), "--run-dir", str(run_dir),
            "--run-id", "ok"]
    assert smoke_test.main(args, backend_factory=fake_factory()) == 0
    sections = json.loads((run_dir / "smoke_report.json").read_text(encoding="utf-8"))["models"]
    assert sections["qwen3vl2b"]["visual_tokens"] == 1024 and sections["medgemma15"]["visual_tokens"] == 256

    def broken(model_key, **kwargs):
        if model_key == "medgemma15":
            raise OSError("gated repo: terms not accepted")
        return make_backend(model_key)

    run_dir2 = workspace.root / "runs" / "err"
    args2 = args[:]
    args2[args2.index("--run-dir") + 1] = str(run_dir2)
    assert smoke_test.main(args2, backend_factory=broken) == 1
    sec = json.loads((run_dir2 / "smoke_report.json").read_text(encoding="utf-8"))["models"]["medgemma15"]
    assert sec["visual_token_assertion"] == "not_run" and "gated" in sec["error"]
    no_smoke = write_items(workspace.root / "ns.csv", [[item_id(1), nih_name(1), "full", 0]])
    args3 = args[:]
    args3[args3.index("--items") + 1] = str(no_smoke)
    assert smoke_test.main(args3, backend_factory=fake_factory()) == 2


# ============================================================================= MedSigLIP
# The MedSigLIP model card recommends, for reproducing its evaluations (to match the Big Vision
# library), ``Image.fromarray(tf.image.resize(images=image, size=[448, 448], method="bilinear",
# antialias=False).numpy().astype(np.uint8))``. With antialias=False that call is TensorFlow's
# ResizeBilinear kernel with half_pixel_centers=True, which works in float32 throughout
# (tensorflow/core/kernels/image/resize_bilinear_op.cc): scale = float(in) / float(out);
# in = (float(i) + 0.5f) * scale - 0.5f; lower = max(floor(in), 0); upper = min(ceil(in), in - 1);
# lerp = in - floor(in); top = tl + (tr - tl) * x_lerp; bottom = bl + (br - bl) * x_lerp;
# out = top + (bottom - top) * y_lerp. The float32 source positions differ from exact arithmetic by
# up to about 1e-4 pixel, so the output differs from exact bilinear math by up to a few hundredths
# of a grey level, and the uint8 truncation then moves about 1% of the pixels of a 1024 -> 448
# resize by one level. A faithful replica must therefore match the float32 kernel bit for bit,
# not a float64 formula.
def tf_kernel_reference(arr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Line-by-line float32 transcription of TensorFlow's CPU ResizeBilinear kernel (half-pixel centres)."""
    f32 = np.float32

    def weights(in_size: int, out_size: int):
        scale = f32(in_size) / f32(out_size)                    # CalculateResizeScale
        lower, upper, lerp = [], [], []
        for i in range(out_size):                                # compute_interpolation_weights
            pos = (f32(i) + f32(0.5)) * scale - f32(0.5)         # HalfPixelScaler
            pos_floor = np.floor(pos)
            lower.append(max(int(pos_floor), 0))
            upper.append(min(int(np.ceil(pos)), in_size - 1))
            lerp.append(f32(pos - pos_floor))
        return np.array(lower), np.array(upper), np.array(lerp, dtype=f32)

    a = np.asarray(arr).astype(f32)
    y_lo, y_hi, y_lerp = weights(a.shape[0], out_h)
    x_lo, x_hi, x_lerp = weights(a.shape[1], out_w)
    xl = x_lerp[:, None]
    out = np.empty((out_h, out_w, a.shape[2]), dtype=f32)
    for y in range(out_h):                                       # ResizeLineChannels / compute_lerp
        top_left, top_right = a[y_lo[y], x_lo], a[y_lo[y], x_hi]
        bottom_left, bottom_right = a[y_hi[y], x_lo], a[y_hi[y], x_hi]
        top = top_left + (top_right - top_left) * xl
        bottom = bottom_left + (bottom_right - bottom_left) * xl
        out[y] = top + (bottom - top) * y_lerp[y]
    return out


def exact_bilinear_reference(arr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Half-pixel bilinear resize in float64 (exact arithmetic up to float64 rounding)."""
    in_h, in_w = arr.shape[:2]

    def axis(n_out, n_in):
        pos = (np.arange(n_out) + 0.5) * (n_in / n_out) - 0.5
        lo = np.floor(pos)
        frac = pos - lo
        lower = np.clip(lo, 0, n_in - 1).astype(int)
        upper = np.clip(np.ceil(pos), 0, n_in - 1).astype(int)
        return lower, upper, frac

    y0, y1, fy = axis(out_h, in_h)
    x0, x1, fx = axis(out_w, in_w)
    a = arr.astype(np.float64)
    top = a[y0][:, x0] * (1 - fx)[None, :, None] + a[y0][:, x1] * fx[None, :, None]
    bottom = a[y1][:, x0] * (1 - fx)[None, :, None] + a[y1][:, x1] * fx[None, :, None]
    return top * (1 - fy)[:, None, None] + bottom * fy[:, None, None]


@pytest.mark.parametrize(("h", "w"), [(1024, 1024), (300, 200), (2500, 2048)])
def test_card_resize_matches_the_tensorflow_float32_kernel_bit_for_bit(h, w):
    rng = np.random.default_rng(h + w)
    arr = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    kernel = tf_kernel_reference(arr, 448, 448)
    resized = run_medsiglip.tf_bilinear_resize(arr, 448, 448)
    assert resized.dtype == np.float32 and np.array_equal(resized, kernel)
    got = np.asarray(run_medsiglip.card_resize(Image.fromarray(arr)))
    assert got.shape == (448, 448, 3) and got.dtype == np.uint8
    assert np.array_equal(got, kernel.astype(np.uint8))  # the card's .numpy().astype(np.uint8)


@pytest.mark.parametrize(("h", "w"), [(1024, 1024), (300, 200)])
def test_card_resize_is_bilinear_up_to_float32_rounding(h, w):
    """Against exact bilinear math: a few hundredths of a grey level before truncation, one level after."""
    rng = np.random.default_rng(h + w)
    arr = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    exact = exact_bilinear_reference(arr, 448, 448)
    assert np.abs(run_medsiglip.tf_bilinear_resize(arr, 448, 448) - exact).max() < 0.05
    got = np.asarray(run_medsiglip.card_resize(Image.fromarray(arr))).astype(int)
    diff = np.abs(got - exact.astype(np.uint8).astype(int))
    assert diff.max() <= 1 and (diff > 0).mean() < 0.03


def test_card_resize_hand_values_flat_images_and_identity():
    # 4 x 4 -> 2 x 2 with half-pixel centres samples at 0.5 and 2.5: each output is its 2 x 2 block mean
    arr = (np.arange(16, dtype=np.uint8).reshape(4, 4) * 4)[:, :, None].repeat(3, axis=2)
    out = run_medsiglip.tf_bilinear_resize(arr, 2, 2)
    assert out[..., 0].tolist() == [[10.0, 18.0], [42.0, 50.0]]
    flat = Image.fromarray(np.full((1024, 1024), 77, np.uint8)).convert("RGB")
    assert np.unique(np.asarray(run_medsiglip.card_resize(flat))).tolist() == [77]
    rng = np.random.default_rng(7)
    same = rng.integers(0, 256, (448, 448, 3), dtype=np.uint8)
    assert np.array_equal(np.asarray(run_medsiglip.card_resize(Image.fromarray(same))), same)


class FakeSiglipProcessor:
    def __init__(self):
        self.calls = []

    def __call__(self, text=None, images=None, padding=None, return_tensors=None):
        self.calls.append({"text": list(text), "sizes": [im.size for im in images], "padding": padding})
        assert padding == "max_length" and return_tensors == "pt"
        means = [float(np.asarray(im, dtype=np.float64).mean()) / 255.0 for im in images]
        return {"input_ids": torch.zeros((len(text), 64), dtype=torch.long),
                "pixel_values": torch.tensor(means, dtype=torch.float32).reshape(-1, 1, 1, 1).expand(-1, 3, 2, 2)}


class FakeSiglipModel:
    def __call__(self, input_ids=None, pixel_values=None, **kwargs):
        m = pixel_values.float().mean(dim=(1, 2, 3))
        return SimpleNamespace(logits_per_image=torch.stack([8.0 * m - 4.0, 4.0 - 8.0 * m], dim=1))


def make_scorer(resize="card"):
    return run_medsiglip.MedSiglipScorer(FakeSiglipModel(), FakeSiglipProcessor(), model_id="google/medsiglip-448",
                                         revision="d" * 40, device="cpu", dtype_name="float32", resize=resize)


def test_medsiglip_scores_softmax_order_and_batching(workspace):
    items = common.read_pod_items(workspace.items)
    a = run_medsiglip.score_items(make_scorer(), items, workspace.images, batch_size=2, log=lambda s: None)
    b = run_medsiglip.score_items(make_scorer(), items, workspace.images, batch_size=5, log=lambda s: None)
    assert [r["item_id"] for r in a] == [it.item_id for it in items]
    for ra, rb in zip(a, b):
        assert set(ra) == set(common.S9_COLUMNS)
        assert ra["s"] == pytest.approx(rb["s"], abs=1e-12)
        e = np.exp([ra["logit_with"], ra["logit_without"]])
        assert ra["s"] == pytest.approx(e[0] / e.sum(), abs=1e-12)
        assert (ra["model_id"], ra["revision"]) == ("google/medsiglip-448", "d" * 40)
    scorer = make_scorer()
    run_medsiglip.score_items(scorer, items[:1], workspace.images, log=lambda s: None)
    call = scorer.processor.calls[0]
    assert call["text"] == list(constants.EXPERT["texts"]) and call["sizes"] == [(448, 448)]
    raw = make_scorer("processor")
    run_medsiglip.score_items(raw, items[:1], workspace.images, log=lambda s: None)
    assert raw.processor.calls[0]["sizes"] == [(64, 64)]
    with pytest.raises(ValueError):
        make_scorer("bicubic")


def test_medsiglip_cli_writes_s9_and_meta(workspace):
    out = workspace.root / "runs" / "r1" / "detector_scores.csv"
    assert run_medsiglip.main(["--items", str(workspace.items), "--images-dir", str(workspace.images),
                               "--out", str(out), "--batch-size", "2"],
                              scorer_factory=lambda **kw: make_scorer(kw["resize"])) == 0
    data = out.read_bytes()
    assert b"\r\n" not in data
    rows = list(csv.DictReader(io.StringIO(data.decode("utf-8"))))
    assert list(rows[0]) == list(common.S9_COLUMNS) and len(rows) == 5
    assert all(0.0 < float(r["s"]) < 1.0 for r in rows)
    meta = json.loads(common.meta_path(out).read_text(encoding="utf-8"))
    assert meta["csv_sha256"] == hashlib.sha256(data).hexdigest() and meta["resize"] == "card"
    assert meta["texts"] == list(constants.EXPERT["texts"]) and meta["padding"] == "max_length"
    bad = write_items(workspace.root / "bad.csv", [[item_id(1), nih_name(1), "full", 0, "x"]],
                      header=POD_ITEM_COLUMNS + ("uid",))
    assert run_medsiglip.main(["--items", str(bad), "--images-dir", str(workspace.images), "--out", str(out)],
                              scorer_factory=lambda **kw: make_scorer()) == 2
    (workspace.images / nih_name(1)).unlink()
    assert run_medsiglip.main(["--items", str(workspace.items), "--images-dir", str(workspace.images),
                               "--out", str(out)], scorer_factory=lambda **kw: make_scorer()) == 2


def test_write_csv_rows_rejects_mismatched_rows(tmp_path):
    with pytest.raises(ValueError):
        common.write_csv_rows(tmp_path / "x.csv", ["a", "b"], [{"a": 1}])
    digest = common.write_csv_rows(tmp_path / "y.csv", ["a", "b"], [{"a": 0.1, "b": "t"}])
    assert (tmp_path / "y.csv").read_text(encoding="utf-8") == "a,b\n0.1,t\n"
    assert digest == hashlib.sha256(b"a,b\n0.1,t\n").hexdigest()


# ============================================================================= real Qwen3-VL processor files
def _load_qwen_processor():
    try:
        return transformers.AutoProcessor.from_pretrained(QWEN_ID, local_files_only=True)
    except Exception as exc:  # noqa: BLE001 - not cached on this machine
        pytest.skip(f"Qwen3-VL processor files are not in the local Hugging Face cache ({type(exc).__name__})")


@pytest.fixture(scope="module")
def qwen_available():
    _load_qwen_processor()
    return True


def test_real_qwen_single_token_yes_no_ids(qwen_available):
    proc = _load_qwen_processor()
    ids = common.signal_ids_from_encoder(lambda s: proc.tokenizer.encode(s, add_special_tokens=False))
    assert len(ids.yes_ids) == len(constants.YES_FORMS) and len(ids.no_ids) == len(constants.NO_FORMS)
    assert ids.skipped["yes"] == [] and ids.skipped["no"] == []
    assert [proc.tokenizer.decode([i]) for i in ids.yes_ids] == list(constants.YES_FORMS)
    assert [proc.tokenizer.decode([i]) for i in ids.no_ids] == list(constants.NO_FORMS)
    assert ids.yes_ids[0] == 9454 and ids.no_ids[0] == 2753  # "Yes", "No"
    assert proc.tokenizer.convert_tokens_to_ids("<|im_end|>") == 151645


def test_real_qwen_chat_template_has_no_system_prompt(qwen_available):
    proc = _load_qwen_processor()
    text = prompt_text("P0")
    content = [{"type": "image"}, {"type": "text", "text": text}]
    rendered = proc.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True,
                                        tokenize=False)
    assert rendered == ("<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>" + text
                        + "<|im_end|>\n<|im_start|>assistant\n")
    assert "system" not in rendered
    blind = proc.apply_chat_template([{"role": "user", "content": [{"type": "text", "text": text}]}],
                                     add_generation_prompt=True, tokenize=False)
    assert blind == "<|im_start|>user\n" + text + "<|im_end|>\n<|im_start|>assistant\n"


def test_real_qwen_image_grid_with_min_equal_max_pixels(qwen_available):
    side_image = Image.fromarray(np.full((1024, 1024), 77, np.uint8)).convert("RGB")
    small_image = Image.fromarray(np.full((512, 512), 77, np.uint8)).convert("RGB")
    default = _load_qwen_processor()
    assert default.image_processor(images=[small_image], return_tensors="pt")["image_grid_thw"].tolist() == [[1, 32, 32]]
    proc = _load_qwen_processor()
    settings = run_vlm.configure_processor(proc, constants.MODELS["qwen3vl2b"])
    assert settings["min_pixels"] == settings["max_pixels"] == 1048576 and settings["patch_size"] == 16
    for image in (side_image, small_image):  # min = max pixels: every input becomes 1024 x 1024
        msgs = [{"role": "user", "content": [{"type": "image", "image": image},
                                             {"type": "text", "text": prompt_text("P0")}]}]
        inputs = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
                                          return_tensors="pt")
        assert inputs["image_grid_thw"].tolist() == [[1, 64, 64]]
        assert int((inputs["input_ids"] == 151655).sum()) == 1024


# ----------------------------------------------------------------------------- tiny random Qwen3-VL model
def _tiny_qwen_model():
    cfg = transformers.Qwen3VLConfig(
        text_config=dict(vocab_size=151936, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=4096,
                         rope_parameters={"rope_type": "default", "rope_theta": 5000000.0,
                                          "mrope_section": [2, 3, 3], "mrope_interleaved": True}),
        vision_config=dict(depth=2, hidden_size=32, intermediate_size=64, num_heads=2, patch_size=16,
                           spatial_merge_size=2, temporal_patch_size=2, out_hidden_size=64,
                           num_position_embeddings=2304, deepstack_visual_indexes=[0]),
        image_token_id=151655, video_token_id=151656, vision_start_token_id=151652, vision_end_token_id=151653,
        tie_word_embeddings=True,
    )
    torch.manual_seed(0)
    model = transformers.Qwen3VLForConditionalGeneration(cfg).eval()
    model.generation_config = transformers.GenerationConfig(  # as in the checkpoint's generation_config.json
        bos_token_id=151643, pad_token_id=151643, do_sample=True, eos_token_id=[151645, 151643],
        top_p=0.8, top_k=20, temperature=0.7, repetition_penalty=1.0,
    )
    return model


@pytest.fixture
def tiny_backend(qwen_available):
    model = _tiny_qwen_model()
    return run_vlm.VLMBackend(model, _load_qwen_processor(), "qwen3vl2b", revision="0" * 40, dtype_name="float32",
                              device="cpu", **common.library_versions())


class ForceTokens:
    """Forward hook on lm_head: +50 on ``first`` at the first call, then on ``then`` (forces the greedy path)."""

    def __init__(self, first: int, then: int):
        self.first, self.then, self.n = first, then, 0

    def __call__(self, module, args, output):
        out = output.clone()
        out[:, -1, self.first if self.n == 0 else self.then] += 50.0
        self.n += 1
        return out


def test_tiny_qwen_real_generate_records(tiny_backend, tmp_path):
    images = tmp_path / "images"
    write_png(images / nih_name(1), seed=11, side=1024)
    cache: dict = {}
    hook = tiny_backend.model.lm_head.register_forward_hook(ForceTokens(9454, 151645))  # "Yes", then <|im_end|>
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            rec = run_vlm.process_task(tiny_backend, Task(item_id(1), nih_name(1), "image", "P0"),
                                       signal="first_token", run_id="tiny", images_dir=images, image_cache=cache)
    finally:
        hook.remove()
    assert rec["visual_tokens"] == 1024 and rec["raw_output"] == "Yes" and rec["parsed"] == "yes"
    assert (rec["n_generated_tokens"], rec["ended_with_eos"]) == (2, True)
    assert rec["coverage"] > 0.99 and rec["z"] > 10
    assert rec["yes_ids"][0] == 9454 and rec["no_ids"][0] == 2753
    # truncation with the real generate loop: "Yes" at every step, never the end of turn
    hook = tiny_backend.model.lm_head.register_forward_hook(ForceTokens(9454, 9454))
    try:
        rec2 = run_vlm.process_task(tiny_backend, Task(BLIND_TEXT := "BLIND-TEXT", "", "BLIND-text", "P1"),
                                    signal="first_token", run_id="tiny", images_dir=None, image_cache=cache)
    finally:
        hook.remove()
    assert BLIND_TEXT == rec2["item_id"] and rec2["visual_tokens"] == 0 and rec2["image_sha256"] == ""
    assert (rec2["n_generated_tokens"], rec2["ended_with_eos"]) == (16, False) and common.record_is_truncated(rec2)
    assert rec2["raw_output"] == "Yes" * 16
    rec3 = run_vlm.process_task(tiny_backend, Task("BLIND-GREY", "", "BLIND-grey", "P2hi"), signal="full_sequence",
                                run_id="tiny", images_dir=None, image_cache=cache)
    assert rec3["visual_tokens"] == 1024 and rec3["signal"] == "full_sequence"


def test_tiny_qwen_signals_match_independent_forward_passes(tiny_backend, tmp_path):
    images = tmp_path / "images"
    write_png(images / nih_name(2), seed=12, side=1024)
    image, _ = common.load_image_file(images / nih_name(2))
    text = prompt_text("P0")
    inputs = tiny_backend.prepare(tiny_backend.build_inputs(text, image))
    assert tiny_backend.model.generation_config.temperature is None
    gen = tiny_backend.generate(inputs)
    # raw first-step logits equal a plain forward pass over the prompt
    with torch.inference_mode():
        last = tiny_backend.model(**inputs, use_cache=False).logits[0, -1].float().numpy()
    np.testing.assert_allclose(gen.first_logits, last, atol=1e-4)
    assert gen.new_ids[0] == int(np.argmax(last))  # greedy on the raw logits (no processor changed the choice)
    first = common.first_token_fields(gen.first_logits, tiny_backend.signal_ids)
    lp = ref_log_softmax(last)
    assert first["logp_yes"] == pytest.approx(ref_lse(lp[list(tiny_backend.signal_ids.yes_ids)]), abs=1e-4)
    # full sequence: log P("Yes") at the prompt end + log P(<|im_end|> | prompt + "Yes")
    full = tiny_backend.score_full_sequence(inputs)
    ext = dict(inputs)
    ext["input_ids"] = torch.cat([inputs["input_ids"], torch.tensor([[9454]])], dim=1)
    ext["attention_mask"] = torch.cat([inputs["attention_mask"], torch.ones(1, 1, dtype=torch.long)], dim=1)
    ext["mm_token_type_ids"] = torch.cat([inputs["mm_token_type_ids"], torch.zeros(1, 1, dtype=torch.long)], dim=1)
    with torch.inference_mode():
        after_yes = tiny_backend.model(**ext, use_cache=False).logits[0, -1].float().numpy()
    expected_yes = lp[9454] + ref_log_softmax(after_yes)[151645]
    assert full["logp_yes"] == pytest.approx(expected_yes, abs=1e-3)
    assert full["coverage"] == pytest.approx(math.exp(full["logp_yes"]) + math.exp(full["logp_no"]))


# ----------------------------------------------------------------------------- opt-in: real weights on CPU
@pytest.mark.cpu_smoke
def test_real_qwen3vl_weights_on_cpu(request, tmp_path):
    if "cpu_smoke" not in (request.config.getoption("markexpr") or ""):
        pytest.skip("opt-in: run with -m cpu_smoke (loads the 2B weights on CPU)")
    from huggingface_hub import try_to_load_from_cache

    if not isinstance(try_to_load_from_cache(QWEN_ID, "model.safetensors"), str):
        pytest.skip("Qwen3-VL-2B weights are not cached")
    backend = run_vlm.load_backend("qwen3vl2b", device="cpu", dtype="bfloat16", local_files_only=True)
    assert re.fullmatch(r"[0-9a-f]{40}", backend.revision)
    images = tmp_path / "images"
    write_png(images / nih_name(1), seed=5, side=1024)
    items = write_items(tmp_path / "pod_items.csv", [[item_id(1), nih_name(1), "p0", 1]])
    section = run_vlm.run_smoke(backend, common.read_pod_items(items), images_dir=images,
                                out_path=tmp_path / "smoke.jsonl", report_path=tmp_path / "smoke_report.json",
                                run_id="cpu-smoke")
    assert section["visual_tokens"] == 1024
    rec = run_vlm.process_task(backend, Task("BLIND-TEXT", "", "BLIND-text", "P0"), signal="full_sequence",
                               run_id="cpu-smoke", images_dir=None, image_cache={})
    assert math.isfinite(rec["z"]) and 0 < rec["coverage"] <= 1


# ============================================================================= shell scripts
def _find_bash() -> str | None:
    candidates = [shutil.which("bash"), r"C:\Program Files\Git\bin\bash.exe", r"C:\Program Files\Git\usr\bin\bash.exe",
                  "/bin/bash", "/usr/bin/bash"]
    for c in candidates:
        if not c or not Path(c).exists() or "system32" in c.lower():
            continue
        try:
            r = subprocess.run([c, "-c", "echo ok"], capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.SubprocessError):
            continue
        if r.returncode == 0 and r.stdout.strip() == "ok":
            return c
    return None


BASH = _find_bash()
needs_bash = pytest.mark.skipif(BASH is None, reason="no usable bash found")


def _posix(p: Path) -> str:
    return p.as_posix()


@needs_bash
@pytest.mark.parametrize("script", ["queue.sh", "setup_pod.sh"])
def test_shell_scripts_parse(script):
    r = subprocess.run([BASH, "-n", _posix(REPO / "pod" / script)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr


def _run_queue(base: Path, run_id="q1", fail_on="", items=True, extra_env=None):
    """Run pod/queue.sh with a stub interpreter; return (result, argv of each call, run directory).

    The stub writes one line per call with every argument followed by a tab, so an argument that
    contains spaces (this repository's own path, or ``base`` when the test puts spaces in it) stays
    one argument; splitting a joined command line on whitespace would cut such paths apart.
    """
    base.mkdir(parents=True, exist_ok=True)
    stub = base / "stub_python.sh"
    stub.write_text('#!/usr/bin/env bash\n{ printf "%s\\t" "$@"; printf "\\n"; } >> "$STUB_LOG"\n'
                    'if [[ -n "${STUB_FAIL_ON:-}" && "$*" == *"$STUB_FAIL_ON"* ]]; then exit 3; fi\nexit 0\n',
                    encoding="utf-8", newline="\n")
    stub.chmod(0o755)
    item_file = base / "pod_items.csv"
    if items:
        write_items(item_file, [[item_id(1), nih_name(1), "p0", 1]])
    (base / "images").mkdir(exist_ok=True)
    env = dict(os.environ)
    env.update({"PYTHON": _posix(stub), "ITEMS": _posix(item_file), "IMAGES_DIR": _posix(base / "images"),
                "RUNS_DIR": _posix(base / "runs"), "STUB_LOG": _posix(base / "calls.log"),
                "STUB_FAIL_ON": fail_on, "HF_HOME": _posix(base / "hf")})
    env.update(extra_env or {})
    args = [BASH, _posix(REPO / "pod" / "queue.sh")] + ([run_id] if run_id is not None else [])
    r = subprocess.run(args, capture_output=True, text=True, timeout=120, env=env)
    log = base / "calls.log"
    lines = log.read_text(encoding="utf-8").splitlines() if log.exists() else []
    calls = [line.split("\t")[:-1] for line in lines]  # every argument ends with a tab
    return r, calls, base / "runs" / (run_id or "")


def _value(argv: list[str], flag: str) -> str:
    """The argument that follows ``flag`` in one recorded call."""
    return argv[argv.index(flag) + 1]


@needs_bash
def test_queue_runs_every_step_in_order_and_writes_done(tmp_path):
    base = tmp_path / "pod work dir"  # spaces in every path the queue receives, as in this repository's path
    r, calls, run_dir = _run_queue(base)
    assert r.returncode == 0, r.stdout + r.stderr
    scripts = [Path(argv[0]).name for argv in calls]
    assert scripts == ["smoke_test.py", "run_vlm.py", "run_vlm.py", "run_vlm.py", "run_vlm.py", "run_medsiglip.py"]
    for argv in calls:  # the script path (under the repository) arrives as one argument
        assert argv[0].endswith(f"{REPO.name}/pod/{Path(argv[0]).name}"), argv[0]
    rd = _posix(run_dir)
    smoke = calls[0]
    assert (_value(smoke, "--items"), _value(smoke, "--images-dir")) == (_posix(base / "pod_items.csv"),
                                                                         _posix(base / "images"))
    assert (_value(smoke, "--run-dir"), _value(smoke, "--run-id")) == (rd, "q1")
    assert smoke[smoke.index("--models") + 1:] == ["qwen3vl2b", "medgemma15"]
    for argv, model, blind in [(calls[1], "qwen3vl2b", False), (calls[2], "medgemma15", False),
                               (calls[3], "qwen3vl2b", True), (calls[4], "medgemma15", True)]:
        assert _value(argv, "--model-key") == model and ("--blind" in argv) is blind
        assert ("--items" in argv) is (not blind) and ("--images-dir" in argv) is (not blind)
        if not blind:
            assert _value(argv, "--items") == _posix(base / "pod_items.csv")
        assert _value(argv, "--out") == f"{rd}/{model}.jsonl"
        assert (_value(argv, "--run-id"), _value(argv, "--signal")) == ("q1", "auto")
        assert _value(argv, "--smoke-report") == f"{rd}/smoke_report.json"
    assert _value(calls[5], "--out") == f"{rd}/detector_scores.csv"
    assert _value(calls[5], "--images-dir") == _posix(base / "images")
    done = (run_dir / "DONE").read_text(encoding="utf-8")
    assert "status: ok" in done and "failed_steps: none" in done and not (run_dir / "FAILED").exists()
    queue_log = (run_dir / "logs" / "queue.log").read_text(encoding="utf-8")
    assert queue_log.count("START") == 6 and "DONE status=ok" in queue_log
    assert (run_dir / "logs" / "smoke.log").exists()


@needs_bash
def test_queue_stops_after_a_failed_smoke(tmp_path):
    r, calls, run_dir = _run_queue(tmp_path, fail_on="smoke_test.py")
    assert r.returncode == 1
    assert len(calls) == 1 and Path(calls[0][0]).name == "smoke_test.py"
    assert "status: smoke_failed" in (run_dir / "DONE").read_text(encoding="utf-8")
    assert (run_dir / "FAILED").exists()


@needs_bash
def test_queue_continues_after_a_later_failure_and_lists_it(tmp_path):
    r, calls, run_dir = _run_queue(tmp_path, fail_on="--model-key medgemma15 --items")
    assert r.returncode == 1 and len(calls) == 6
    assert Path(calls[5][0]).name == "run_medsiglip.py"  # later steps still ran
    done = (run_dir / "DONE").read_text(encoding="utf-8")
    assert "status: failed" in done and "failed_steps: run_medgemma15" in done
    assert (run_dir / "FAILED").exists()


@needs_bash
def test_queue_rejects_bad_arguments_and_missing_inputs(tmp_path):
    r, calls, _ = _run_queue(tmp_path / "a", run_id=None)
    assert r.returncode == 2 and calls == []
    r, calls, _ = _run_queue(tmp_path / "b", run_id="bad id")
    assert r.returncode == 2 and calls == []
    r, calls, run_dir = _run_queue(tmp_path / "c", items=False)
    assert r.returncode == 2 and calls == []
    assert "status: preflight_failed" in (run_dir / "DONE").read_text(encoding="utf-8")
