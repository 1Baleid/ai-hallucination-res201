"""MedSigLIP zero-shot ETT score for every pod item (schema S9, ``detector_scores.csv``).

Usage, from the repository root on the pod::

    python pod/run_medsiglip.py --items items/pod_items.csv --images-dir images \
        --out runs/<run_id>/detector_scores.csv [--batch-size 16] [--dtype float32]

Expert of Arm 1 (PROTOCOL.md sections 3 and 7): ``google/medsiglip-448`` (Health AI Developer
Foundations terms) loaded with ``AutoModel`` / ``AutoProcessor`` as on the model card. Each image is
scored against the CCD prompt pair in ``constants.EXPERT["texts"]`` ("a chest x-ray with an
endotracheal tube" / "... without ..."), tokenised with ``padding="max_length"``; ``s`` is the
softmax over ``logits_per_image`` across the two prompts, P(with ETT), and the two logits are kept;
each row also records ``model_id``, the resolved ``revision`` and the ``dtype`` the scorer ran in.
MedSigLIP is never an evaluated model; its scores are reported per arm only after the local join.

Images are read like the VLM inputs (``convert("L").convert("RGB")``). With ``--resize card``
(default) each image is first resized to 448 x 448 by bilinear interpolation without antialiasing
(half-pixel centres, values truncated to uint8), the resize the model card recommends for
reproducing its evaluations (``tf.image.resize(..., method="bilinear", antialias=False)``),
replicated here in numpy because TensorFlow is not installed; the transformers processor then
only rescales and normalises the 448 x 448 image. ``--resize processor`` leaves the resize to the
transformers image processor. The output also records ``model_id`` and the resolved ``revision``;
a ``detector_scores.meta.json`` sidecar holds the settings and the SHA-256 of the CSV.
"""
from __future__ import annotations

import argparse
import platform
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from cxrdev import constants  # noqa: E402
from cxrdev.fileio import utc_now  # noqa: E402
from pod import common  # noqa: E402
from pod.common import S9_COLUMNS, PodItem, PodItemsError  # noqa: E402

RESIZE_MODES: tuple[str, ...] = ("card", "processor")


def _tf_axis(in_size: int, out_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lower index, upper index and interpolation weight per output position (TensorFlow, half-pixel)."""
    scale = np.float32(in_size) / np.float32(out_size)
    pos = (np.arange(out_size, dtype=np.float32) + np.float32(0.5)) * scale - np.float32(0.5)
    floor = np.floor(pos)
    lower = np.maximum(floor, 0).astype(np.int64)
    upper = np.minimum(np.ceil(pos), in_size - 1).astype(np.int64)
    upper = np.maximum(upper, 0)
    return lower, upper, (pos - floor).astype(np.float32)


def tf_bilinear_resize(arr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Float32 replica of TensorFlow's ``resize_bilinear`` with ``half_pixel_centers=True``.

    ``arr`` has shape (H, W, C). For each output pixel the source position is
    ``(i + 0.5) * in / out - 0.5``; lower = max(floor, 0), upper = min(ceil, in - 1),
    lerp = pos - floor; the value is computed in TensorFlow's order
    ``top = tl + (tr - tl) * x_lerp``, ``bottom = bl + (br - bl) * x_lerp``,
    ``top + (bottom - top) * y_lerp`` (so flat regions stay exact).

    Every step is float32, as in TensorFlow's CPU kernel (``resize_bilinear_op.cc``), and the output
    is bit-identical to a scalar transcription of that kernel (tests/test_pod_runner.py). The float32
    source positions differ from exact arithmetic by up to about 1e-4 pixel, so the output differs
    from a float64 bilinear formula by up to a few hundredths of a grey level, and after uint8
    truncation about 1% of the pixels of a 1024 -> 448 resize differ by one level. That is
    TensorFlow's behaviour; do not "fix" it to float64.
    """
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim != 3:
        raise ValueError(f"expected an (H, W, C) array, got shape {a.shape}")
    y_lo, y_hi, y_l = _tf_axis(a.shape[0], out_h)
    x_lo, x_hi, x_l = _tf_axis(a.shape[1], out_w)
    top_rows, bottom_rows = a[y_lo], a[y_hi]
    xl = x_l[None, :, None]
    top = top_rows[:, x_lo] + (top_rows[:, x_hi] - top_rows[:, x_lo]) * xl
    bottom = bottom_rows[:, x_lo] + (bottom_rows[:, x_hi] - bottom_rows[:, x_lo]) * xl
    return (top + (bottom - top) * y_l[:, None, None]).astype(np.float32)


def card_resize(image: Image.Image, size: int = int(constants.EXPERT["image_size"])) -> Image.Image:
    """Bilinear resize to ``size`` x ``size`` without antialiasing, truncated to uint8 (model-card recipe).

    Equivalent to ``tf.image.resize(img, [size, size], method="bilinear", antialias=False)``
    followed by ``.astype(np.uint8)`` as in the MedSigLIP model card, computed with
    ``tf_bilinear_resize`` (numpy) because TensorFlow is not installed on the pod.
    """
    res = tf_bilinear_resize(np.asarray(image.convert("RGB")), size, size)
    return Image.fromarray(np.clip(res, 0.0, 255.0).astype(np.uint8))


def softmax_rows(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax in float64 (stable)."""
    x = np.asarray(logits, dtype=np.float64)
    m = x.max(axis=1, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=1, keepdims=True)


class MedSiglipScorer:
    """MedSigLIP model and processor with a batch scoring method.

    ``model`` and ``processor`` are a transformers ``AutoModel`` / ``AutoProcessor`` pair
    (``load_scorer``) or test doubles: ``processor(text=..., images=..., padding="max_length",
    return_tensors="pt")`` and ``model(**inputs).logits_per_image`` of shape (n_images, 2).
    """

    def __init__(self, model: Any, processor: Any, *, model_id: str, revision: str, device: str,
                 dtype_name: str, resize: str = "card", texts: Sequence[str] = tuple(constants.EXPERT["texts"])) -> None:
        if resize not in RESIZE_MODES:
            raise ValueError(f"resize must be one of {RESIZE_MODES}")
        if len(texts) != 2:
            raise ValueError("MedSigLIP scoring needs exactly two prompts (with, without)")
        self.model = model
        self.processor = processor
        self.model_id = model_id
        self.revision = revision
        self.device = device
        self.dtype_name = dtype_name
        self.resize = resize
        self.texts = [str(t) for t in texts]

    def score_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        """Logits of shape (n, 2): column 0 "with an endotracheal tube", column 1 "without"."""
        import torch

        if not images:
            return np.zeros((0, 2))
        prepared = [card_resize(im) if self.resize == "card" else im for im in images]
        inputs = self.processor(text=self.texts, images=prepared, padding="max_length", return_tensors="pt")
        dtype = common.torch_dtype(self.dtype_name)
        batch: dict[str, Any] = {}
        for key in list(inputs.keys()):
            value = inputs[key]
            if isinstance(value, torch.Tensor):
                value = value.to(self.device)
                if torch.is_floating_point(value):
                    value = value.to(dtype)
            batch[key] = value
        with torch.inference_mode():
            out = self.model(**batch)
        logits = out.logits_per_image.detach().to(torch.float32).cpu().numpy()
        if logits.shape != (len(images), 2):
            raise RuntimeError(f"logits_per_image has shape {logits.shape}, expected ({len(images)}, 2)")
        if not np.all(np.isfinite(logits)):
            raise RuntimeError("non-finite MedSigLIP logits")
        return logits.astype(np.float64)


def load_scorer(*, device: str | None = None, dtype: str = "float32", revision: str | None = None,
                local_files_only: bool = False, resize: str = "card") -> MedSiglipScorer:
    """Load google/medsiglip-448 with AutoModel / AutoProcessor at one resolved commit."""
    from transformers import AutoModel, AutoProcessor

    hf_id = str(constants.EXPERT["hf_id"])
    dev = common.normalise_device(device)
    rev = common.resolve_model_revision(hf_id, revision, local_files_only=local_files_only)
    processor = AutoProcessor.from_pretrained(hf_id, revision=rev, local_files_only=local_files_only)
    model = AutoModel.from_pretrained(hf_id, revision=rev, dtype=common.torch_dtype(dtype),
                                      local_files_only=local_files_only)
    model.to(dev)
    model.eval()
    return MedSiglipScorer(model, processor, model_id=hf_id, revision=rev, device=dev, dtype_name=dtype, resize=resize)


def score_items(scorer: MedSiglipScorer, items: Sequence[PodItem], images_dir: str | Path, *,
                batch_size: int = 16, log: Callable[[str], None] = print) -> list[dict[str, Any]]:
    """S9 rows for ``items`` in their order (one row per item; an image shared by two items is scored once)."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    root = Path(images_dir)
    missing = sorted({it.nih_filename for it in items if not (root / it.nih_filename).is_file()})
    if missing:
        raise FileNotFoundError(f"{len(missing)} image(s) missing under {root}, e.g. {missing[:5]}")
    names = list(dict.fromkeys(it.nih_filename for it in items))
    scores: dict[str, np.ndarray] = {}
    started = time.perf_counter()
    for start in range(0, len(names), batch_size):
        chunk = names[start:start + batch_size]
        images = [common.load_image_file(root / n)[0] for n in chunk]
        logits = scorer.score_images(images)
        for n, row in zip(chunk, logits):
            scores[n] = row
        done = min(start + batch_size, len(names))
        log(f"[{done:>4}/{len(names)}] images scored, {time.perf_counter() - started:.1f} s")
    rows: list[dict[str, Any]] = []
    for it in items:
        logit_with, logit_without = (float(v) for v in scores[it.nih_filename])
        s = float(softmax_rows(np.array([[logit_with, logit_without]]))[0, 0])
        rows.append({"item_id": it.item_id, "nih_filename": it.nih_filename, "s": s, "logit_with": logit_with,
                     "logit_without": logit_without, "model_id": scorer.model_id, "revision": scorer.revision,
                     "dtype": scorer.dtype_name})
    return rows


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface."""
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n", 1)[0])
    p.add_argument("--items", required=True, help="pod_items.csv (schema S6)")
    p.add_argument("--images-dir", required=True)
    p.add_argument("--out", required=True, help="detector_scores.csv (schema S9)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default="float32", choices=("float32", "bfloat16"))
    p.add_argument("--revision", default=None)
    p.add_argument("--resize", default="card", choices=RESIZE_MODES)
    p.add_argument("--limit", type=int, default=None, help="first N items only (quick tests)")
    p.add_argument("--local-files-only", action="store_true")
    return p


def main(argv: Sequence[str] | None = None, *, scorer_factory: Callable[..., MedSiglipScorer] | None = None) -> int:
    """Entry point; writes S9 and its meta sidecar. Returns 0, or 2 for an input error."""
    args = build_parser().parse_args(argv)
    if args.limit is not None and args.limit < 1:
        print("error: --limit must be positive", file=sys.stderr)
        return common.EXIT_USAGE
    try:
        items = common.read_pod_items(args.items)
    except PodItemsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return common.EXIT_USAGE
    if args.limit is not None:
        items = items[:args.limit]
    root = Path(args.images_dir)
    missing = sorted({it.nih_filename for it in items if not (root / it.nih_filename).is_file()})
    if missing:
        print(f"error: {len(missing)} image(s) missing under {root}, e.g. {missing[:5]}", file=sys.stderr)
        return common.EXIT_USAGE
    started = utc_now()
    factory = scorer_factory or load_scorer
    scorer = factory(device=args.device, dtype=args.dtype, revision=args.revision,
                     local_files_only=args.local_files_only, resize=args.resize)
    rows = score_items(scorer, items, root, batch_size=args.batch_size)
    digest = common.write_csv_rows(args.out, S9_COLUMNS, rows)
    meta: dict[str, Any] = {
        "started_utc": started,
        "finished_utc": utc_now(),
        "model_id": scorer.model_id,
        "revision": scorer.revision,
        "texts": scorer.texts,
        "padding": "max_length",
        "resize": scorer.resize,
        "image_size": int(constants.EXPERT["image_size"]),
        "dtype": scorer.dtype_name,
        "device": scorer.device,
        "batch_size": args.batch_size,
        "n_rows": len(rows),
        "csv_sha256": digest,
        "items_file": str(args.items),
        "limit": args.limit,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    try:
        meta.update(common.library_versions())
        meta["gpu_name"] = common.gpu_name(scorer.device)
    except ImportError:  # pragma: no cover - torch and transformers are present wherever this runs
        pass
    common.write_json(common.meta_path(args.out), meta)
    print(f"wrote {len(rows)} rows to {args.out} (sha256 {digest})")
    return common.EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
