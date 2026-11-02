"""Smoke test of the CORE models on the dev-pool smoke items (PROTOCOL.md section 3).

Usage, from the repository root on the pod::

    python pod/smoke_test.py --items items/pod_items.csv --images-dir images \
        --run-dir runs/<run_id> --run-id <run_id> [--models qwen3vl2b medgemma15]

For each model, in turn (one model in GPU memory at a time): load it, run the smoke items
(``smoke == 1`` in ``pod_items.csv``, 20 dev-pool images) under P0 with the first-token signal,
assert the visual-token count on every input, and write the model's section of
``<run-dir>/smoke_report.json``: visual tokens, median P(yes forms) + P(no forms), the decision
("first_token" when the median coverage is at least 0.90, else "full_sequence"), seconds per item
and peak GPU memory. The records go to ``<run-dir>/smoke/<model_key>.smoke.jsonl`` with run id
``<run_id>-smoke``, apart from the analysis runs.

Exit codes: 0 when every model passed its visual-token assertion (whatever the coverage decision:
a low coverage switches the runs to the full-sequence signal, it does not fail the smoke test);
3 when a visual-token assertion failed; 1 when a model could not be run for another reason.
``pod/queue.sh`` stops on any non-zero exit.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cxrdev import constants  # noqa: E402
from cxrdev.fileio import utc_now  # noqa: E402
from pod import common  # noqa: E402
from pod.common import PodItemsError, VisualTokenAssertionError  # noqa: E402
from pod.run_vlm import VLMBackend, load_backend, run_smoke  # noqa: E402


def smoke_paths(run_dir: str | Path, model_key: str) -> tuple[Path, Path]:
    """(smoke JSONL path, smoke report path) for one model inside a run directory."""
    root = Path(run_dir)
    return root / "smoke" / f"{model_key}.smoke.jsonl", root / "smoke_report.json"


def run_all(
    model_keys: Sequence[str],
    *,
    items_path: str | Path,
    images_dir: str | Path,
    run_dir: str | Path,
    run_id: str,
    factory: Callable[..., VLMBackend] = load_backend,
    device: str | None = None,
    dtype: str | None = None,
    limit: int | None = None,
    local_files_only: bool = False,
    log: Callable[[str], None] = print,
) -> dict[str, str]:
    """Smoke-test each model in turn; return {model_key: "pass" | "token_fail" | "error"}.

    Every model is attempted even if an earlier one fails, so one session reports on all of them;
    a model that cannot be loaded or run gets an error section in the report.
    """
    items = common.read_pod_items(items_path)
    if not any(it.smoke == 1 for it in items):
        raise PodItemsError(f"{items_path}: no smoke items (smoke == 1)")
    missing = common.missing_images(common.plan_tasks(items, "smoke", limit), images_dir)
    if missing:
        raise FileNotFoundError(f"{len(missing)} smoke image(s) missing under {images_dir}, e.g. {missing[:5]}")
    status: dict[str, str] = {}
    for key in model_keys:
        out_path, report_path = smoke_paths(run_dir, key)
        log(f"=== smoke test: {key} ({constants.MODELS[key]['hf_id']}) ===")
        backend: VLMBackend | None = None
        try:
            common.reset_cuda_peak()
            backend = factory(key, device=device, dtype=dtype, revision=None,
                              local_files_only=local_files_only, max_new_tokens=constants.MAX_NEW_TOKENS)
            run_smoke(backend, items, images_dir=images_dir, out_path=out_path, report_path=report_path,
                      run_id=f"{run_id}-smoke", limit=limit, log=log)
            status[key] = "pass"
        except VisualTokenAssertionError as exc:
            log(f"{key}: VISUAL-TOKEN ASSERTION FAILED: {exc}")
            status[key] = "token_fail"
        except Exception as exc:  # noqa: BLE001 - report every model, then fail the step
            log(f"{key}: ERROR {exc!r}")
            log(traceback.format_exc())
            common.update_smoke_report(report_path, key, {
                "run_id": f"{run_id}-smoke", "model_id": constants.MODELS[key]["hf_id"],
                "visual_token_assertion": "not_run", "decision": None,
                "error": f"{type(exc).__name__}: {exc}", "timestamp_utc": utc_now(),
            })
            status[key] = "error"
        finally:
            del backend
            common.free_accelerator_memory()
    return status


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface."""
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n", 1)[0])
    p.add_argument("--items", required=True, help="pod_items.csv (schema S6)")
    p.add_argument("--images-dir", required=True)
    p.add_argument("--run-dir", required=True, help="run directory; smoke_report.json and smoke/ go here")
    p.add_argument("--run-id", required=True)
    p.add_argument("--models", nargs="+", default=list(constants.CORE_MODELS), choices=sorted(constants.MODELS))
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default=None, choices=("bfloat16", "float16", "float32"))
    p.add_argument("--limit", type=int, default=None, help="first N smoke items only")
    p.add_argument("--local-files-only", action="store_true")
    return p


def main(argv: Sequence[str] | None = None, *, backend_factory: Callable[..., VLMBackend] | None = None) -> int:
    """Entry point; returns 0 (all passed), 3 (visual-token assertion failed) or 1 (other error)."""
    args = build_parser().parse_args(argv)
    try:
        status = run_all(
            args.models, items_path=args.items, images_dir=args.images_dir, run_dir=args.run_dir,
            run_id=args.run_id, factory=backend_factory or load_backend, device=args.device, dtype=args.dtype,
            limit=args.limit, local_files_only=args.local_files_only,
        )
    except (PodItemsError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return common.EXIT_USAGE
    report_path = smoke_paths(args.run_dir, args.models[0])[1]
    report: dict[str, Any] = common.read_json(report_path) if report_path.exists() else {}
    print("smoke summary:")
    for key, st in status.items():
        sec = (report.get("models") or {}).get(key, {})
        print(f"  {key}: {st}; visual tokens {sec.get('visual_tokens')}; median coverage "
              f"{sec.get('median_coverage')}; decision {sec.get('decision')}; "
              f"{sec.get('s_per_item_mean')} s/item; peak GPU {sec.get('peak_gpu_mem_gb')} GiB")
    if any(st == "token_fail" for st in status.values()):
        return common.EXIT_TOKEN_ASSERTION
    if any(st != "pass" for st in status.values()):
        return common.EXIT_ERROR
    return common.EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
