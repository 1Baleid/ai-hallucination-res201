#!/usr/bin/env bash
# Pod setup (safe to re-run): Python environment, Hugging Face access, bf16 GPU check, and the
# pre-download of the three models onto the network volume.
#
# Usage, in an SSH session on the pod, from the repository copy (/workspace/res201):
#   export HF_TOKEN=hf_...      # typed in this session only; huggingface_hub reads it from the
#                               # environment, so no token file is written (and never in the repo)
#   bash pod/setup_pod.sh
#
# Steps: 1. base Python >= 3.11 and torch >= 2.6 (transformers 5.22.1 requires both), CUDA visible;
#        2. virtual environment /workspace/venv on the network volume with --system-site-packages
#           (reuses the template's torch and torchvision, which a constraints file pins while
#           pod/requirements-pod.txt is installed); 3. Hugging Face identity check with HF_TOKEN;
#        4. bf16 GPU check (a bf16 matmul against float32); 5. snapshot download of
#           Qwen/Qwen3-VL-2B-Instruct, google/medgemma-1.5-4b-it and google/medsiglip-448 into
#           HF_HOME=/workspace/hf (config, tokenizer and safetensors files only), with the commit of
#           each written to $HF_HOME/res201_model_revisions.json; 6. import check of the pod scripts.
#
# Environment: HF_TOKEN (required unless SKIP_DOWNLOAD=1), HF_HOME (default /workspace/hf),
# VENV (default /workspace/venv), PYBASE (default python3), SKIP_DOWNLOAD=1 to skip step 5.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_HOME="${HF_HOME:-/workspace/hf}"
VENV="${VENV:-/workspace/venv}"
PYBASE="${PYBASE:-python3}"
mkdir -p "$HF_HOME"

say() { printf '\n== %s\n' "$*"; }

say "1/6 base Python, torch and CUDA ($PYBASE)"
"$PYBASE" - <<'PY'
import sys
if sys.version_info < (3, 11):
    sys.exit(f"Python {sys.version.split()[0]}: transformers 5.22.1 needs Python >= 3.11; "
             "deploy a newer RunPod PyTorch template")
try:
    import torch
except ImportError:
    sys.exit("torch is not installed: use the official RunPod PyTorch template")
major_minor = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
if major_minor < (2, 6):
    sys.exit(f"torch {torch.__version__}: transformers 5.22.1 needs torch >= 2.6; "
             "deploy a RunPod PyTorch template with torch 2.6 or newer")
if not torch.cuda.is_available():
    sys.exit("no CUDA GPU visible to torch")
print("python", sys.version.split()[0], "| torch", torch.__version__, "| CUDA", torch.version.cuda)
PY

say "2/6 virtual environment $VENV and pod/requirements-pod.txt"
if [[ ! -x "$VENV/bin/python" ]]; then
  "$PYBASE" -m venv --system-site-packages "$VENV"
fi
PY="$VENV/bin/python"
CONSTRAINTS="$(mktemp)"
"$PY" - >"$CONSTRAINTS" <<'PY'
import importlib.metadata as md
for name in ("torch", "torchvision"):
    try:
        print(f"{name}=={md.version(name)}")
    except md.PackageNotFoundError:
        pass
PY
echo "pinned while installing: $(tr '\n' ' ' <"$CONSTRAINTS")"
"$PY" -m pip install --disable-pip-version-check -r "$REPO/pod/requirements-pod.txt" -c "$CONSTRAINTS"
rm -f "$CONSTRAINTS"
"$PY" - <<'PY'
import importlib.metadata as md
import sys
want = {"transformers": "5.22.1", "accelerate": "1.15.0"}
for name, version in want.items():
    have = md.version(name)
    if have != version:
        sys.exit(f"{name} {have} installed, {version} expected")
for name in ("transformers", "accelerate", "huggingface_hub", "tokenizers", "torch", "torchvision",
             "sentencepiece", "pillow", "numpy"):
    try:
        print(f"{name}=={md.version(name)}")
    except md.PackageNotFoundError:
        if name == "torchvision":
            sys.exit("torchvision is missing: the Qwen image processor needs it (use the PyTorch template)")
        print(f"{name}: not installed")
PY

if [[ "${SKIP_DOWNLOAD:-0}" != "1" ]]; then
  say "3/6 Hugging Face identity (HF_TOKEN from the environment; nothing written to disk)"
  if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "HF_TOKEN is not set. Run: export HF_TOKEN=hf_...  (a Read token of the account that accepted" >&2
    echo "the Health AI Developer Foundations terms for MedGemma 1.5 and MedSigLIP), then re-run." >&2
    exit 1
  fi
  "$PY" - <<'PY'
from huggingface_hub import whoami
info = whoami()  # reads HF_TOKEN from the environment
print("Hugging Face account:", info.get("name"))
PY
else
  say "3/6 skipped (SKIP_DOWNLOAD=1)"
fi

say "4/6 bf16 GPU check"
"$PY" - <<'PY'
import sys
import torch
if not torch.cuda.is_bf16_supported():
    sys.exit("this GPU does not support bf16; the registered runs are bf16 (use an RTX A5000 or newer)")
torch.manual_seed(0)
a = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
got = (a @ a.T).float()
ref = a.float() @ a.float().T
rel = ((got - ref).abs().max() / ref.abs().max()).item()
if not rel < 0.05:
    sys.exit(f"bf16 matmul relative error {rel:.4f} is implausible")
props = torch.cuda.get_device_properties(0)
print(f"{props.name}: bf16 matmul ok (relative error {rel:.5f}), {props.total_memory / 2**30:.1f} GiB")
PY

if [[ "${SKIP_DOWNLOAD:-0}" != "1" ]]; then
  say "5/6 model download into $HF_HOME"
  "$PY" - "$REPO" <<'PY'
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])
from cxrdev import constants
from huggingface_hub import snapshot_download

repos = [constants.MODELS[k]["hf_id"] for k in constants.CORE_MODELS] + [constants.EXPERT["hf_id"]]
allow = ["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja", "*.tiktoken"]
pins = {}
for repo in repos:
    try:
        path = Path(snapshot_download(repo, allow_patterns=allow))
    except Exception as exc:  # gated repo without accepted terms, wrong token, network
        sys.exit(f"download of {repo} failed: {type(exc).__name__}: {exc}\n"
                 "For google/* models, accept the Health AI Developer Foundations terms on the model page "
                 "with the account of HF_TOKEN.")
    if not any(path.glob("*.safetensors")):
        sys.exit(f"{repo}: no safetensors weights in {path}")
    size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 2**30
    pins[repo] = path.name
    print(f"{repo} @ {path.name} ({size:.1f} GiB)")
out = Path(os.environ["HF_HOME"]) / "res201_model_revisions.json"
out.write_text(json.dumps(pins, indent=2) + "\n", encoding="utf-8")
print("revisions written to", out)
PY
else
  say "5/6 skipped (SKIP_DOWNLOAD=1)"
fi

say "6/6 import check of the pod scripts"
(cd "$REPO" && HF_HUB_OFFLINE=1 "$PY" -c "import sys; sys.path.insert(0, '.'); \
from pod import common, run_vlm, smoke_test, run_medsiglip; print('pod scripts import ok')")

say "setup complete: next, nohup bash pod/queue.sh <RUN_ID> > /workspace/queue_<RUN_ID>.out 2>&1 &"
