#!/usr/bin/env bash
# Stage 1 pod queue: smoke test, full runs of both CORE models, BLIND rows, then MedSigLIP.
#
# Usage, on the pod (nohup-friendly: no prompts, every step logs to its own file):
#   cd /workspace/res201
#   nohup bash pod/queue.sh <RUN_ID> > /workspace/queue_<RUN_ID>.out 2>&1 &
#   tail -f runs/<RUN_ID>/logs/queue.log          # wait for runs/<RUN_ID>/DONE
#
# Steps, in order:
#   1. smoke      pod/smoke_test.py on the 20 dev-pool smoke items under P0 (both models); a failed
#                 smoke test (visual-token assertion or error) STOPS the queue.
#   2. run_<m>    pod/run_vlm.py per model on every item: prompt_set "full" = P0, P1, P2hi on the
#                 image, "p0" = P0 only; signal from the smoke decision (--signal auto).
#   3. blind_<m>  pod/run_vlm.py --blind per model: BLIND-text and BLIND-grey under P0, P1, P2hi.
#   4. medsiglip  pod/run_medsiglip.py on every item (detector_scores.csv, schema S9).
# Steps 2-4 continue after a failure; the failed steps are listed in DONE (and FAILED).
# Re-running the queue with the same RUN_ID resumes the runs (finished records are skipped).
#
# Writes runs/<RUN_ID>/: smoke_report.json, smoke/, <model>.jsonl and <model>.meta.json,
# BLIND-GREY_1024.png, detector_scores.csv and detector_scores.meta.json, logs/, DONE.
#
# Environment overrides: PYTHON (default /workspace/venv/bin/python, else python3), ITEMS (default
# <repo>/items/pod_items.csv), IMAGES_DIR (default <repo>/images), RUNS_DIR (default <repo>/runs),
# MODELS (default "qwen3vl2b medgemma15"), HF_HOME (default /workspace/hf), HF_HUB_OFFLINE
# (default 1: the models come from the cache filled by pod/setup_pod.sh, no download mid-run).
set -uo pipefail

RUN_ID="${1:-}"
if [[ -z "$RUN_ID" ]]; then
  echo "usage: bash pod/queue.sh <RUN_ID>" >&2
  exit 2
fi
if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "RUN_ID may contain only letters, digits, '.', '_' and '-'" >&2
  exit 2
fi

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x /workspace/venv/bin/python ]]; then PYTHON=/workspace/venv/bin/python; else PYTHON=python3; fi
fi
ITEMS="${ITEMS:-$REPO/items/pod_items.csv}"
IMAGES_DIR="${IMAGES_DIR:-$REPO/images}"
RUNS_DIR="${RUNS_DIR:-$REPO/runs}"
MODELS="${MODELS:-qwen3vl2b medgemma15}"
export HF_HOME="${HF_HOME:-/workspace/hf}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

RUN_DIR="$RUNS_DIR/$RUN_ID"
LOG_DIR="$RUN_DIR/logs"
SMOKE_REPORT="$RUN_DIR/smoke_report.json"
mkdir -p "$LOG_DIR"
rm -f "$RUN_DIR/DONE" "$RUN_DIR/FAILED"
QUEUE_LOG="$LOG_DIR/queue.log"
FAILED_STEPS=()
read -r -a MODEL_LIST <<< "$MODELS"

log() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$QUEUE_LOG"
}

# run_step NAME COMMAND...: append the command's output to logs/NAME.log; return its exit status.
run_step() {
  local name="$1"
  shift
  local step_log="$LOG_DIR/$name.log"
  local t0=$SECONDS
  log "START $name"
  "$@" >>"$step_log" 2>&1
  local rc=$?
  log "END   $name rc=$rc after $((SECONDS - t0)) s (logs/$name.log)"
  return "$rc"
}

# finish STATUS: write DONE (and FAILED unless STATUS is ok).
finish() {
  local status="$1"
  {
    echo "status: $status"
    echo "run_id: $RUN_ID"
    echo "finished_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "failed_steps: ${FAILED_STEPS[*]:-none}"
  } >"$RUN_DIR/DONE"
  if [[ "$status" != "ok" ]]; then
    cp "$RUN_DIR/DONE" "$RUN_DIR/FAILED"
  fi
  log "DONE status=$status failed_steps=${FAILED_STEPS[*]:-none}"
}

log "queue start: run_id=$RUN_ID python=$PYTHON models=${MODEL_LIST[*]}"
log "items=$ITEMS images=$IMAGES_DIR run_dir=$RUN_DIR HF_HOME=$HF_HOME HF_HUB_OFFLINE=$HF_HUB_OFFLINE"
if [[ ! -f "$ITEMS" || ! -d "$IMAGES_DIR" || ${#MODEL_LIST[@]} -eq 0 ]]; then
  log "preflight failed: need the item file, the image directory and at least one model"
  FAILED_STEPS+=("preflight")
  finish preflight_failed
  exit 2
fi

# 1. smoke test: stop on failure
if ! run_step smoke "$PYTHON" "$REPO/pod/smoke_test.py" --items "$ITEMS" --images-dir "$IMAGES_DIR" \
    --run-dir "$RUN_DIR" --run-id "$RUN_ID" --models "${MODEL_LIST[@]}"; then
  FAILED_STEPS+=("smoke")
  log "smoke test failed: queue stopped (see logs/smoke.log and smoke_report.json)"
  finish smoke_failed
  exit 1
fi

# 2. full runs (items)
for mk in "${MODEL_LIST[@]}"; do
  run_step "run_$mk" "$PYTHON" "$REPO/pod/run_vlm.py" --model-key "$mk" --items "$ITEMS" \
    --images-dir "$IMAGES_DIR" --out "$RUN_DIR/$mk.jsonl" --run-id "$RUN_ID" \
    --signal auto --smoke-report "$SMOKE_REPORT" || FAILED_STEPS+=("run_$mk")
done

# 3. BLIND-text and BLIND-grey rows
for mk in "${MODEL_LIST[@]}"; do
  run_step "blind_$mk" "$PYTHON" "$REPO/pod/run_vlm.py" --model-key "$mk" --blind \
    --out "$RUN_DIR/$mk.jsonl" --run-id "$RUN_ID" \
    --signal auto --smoke-report "$SMOKE_REPORT" || FAILED_STEPS+=("blind_$mk")
done

# 4. MedSigLIP detector scores
run_step medsiglip "$PYTHON" "$REPO/pod/run_medsiglip.py" --items "$ITEMS" --images-dir "$IMAGES_DIR" \
  --out "$RUN_DIR/detector_scores.csv" || FAILED_STEPS+=("medsiglip")

if ((${#FAILED_STEPS[@]})); then
  finish failed
  exit 1
fi
finish ok
exit 0
