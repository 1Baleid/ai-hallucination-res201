# Running Stage 1 on a RunPod GPU pod

This folder holds everything that runs on the rented GPU: the model runner (`run_vlm.py`), the smoke
test (`smoke_test.py`), the MedSigLIP detector scorer (`run_medsiglip.py`), the setup script
(`setup_pod.sh`), the unattended queue (`queue.sh`) and the Python requirements
(`requirements-pod.txt`). This page goes through one pod session from start to finish: deploy,
upload, set up, run, download, then stop and delete the pod.

Commands marked **local** run in PowerShell on your own machine, from the repository folder:

```powershell
cd "C:\Belly'sProjects\Ai Hallucination - RES 299"
```

Commands marked **pod** run in the SSH session on the pod (bash).

## 0. What may go to the pod (and what never does)

The pod gets **only** these four things (PROTOCOL.md section 9):

| Upload | Why |
|---|---|
| `cxrdev/` | the constants, prompts, parser and signal code the runner imports |
| `pod/` | this folder |
| `items/pod_items.csv` | the item list: opaque `item_id`, `nih_filename`, `prompt_set`, `smoke`, nothing else |
| the selected NIH PNGs (`data/kaggle/res201-export/selected_nih_png.zip`) | the pixels, all from NIH ChestX-ray14 |

**Nothing RANZCR-derived ever goes to the pod.** Never upload `items/items.csv`, the reading lists or
keys, `match_table.csv`, `train.csv`, anything under `data/ranzcr/` or `data/kaggle/res201-match/`,
or any file with a uid, an arm, a label or a patient id. `read_pod_items` refuses a
`pod_items.csv` with any extra column, but do not rely on that: check what you upload.

The Hugging Face token goes into the **environment of the SSH session only**, never into a file
in the repository, a script or the pod template (section 4).

## 1. Before the first session (once)

These are the RunPod steps in `docs/HUMAN_STEPS.md` section 3:

1. A RunPod account with credit, and a 50 GB **network volume** in **Secure Cloud**, in a data
   center that lists the **RTX A5000**.
2. An SSH key pair on your machine (**local**), and its public half added on RunPod under
   Settings, SSH Public Keys:

   ```powershell
   ssh-keygen -t ed25519 -f $HOME\.ssh\runpod_ed25519
   Get-Content $HOME\.ssh\runpod_ed25519.pub     # paste this line into RunPod
   ```

3. On huggingface.co, with the account whose token you will use: accept the Health AI Developer
   Foundations terms on the `google/medgemma-1.5-4b-it` and `google/medsiglip-448` model pages,
   and create a **Read** token.

## 2. Deploy the pod

On runpod.io, Pods, Deploy:

1. **Secure Cloud**, GPU **RTX A5000** (24 GB), 1 GPU.
2. Attach the **50 GB network volume**. RunPod mounts it at **`/workspace`**; everything that must
   survive the session (code, images, Python environment, model weights, outputs) lives there.
3. Template: the **official RunPod PyTorch** template. Pick a version whose image tag shows
   Python 3.11 or newer and torch 2.6 or newer (transformers 5.22.1 needs both);
   `setup_pod.sh` stops with a clear message if they are too old.
4. Make sure **TCP port 22** is exposed (the PyTorch template does this) so the pod offers
   **"SSH over exposed TCP"**. That connection supports `scp`; the basic proxied SSH does not.
5. Do **not** put the Hugging Face token in the template's environment variables.
6. Deploy. When it is running, open Connect and copy the "SSH over exposed TCP" line. It looks
   like `ssh root@<IP> -p <PORT> -i ~/.ssh/id_ed25519`. Below, `<IP>` and `<PORT>` are those values.

Check the connection before uploading anything (**local**):

```powershell
ssh -p <PORT> -i $HOME\.ssh\runpod_ed25519 root@<IP> "nvidia-smi --query-gpu=name,memory.total --format=csv; df -h /workspace"
```

You should see `NVIDIA RTX A5000` and the volume at `/workspace`. The GPU bills from now on.

## 3. Upload

Pack the code and the item list (**local**; `tar` ships with Windows 10 and 11):

```powershell
tar -czf "$env:TEMP\res201_pod_upload.tgz" --exclude=__pycache__ cxrdev pod items/pod_items.csv
tar -tzf "$env:TEMP\res201_pod_upload.tgz"    # check: only cxrdev/, pod/ and items/pod_items.csv
scp -P <PORT> -i $HOME\.ssh\runpod_ed25519 "$env:TEMP\res201_pod_upload.tgz" data\kaggle\res201-export\selected_nih_png.zip root@<IP>:/workspace/
```

Unpack on the pod (**pod**):

```bash
mkdir -p /workspace/res201 && cd /workspace/res201
tar -xzf /workspace/res201_pod_upload.tgz
python3 -m zipfile -t /workspace/selected_nih_png.zip     # no "corrupted" line before "Done testing"
python3 -m zipfile -e /workspace/selected_nih_png.zip images/
ls images | wc -l                                         # = the number of selected PNGs (660 at the registered n)
python3 -c "import csv, os; r = list(csv.DictReader(open('items/pod_items.csv'))); m = [x['nih_filename'] for x in r if not os.path.isfile('images/' + x['nih_filename'])]; print(len(r), 'items,', len(m), 'missing', m[:5])"
```

The last line must say `0 missing`. The repository copy on the pod is `/workspace/res201`, with the
images in `/workspace/res201/images` and the item list in `/workspace/res201/items/pod_items.csv`,
which are the defaults of `queue.sh`.

## 4. Set up (Python environment and models)

Give the session your Hugging Face token without writing it anywhere (**pod**). `read -rs` does
not echo the token and keeps it out of the shell history:

```bash
cd /workspace/res201
read -rsp "Hugging Face token: " HF_TOKEN && export HF_TOKEN && echo
bash pod/setup_pod.sh
unset HF_TOKEN
```

Do not run `hf auth login` on the pod: it would save the token in `/workspace/hf/token` on the
volume. `setup_pod.sh` is safe to re-run. It:

1. checks Python, torch and CUDA;
2. creates the virtual environment `/workspace/venv` (it reuses the template's torch) and installs
   `pod/requirements-pod.txt` (transformers 5.22.1, accelerate 1.15.0, ...);
3. checks the token (`whoami`), from the environment only;
4. checks bf16 on the GPU;
5. downloads `Qwen/Qwen3-VL-2B-Instruct`, `google/medgemma-1.5-4b-it` and `google/medsiglip-448`
   into `/workspace/hf` (about 17 GB in total) and writes their commits to
   `/workspace/hf/res201_model_revisions.json`;
6. checks that the pod scripts import.

A download failure for a `google/` model almost always means the terms (section 1, step 3) were
not accepted with the account of this token.

## 5. Run the queue

Choose a run id (letters, digits, `.`, `_`, `-`), for example `core-20261110`, and start the queue
so that it keeps running if the SSH connection drops (**pod**):

```bash
cd /workspace/res201
RUN_ID=core-20261110
nohup bash pod/queue.sh "$RUN_ID" > "/workspace/queue_$RUN_ID.out" 2>&1 &
tail -f "runs/$RUN_ID/logs/queue.log"        # Ctrl+C stops watching, not the queue
```

The queue runs these steps in order, each with its own log in `runs/<RUN_ID>/logs/`:

1. `smoke`: `smoke_test.py` on the 20 dev-pool smoke items under P0, for both models. It asserts the
   visual-token count (1,024 for Qwen3-VL-2B at min = max pixels = 1,048,576; 256 for MedGemma) and
   checks the median first-token coverage (P(yes forms) + P(no forms) >= 0.90; below that, the runs
   switch to the full-sequence signal). **A failed smoke test stops the queue.**
2. `run_qwen3vl2b`, `run_medgemma15`: every item: P0, P1 and P2hi on the A, B and U images
   (`prompt_set` "full"), P0 on the dev pool (`prompt_set` "p0").
3. `blind_qwen3vl2b`, `blind_medgemma15`: BLIND-text and BLIND-grey under P0, P1 and P2hi.
4. `medsiglip`: the MedSigLIP ETT score for every item (`detector_scores.csv`).

It is finished when `runs/<RUN_ID>/DONE` exists (**pod**):

```bash
cat "runs/$RUN_ID/DONE"          # status: ok, failed_steps: none
python3 -m json.tool "runs/$RUN_ID/smoke_report.json" | head -60
```

The CORE queue should take under an hour. Progress lines show counts and timing only, never
answers or scores (apart from the smoke step, which uses dev-pool images only), so the logs are safe
to watch before the blind reading.

If a step failed, `DONE` (and `FAILED`) lists it; its log in `runs/<RUN_ID>/logs/` says why. Fix the
cause and start the queue again with the **same** run id: finished records are skipped and the runs
resume. A visual-token assertion failure (exit code 3) means the image settings or the
transformers version are not the registered ones: do not work around it. If the smoke decision for a
model is `full_sequence`, the switch must be recorded in PROTOCOL.md section 11.

What `runs/<RUN_ID>/` holds at the end: `smoke_report.json`, `smoke/` (dev-pool smoke records),
`qwen3vl2b.jsonl` and `medgemma15.jsonl` (one record per item x prompt, plus 6 BLIND rows) with their
`.meta.json` run settings, `BLIND-GREY_1024.png`, `detector_scores.csv` with its `.meta.json`,
`logs/` and `DONE`. Each model's JSONL should have 3 x (number of "full" items) + (number of "p0"
items) + 6 lines: 3 x 360 + 300 + 6 = 1,386 at the registered n.

## 6. Download the results

Pack the run on the pod, with the queue's console output and the model commits (**pod**):

```bash
cd /workspace/res201
cp "/workspace/queue_$RUN_ID.out" "runs/$RUN_ID/logs/"
cp /workspace/hf/res201_model_revisions.json "runs/$RUN_ID/"
tar -czf "/workspace/${RUN_ID}_runs.tgz" -C runs "$RUN_ID"
```

Copy it to your machine and unpack it into the git-ignored `runs/` folder (**local**):

```powershell
New-Item -ItemType Directory -Force runs | Out-Null
scp -P <PORT> -i $HOME\.ssh\runpod_ed25519 root@<IP>:/workspace/<RUN_ID>_runs.tgz runs\
tar -xzf runs\<RUN_ID>_runs.tgz -C runs
Get-Content runs\<RUN_ID>\DONE
```

Every record carries `image_sha256`, the SHA-256 of the PNG it was run on, so the local analysis can
check each one against `data/kaggle/res201-export/manifest.csv`.

**Blind reading rule (PROTOCOL.md section 2):** until `verified_items.csv` is saved, open only
`DONE`, the logs, `smoke_report.json` and the `smoke/` records (dev-pool items). Do not open
`qwen3vl2b.jsonl`, `medgemma15.jsonl` or `detector_scores.csv` before then.

## 7. Stop and delete the pod

As soon as the download is checked:

1. On runpod.io, Pods: **Stop** the pod, then **Terminate** (delete) it. With a network volume
   attached RunPod may offer only Terminate; that is fine, because the code, images, environment,
   models and outputs are on the volume and the results are on your machine.
2. Open the Billing page and confirm that no pod is still running.
3. Keep the network volume while Stage 1 may still need the pod (it saves the setup and the 17 GB
   download next time). Delete it at the end of the project. It holds NIH PNGs, code, model weights
   and outputs only.

The next session attaches the same volume and starts at section 5 (`setup_pod.sh` is quick to
re-run if the template changed).

## 8. Expected cost

From STAGE1_PLAN.md section 6: a Secure Cloud RTX A5000 costs **$0.27 per hour**. The CORE runs
need **2-4 pod-hours** including setup, upload and download (about **$0.55-$1.10**). The 50 GB network
volume costs $3.50 per month, and $3 is set aside for idle time. The Stage 1 ceiling is $15. The pod
bills every hour it runs, even when it is idle, so stop it right after the download. Record each
session's hours and cost in `ledger.csv`.

## Files in this folder

| File | Role |
|---|---|
| `setup_pod.sh` | environment, token check, bf16 check, model download (section 4) |
| `queue.sh` | smoke, full runs, BLIND rows, MedSigLIP, with logs and a `DONE` marker (section 5) |
| `smoke_test.py` | 20-item dev-pool smoke test per model, writes `smoke_report.json` |
| `run_vlm.py` | runs one model over the item list or the BLIND rows, writes the JSONL records (schema S7) |
| `run_medsiglip.py` | MedSigLIP zero-shot ETT score per item, writes `detector_scores.csv` (schema S9) |
| `common.py` | shared helpers: item list checks, task plan, images, signals, record checks, resume |
| `requirements-pod.txt` | pinned Python packages installed by `setup_pod.sh` |
