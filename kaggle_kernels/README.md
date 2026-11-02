# Kaggle kernels: hashing, RANZCR-to-NIH matching, PNG export

Three private Kaggle CPU kernels build the image linkage for Stage 1 (PROTOCOL sections 1, 2, 8 and 9).
The only linkage is between two public de-identified releases of the same radiographs (RANZCR CLiP
JPEGs and NIH ChestX-ray14 PNGs), matched by image content. No attempt is made to identify any person.

| Kernel (slug) | Inputs | Outputs (in `/kaggle/working`) |
|---|---|---|
| `<user>/res201-hash` | competition `ranzcr-clip-catheter-line-classification`, dataset `nih-chest-xrays/data` | `yield_table.csv`, `dimkey_test.json`, `nih_hashes_<folder>.csv` (one per NIH folder), `run_summary.json` |
| `<user>/res201-match` | the competition, the NIH dataset, kernel output `<user>/res201-hash` | `match_table.csv` (S3), `match_stats.json`, `calibration.png`, `spotcheck_01.png` .. `spotcheck_06.png` |
| `<user>/res201-export` | the NIH dataset, private dataset `<user>/res201-nih-filelist` | `selected_nih_png.zip`, `manifest.csv` |

All three are scripts with `is_private: true`, `enable_internet: false` and `enable_gpu: false`, so no
phone verification is needed. Inputs are found by globbing under `/kaggle/input`. Mount names are never
hard-coded, so the kernels work with both the older `/kaggle/input/<slug>/` layout and the newer
`/kaggle/input/{competitions,datasets,notebooks}/...` layout.

## What each kernel does

**res201-hash** (expected 1-2 h on 4 CPU cores):

1. Yield table: reads `train.csv`, which holds the RANZCR CLiP labels (43 trained labellers:
   radiologists, radiology trainees, hospital medical officers). It counts images and patients by ETT
   state and the U strata by other labelled lines (`cxrdev.ranzcr.yield_table`) and writes aggregates
   only.
2. Dimension-key test: reads the header sizes of 200 JPEGs (evenly spaced over the sorted list) and
   compares them with NIH `OriginalImage[Width,Height]`. It reports the share found and the candidates
   per size, with no file names.
3. NIH hashes: hashes the NIH folders (`images_001` .. `images_012`) one at a time, each inside
   try/except. Each folder gets a 256-bit pHash (`imagehash.phash` bit for bit, `hash_size=16`) and a
   256-bit dHash per PNG, written to `nih_hashes_<folder>.csv` (`nih_filename, folder, ph, dh`) as soon
   as that folder finishes.
4. `run_summary.json` is rewritten after every step and folder. The script always exits with code 0,
   so a commit saves whatever finished, even if a folder fails.

**res201-match**:

1. Hashes every labelled RANZCR JPEG (`train/<StudyInstanceUID>.jpg`) in memory. It uses one
   reduced-scale JPEG decode (draft mode, at least 128 px) for both the 256-bit pHash and the 128-px
   thumbnail. RANZCR hashes are never written to `/kaggle/working`.
2. Runs `cxrdev.matching.match_all`:
   - candidates come from the dimension key (JPEG size = NIH original size), with a per-image global
     fallback when that bucket is empty;
   - the cut is the Otsu valley of the best-distance histogram, capped at 64 of 256 bits;
   - a margin of 16 bits is required;
   - 128-px NCC confirmation needs NCC >= 0.95;
   - the NCC tie-break over the top 5 candidates needs a gap of at least 0.02;
   - then one-to-one assignment and a single NIH patient per RANZCR patient.
   - If the hash kernel's `dimkey_test.json` shows that fewer than half of the sampled JPEG sizes are
     NIH sizes, the key is switched off and every image is matched globally. The reason is recorded in
     `match_stats.json` (`run.config_reason`).
3. Writes `match_table.csv` (one row per hashed labelled JPEG; decode failures are counted in
   `run.n_ranzcr_errors`), `match_stats.json` and `calibration.png`. `match_stats.json` holds:
   - histograms of d1 and d2 - d1, counts per status and route, and the match rate;
   - the ambiguous count and `ambiguous_share_within_patient`, the share of ambiguous UIDs whose two
     closest NIH candidates are films of one NIH patient;
   - the cut used, the configuration and timings.
4. Draws 60 random matched pairs (seed `derive_seed("spotcheck")`) as six grids of 10. Each row shows
   the RANZCR JPEG on the left and the NIH PNG on the right at 256 px, labelled with the pair number,
   d1, NCC, route and NIH file name (no UID). This is the 60-pair visual spot check of PROTOCOL
   section 1. Review the grids in the Kaggle notebook viewer (or in the downloaded copies under
   `data/`) before the mapping is used.

**res201-export**: reads `nih_filelist.csv` (exactly one column, `nih_filename`; any other column is
refused). It copies the listed PNGs to a staging folder under `/kaggle/temp`, hashes them, writes
`selected_nih_png.zip` (stored, fixed timestamps, sorted, so the same list gives the same bytes) and
`manifest.csv` (`nih_filename, sha256, bytes`). It verifies every zip entry against the manifest, then
removes the loose copies, so only the zip and the manifest remain. Names that are not found are
printed in the log and left out.

## Build, push, poll, download

From the repository root, with the project venv (`.venv\Scripts\python.exe` on Windows) and Kaggle
credentials in `%USERPROFILE%\.kaggle\kaggle.json` (or `KAGGLE_USERNAME` / `KAGGLE_KEY`):

```
python kaggle_kernels/build_kernels.py --username <user>          # writes kaggle_kernels/build/<slug>/
python kaggle_kernels/run_kernel.py hash  --username <user>       # push, poll, download, summary
python kaggle_kernels/run_kernel.py match --username <user>
python kaggle_kernels/make_filelist_dataset.py --username <user> --filelist items/nih_filelist.csv
python kaggle_kernels/run_kernel.py export --username <user>
```

- `build_kernels.py` bundles each kernel into one self-contained `kernel.py`. The bundle holds every
  `cxrdev` module it needs in dependency order, with the `from cxrdev...` lines removed, then the kernel
  main. The build also writes `kernel-metadata.json` and checks that the bundle compiles, parses as
  Python 3.10, imports no `cxrdev`, and has no top-level name defined twice. Options:
  - `--kernels hash,match` builds a subset.
  - `--hash-folders images_005,images_009` hashes only those folders.
  - `--match-config '{"cut": 30}'` overrides `MatchConfig` fields, for example a fixed cut after
    reviewing `calibration.png`.
- `run_kernel.py <hash|match|export>` follows these steps:
  1. Checks that the build folder's metadata is private with internet off. It also checks that the
     download folder `data/kaggle/<slug>/` is git-ignored and holds no tracked file. If either check
     fails, it refuses.
  2. Runs `kaggle kernels push -p kaggle_kernels/build/<slug>`.
  3. Polls `kaggle kernels status <user>/<slug>/<version>` every 60 s, up to `--timeout-min`
     (default 360). Kaggle CLI 2.x prints `has status "KernelWorkerStatus.COMPLETE"`.
  4. Runs `kaggle kernels output <user>/<slug>/<version> -p data/kaggle/<slug> -o`, also after an
     error, so the log is kept.
  5. Prints a summary.

  The options `--no-push`, `--no-wait`, `--no-download` and `--version N` re-download an earlier run.
  Exit code 0 means complete; 1 means error, cancelled or timed out; 2 means refused or the push
  failed.
- `make_filelist_dataset.py` validates the list (one column `nih_filename`, NIH names only, no
  duplicates). It stages it with `dataset-metadata.json` (`isPrivate: true`) in the git-ignored folder
  `data/kaggle/res201-nih-filelist-upload/`, which may hold nothing else. It then runs
  `kaggle datasets version -p <stage> -m <message>` if the dataset exists, else
  `kaggle datasets create -p <stage>`. It never passes `--public` / `-u`. Use `--dry-run` to see the
  command without running it.

The Kaggle CLI is always called through `subprocess`, because importing the `kaggle` package starts
authentication. The metadata fields follow the Kaggle CLI 2.2.5 source (`kernels push` reads `id`,
`title`, `code_file`, `language`, `kernel_type`, `is_private`, `enable_gpu`, `enable_tpu`,
`enable_internet`, `dataset_sources`, `competition_sources`, `kernel_sources`, `model_sources`). Each
title slugifies to its slug (`RES201 hash` -> `res201-hash`), so the CLI does not warn.

### If something fails

- A hash folder failed: `run_summary.json` lists it. Push the hash kernel again; a new version
  re-hashes everything in about 1 h. Alternatively, rebuild with `--hash-folders <missing>`. A partial
  version holds only those shards, so keep the earlier shards: download them and attach them to the
  match kernel as a private dataset. They hold NIH hashes only, never RANZCR hashes.
- The calibration plot shows no clean valley, or the cut is at the cap: review the spot-check grids.
  Re-run the match kernel with `--match-config '{"cut": N}'` if a fixed cut is justified, and record the
  decision in `PROTOCOL.md` section 11.
- No clean pair list by Day-3 noon: PROTOCOL section 8 (RANZCR-direct fallback).

## Data-boundary rules (PROTOCOL section 9; Kaggle rules 7.A and 7.B)

- The kernels and every output dataset (hash shards, the mapping table, the file list) are private.
  The filename-list dataset is created private and only versioned afterwards.
- The RANZCR hash shard never leaves Kaggle: RANZCR hashes and thumbnails exist only in the memory of
  the match session. The match kernel writes no RANZCR hash to `/kaggle/working`, and the tests check
  this.
- Downloads go only to `data/kaggle/<slug>/`, which is git-ignored. `run_kernel.py` refuses any folder
  that is tracked or not ignored. `match_table.csv` (UIDs linked to NIH file names) and `train.csv` are
  the single working copy on the student's machine. They never go to the pod, git or any public
  place.
- The pod receives NIH PNGs (from `selected_nih_png.zip`) and `pod_items.csv` only. `nih_filelist.csv`
  carries NIH file names only: no arm, label, UID or patient column.
- NIH attribution on every artefact: NIH Clinical Center; Wang et al., CVPR 2017; the NIH Box release.
- No file or folder name contains `phash` or `mapping` (the `.gitignore` hides those patterns).

## Local runs and tests

Each `kernel_main.py` runs unbundled against a fake tree:

```
python kaggle_kernels/hash_kernel/kernel_main.py --input-root <fake_input> --working <out> --temp <tmp> --workers 1
```

`tests/test_kernels.py` builds the bundles and runs each bundled `kernel.py` in an isolated
interpreter (`python -I`) against a synthetic `/kaggle/input` tree. The tree holds random smooth PNGs,
JPEG copies, a fake `train.csv` with the real RANZCR column names, and a split-header
`Data_Entry_2017.csv`. The tests check every output and its columns. They cover the per-folder
try/except, the dimension-key switch and the export refusal of extra columns. They check that no
RANZCR hash appears under `working`, and they cover push/poll/download with a fake CLI. Use
`--workers 1` on Windows. On Kaggle (Linux) the default `--workers 4` uses a fork-based process pool.
That path, the real Kaggle runtime, the mount layout and the CLI's network calls cannot be exercised
on this machine.
