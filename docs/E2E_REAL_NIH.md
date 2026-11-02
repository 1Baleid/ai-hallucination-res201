# End-to-end test on real NIH chest X-rays (2026-11-02)

## What the test does

`scripts/e2e_real_nih.py` (run by the slow test `tests/test_e2e_real_nih.py`) pushes real public NIH ChestX-ray14 images through the whole Stage 1 pipeline on this PC. Only two things are simulated: the RANZCR CLiP competition data, which never exists on this machine, and the models, which are never loaded.

1. **Fake Kaggle folder** (`runs/e2e_real/kaggle_input`, git-ignored):
   - **NIH part:** 1,510 real NIH PNGs from 400 patients (up to 4 follow-ups each), with their real metadata rows in Kaggle's split-header CSV form.
   - **Simulated RANZCR part:** 595 JPEGs made the way RANZCR made theirs (bicubic resize to the image's original size, JPEG quality 95):
     - 565 come from 150 of those patients;
     - 30 come from NIH images that are not in the NIH part: 20 from other patients and 10 from unused follow-ups of included patients, which is the hardest case.
   - **Made-up `train.csv`:** it has the real RANZCR column names. ETT appears on 44% of the images, in runs within each patient. Central lines are on about 97% of the images, as in the real data, so there are very few no-line patients.
2. **The three bundled Kaggle kernels** (hash, match, export) run exactly as they will on Kaggle, each in a separate isolated Python process.
3. **The other steps** then run with the project's own scripts:
   - `select_items.py`, with an excluded-filenames list of 6 names: 5 matched images and 1 that is not in the pool;
   - the pod scripts (smoke test, model runs, BLIND rows, MedSigLIP), driven by fake models that only see pixels;
   - `parse_runs.py`;
   - a blind reading and a retest 49 hours later, recorded through the real reading-tool code;
   - `verify_join.py`, `analyze.py --fast` and `build_report.py`.
4. **Leak check:** the test checks that no RANZCR UID or PatientID, read id or item id appears in the results or in the built report.

## What it found

| Check | Result |
|---|---|
| JPEGs whose source is in the NIH part | **565 of 565 matched to exactly the right PNG (100%)** |
| JPEGs whose source is absent | **30 of 30 unmatched**, including the 10 same-patient follow-ups |
| Wrong matches | **0** |
| Hamming cut chosen automatically (Otsu valley) | **29 bits** (of 256) |
| Distance d1, true matches | 0 to 12 bits (median 2; 95% at or below 6) |
| Distance d1, absent sources | 46 to 120 bits (median 80): a wide empty gap around the cut |
| Margin to the second-best candidate | at least 44 bits (rule: 16); 79 JPEGs were alone in their size bucket |
| NCC confirmation (true matches) | 0.9967 to 0.99999 (median 0.9996; rule: at least 0.95) |
| Route | all 565 accepted by the margin rule; the NCC tie-break was never needed |
| Hash kernel speed (1 worker) | 0.015 s per PNG (26 s for 1,510) |
| Match kernel speed (1 worker) | 0.017 s per JPEG to hash, plus 0.010 s per JPEG to match |
| Hamming scan at Kaggle bucket size (41,500 PNGs) | 0.005 s per JPEG |
| **Extrapolated Kaggle time** (112,120 PNGs, 30,083 JPEGs, 4 workers) | **hash about 0.12 h (7 min), match about 0.16 h (10 min)**: computing time only |

**Caveat on Kaggle time.** The figures above use this PC's speed with the files on a local disk. Kaggle's CPUs are slower, and the kernel must read about 42 GB of PNGs from `/kaggle/input`, so reading files will probably take most of the time. Keep planning for the 1 to 2 hours given in `kaggle_kernels/README.md`. The first real run records the true timings in `run_summary.json` and `match_stats.json`.

**The rest of the pipeline:**

- **Selection:**
  - 101 eligible patients gave 40 pairs on the standard rung.
  - Only 11 patients had no other labelled line, which is below 30, so the H1 substitute (2 or more lines vs exactly 1) switched on automatically. U holds 20 + 20 images plus 11 no-line images that are reported but not tested.
  - The dev pool holds 15 + 15 images, 10 of them smoke-test images.
  - The reading list holds 131 images, which is every A, B and U image. The retest list holds 30 and the practice list 10 dev-pool images.
  - The 5 excluded images that were in the pool were removed before any selection.
- **Pod:** the export kernel's zip matched its SHA-256 manifest for all 161 PNGs. Each model wrote 429 records. The visual-token counts were 1,024 and 256, and smoke coverage was 0.9999, so both models used the first-token signal. `parse_runs.py` skipped the two smoke files and dropped no duplicates.
- **Reading:**
  - The 3 A images flagged as tracheostomy were excluded for that reason. None of them was counted as a disagreement, including the 2 read as "absent".
  - Exactly the 2 planted disagreements were counted.
  - The retest gap was measured at 49.0 hours (rule met), with no warnings.
- **Analysis and report:**
  - The analysis wrote all three tables, both figures, `hypotheses.json` and `numbers.json`. Both JSON files are strict JSON: undefined values are written as null, never NaN.
  - The report built. Four of its placeholders stayed unfilled:
    - `run.gpu_name`, because there was no GPU here;
    - `parse.handcheck_agree`, a number the student supplies;
    - `u.line.n`, because the report text still assumes the line/no-line strata. With the substitute these are `u.ge2.n`, `u.eq1.n` and `u.noline.n`.

The fake models' answers are synthetic, so every model result in `runs/e2e_real/results` is meaningless. The test checks the plumbing, not the science.

**What this test cannot show:** what RANZCR really did to its JPEGs beyond resizing (if it cropped or changed contrast, real distances will be larger; the calibration plot and the 60-pair spot check exist for that). It also cannot show the Kaggle runtime itself, Kaggle's folder layout, the 4-worker process pool, or the real models.

## How to rerun it

From the repository folder in PowerShell (it takes about 2 to 3 minutes; the input folder is reused when nothing changed):

```powershell
.\.venv\Scripts\python.exe scripts\e2e_real_nih.py            # add --fresh to rebuild the fake Kaggle folder
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider -q tests\test_e2e_real_nih.py
```

The full summary is written to `runs/e2e_real/e2e_summary.json`, and the kernel logs to `runs/e2e_real/kaggle/*.log`. The test is skipped if `data/nih_box` or `data/nih/Data_Entry_2017_v2020.csv` is missing. Data: NIH ChestX-ray14 (NIH Clinical Center; Wang et al., CVPR 2017; the NIH Box release).
