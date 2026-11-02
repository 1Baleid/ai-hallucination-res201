# PIPELINE SPEC: Stage 1 code contract (v1.0, 2026-11-02)

This file is the contract every piece of Stage 1 code follows. `PROTOCOL.md` (v0.3) defines the science; this file defines the software. Where they disagree, `PROTOCOL.md` wins and the disagreement is reported. `cxrdev/constants.py` and `cxrdev/prompts.json` are fixed inputs written by the lead; do not edit them (report a needed change instead).

## 0. Ground rules for every builder

- Python 3.13 in the project venv: `"/c/Belly'sProjects/Ai Hallucination - RES 299/.venv/Scripts/python.exe"` (bash) or `& ".\.venv\Scripts\python.exe"` (PowerShell, from the repo root). Run tests with `<venv-python> -m pytest -q tests/<file>.py`. Do not install packages; if you need one, say so in your report.
- Installed locally: numpy, scipy, pandas, matplotlib, pillow, imagehash, scikit-learn, pytest, tqdm, kaggle, huggingface_hub, and a CPU stack (torch, torchvision, transformers >= 4.57, accelerate). Library code under `cxrdev/` may use only stdlib, numpy, scipy, pandas, matplotlib, pillow. Tests may also use scikit-learn and imagehash as cross-checks.
- Packages `pod/`, `tools/`, `scripts/`, `kaggle_kernels/` have empty `__init__.py` files (lead-owned). Scripts run as `python scripts/x.py` must insert the repo root into `sys.path` at the top. `cxrdev/prompts.py`, `cxrdev/parse.py`, `cxrdev/logits.py` and `cxrdev/constants.py` must import only the standard library and numpy (they run on the pod, which may lack pandas and scipy).
- Touch only the files your task lists. Never edit another builder's files, `PROTOCOL.md`, `STAGE1_PLAN.md`, `STRATEGY.md`, `HANDOFF.md`, `PROGRESS.md`, `drafts/`, `research/`. Never run git commit/push.
- No file or directory name may contain `phash` or `mapping` (the `.gitignore` hides `*phash*` and `*mapping*.csv/json`). No code under `data/`, `items/`, `runs/`, `outputs/`, `artifacts/`, `models/` (all git-ignored working directories).
- Data boundaries (PROTOCOL section 9): RANZCR files (train.csv, JPEGs, polylines, StudyInstanceUIDs, anything derived row-by-row from them) live only on Kaggle and in `data/` / `items/` on the student's machine. Nothing RANZCR-derived goes to the pod (`pod_items.csv` carries opaque ids and NIH filenames only), to `results/` (aggregates only), to git, or to any public place. Kaggle kernels and datasets are always private.
- Determinism: every random choice takes an explicit seed from `constants.derive_seed(purpose)`; same inputs and seed give byte-identical outputs.
- Wording in any text, docstring or report: "RANZCR CLiP labels (43 trained labellers: radiologists, radiology trainees, hospital medical officers)", never "radiologist labels"; "within-patient contrast", never "counterfactual"; never "verbatim" for the Kalai prompt (say "t substituted as a percentage"); never "label-clean" or "unseen"; "to our knowledge" on every first; "not for clinical use" on any edited image.
- Type hints and docstrings on public functions. Pure functions where possible. Library code never touches the network.

## 1. Repository layout

```
cxrdev/                 importable package (stdlib + numpy/scipy/pandas/matplotlib/pillow)
  __init__.py           empty (keep it empty: the pod and kernels import submodules directly)
  constants.py          FIXED (lead)
  prompts.json          FIXED (lead)
  prompts.py            prompt loading, text, sha256                      [core-lib]
  parse.py              two-stage answer parser                           [core-lib]
  logits.py             first-token signal helpers (numpy only)           [core-lib]
  stats.py              CIs, clustered BCa bootstrap, tests               [core-lib]
  fileio.py             CSV/JSONL helpers, sha256, schema checks          [core-lib]
  imghash.py            perceptual hashes, thumbnails, NCC                [kaggle]
  matching.py           RANZCR->NIH image matching                        [kaggle]
  ranzcr.py             train.csv label presence, yield table             [kaggle]
  nih.py                NIH metadata loading                               [kaggle]
  items.py              item selection and derived lists                  [items]
  metrics.py            per model x prompt x arm metrics, H0 gate          [analysis]
  hypotheses.py         H1, H3, H5 (and PLUS H2) primary tests            [analysis]
  arm1.py               detector-gated answers and controls               [analysis]
  tables.py             Tables 1-3 builders                               [analysis]
  figures.py            Figure 1 (and PLUS Figure 2)                       [analysis]
kaggle_kernels/         kernel sources, bundler, orchestration helpers    [kaggle] (not `kaggle/`: that name would shadow the installed kaggle package)
pod/                    RunPod runner, MedSigLIP scorer, setup, queue     [pod]
tools/                  read_tool.py (tkinter) + reading_core.py           [read-tool]
scripts/                select_items.py [items]; parse_runs.py, verify_join.py, analyze.py, synthetic_dryrun.py [analysis]
tests/                  test_<module>.py per builder
report/sections/        Markdown sections; report/references.bib          [writers]
docs/                   this spec, RUNBOOK, HUMAN_STEPS, ETT reference sheet
results/                committed aggregate outputs (tables, figures, hypotheses.json, numbers.json)
data/ items/ runs/      git-ignored working directories (created at run time)
```

## 2. Data schemas (column names are exact)

S1 `presence` (from `ranzcr.presence(train_df)`), one row per RANZCR image:
`uid` (str), `ranzcr_patient_id` (str), `ett`, `ngt`, `ngt_incomplete`, `cvc`, `swan` (int 0/1; class present = any of its columns is 1), `n_other_lines` (int = ngt + cvc + swan).

S2 `nih_meta` (from `nih.load_nih_meta(path)`), one row per NIH image:
`nih_filename` ("00000001_000.png"), `nih_patient_id` (int), `followup` (int), `age` (int), `sex` ("M"/"F"), `view` ("AP"/"PA"), `orig_w`, `orig_h` (int), `finding_labels` (str). Must accept both the Kaggle CSV whose header splits `OriginalImage[Width, Height]` into two columns (`OriginalImage[Width`, `Height]`) and the v2020 file.

S3 `match_table` (from `matching.match_all`), one row per RANZCR image:
`uid`, `ranzcr_patient_id`, `jpg_w`, `jpg_h`, `nih_filename` (str or ""), `nih_patient_id` (int or -1), `d1`, `d2` (Hamming bits of best and second-best candidate; d2 = -1 if only one candidate), `ncc1`, `ncc2` (float or NaN), `n_candidates` (int), `key_used` ("dim" | "global"), `route` ("margin" | "ncc_tiebreak" | "none"), `status` ("matched" | "ambiguous" | "unmatched" | "duplicate_conflict" | "patient_conflict").

S4 `master` (from `items.build_master_table(match_table, presence, nih_meta)`), one row per image with status "matched":
`uid`, `ranzcr_patient_id`, `nih_filename`, `nih_patient_id`, `followup`, `view`, `age`, `sex`, `ett`, `ngt`, `ngt_incomplete`, `cvc`, `swan`, `n_other_lines`.

S5 `items.csv` (local only), one row per selected image:
`item_id` (10-char lowercase hex, unique, unrelated to order), `uid`, `nih_filename`, `nih_patient_id`, `arm` ("A" | "B" | "U" | "DEV"), `pair_id` ("p001".. for A/B, "" otherwise), `stratum` ("" for A/B; U: "line" | "noline" or, under the H1 substitute, "ge2" | "eq1"; DEV: "pos" | "neg"), `ett`, `ngt`, `ngt_incomplete`, `cvc`, `swan`, `n_other_lines`, `view`, `followup`, `age`, `sex`, `direction` ("B_after_A" | "B_before_A" for A/B, "" otherwise), `gap` (abs follow-up difference for A/B, -1 otherwise), `rung` ("standard" | "two_per_patient" | "n80"), `smoke` (1 for the 20 smoke DEV items, else 0).

S6 `pod_items.csv` (the only item file that goes to the pod): `item_id`, `nih_filename`, `prompt_set` ("full" for A/B/U = P0,P1,P2hi; "p0" for DEV = P0 only), `smoke` (0/1). No arm, label, uid or patient columns.

S7 run JSONL record (pod output; one line per item x prompt x condition):
`run_id`, `model_key`, `model_id`, `revision`, `transformers_version`, `torch_version`, `dtype`, `device`, `item_id` ("BLIND-TEXT" / "BLIND-GREY" for blind rows), `nih_filename` ("" for blind rows), `image_sha256` (sha256 of the PNG file bytes; of the generated grey PNG bytes for BLIND-GREY; "" for BLIND-TEXT), `condition` ("image" | "BLIND-text" | "BLIND-grey" | "BLIND-demo"), `prompt_id`, `prompt_sha256`, `prompt_text`, `visual_tokens` (int; 0 for BLIND-text), `seed`, `max_new_tokens`, `raw_output`, `n_generated_tokens`, `ended_with_eos` (bool), `parsed`, `parser_version`, `logp_yes`, `logp_no`, `logp_abstain`, `z`, `coverage`, `signal` ("first_token" | "full_sequence"), `yes_ids`, `no_ids` (lists of int), `wall_s`, `timestamp_utc` (ISO 8601, "Z").

S8 `parsed_results.csv` (from `scripts/parse_runs.py`): all S7 fields except the id lists, with `parsed` recomputed by the current parser, `parser_version` updated, and `truncated` (bool) added.

S9 `detector_scores.csv` (pod): `item_id`, `nih_filename`, `s` (MedSigLIP P(with ETT) from the softmax over the two prompts), `logit_with`, `logit_without`, `model_id`, `revision`.

S10 reading lists (local): `reading_list.csv` = `read_id` (8-char uppercase hex, unrelated to item_id and order), `nih_filename`, `image_path`; `reading_key.csv` = `read_id`, `item_id` (never shown by the tool); `retest_list.csv` = same columns as reading_list for the 100 retest images in a new order.

S11 `verified_items.csv` / `retest.csv` (tool output): `read_id`, `nih_filename`, `response` ("present" | "absent" | "skip"), `flag_ambiguous`, `flag_tracheostomy`, `flag_out_of_field` (0/1), `note`, `seconds` (float), `order_index` (int), `timestamp_utc`. Each has a sidecar `<name>.meta.json` = {`sha256`, `saved_utc`, `seed`, `n`, `mode` ("read" | "retest"), `tool_version`}.

S12 `exclusions.csv` (from `scripts/verify_join.py`): `item_id`, `reason` ("flag_ambiguous" | "flag_tracheostomy" | "flag_out_of_field" | "label_disagreement" | "skip").

## 3. Module interfaces

### prompts.py
- `load_prompts(path: str | Path | None = None) -> dict` (default `cxrdev/prompts.json`).
- `prompt_text(prompt_id: str, device_key: str = "ett") -> str` = template with `{device}` replaced by `devices[device_key]`.
- `prompt_sha256(text: str) -> str` (UTF-8 bytes).
- `penalty_t(prompt_id: str) -> float | None`; `PROMPT_IDS = ("P0", "P1", "P2hi", "P2lo")`.

### parse.py
- `PARSER_VERSION = "2"`.
- `normalise(text: str) -> str`: map U+2018/U+2019 to `'`, U+201C/U+201D to `"`, remove `*`, `_`, backticks, collapse whitespace, strip.
- `parse_answer(text: str) -> str` in {"yes", "no", "abstain", "unparsable"}: stage 1 `re.match(r"^\W*(yes|no)\b", s, re.I)`; stage 2 `re.search(ABSTAIN_RE, s, re.I)` with `ABSTAIN_RE = r"(cannot tell|can't tell|don't know|do not know|not sure|unsure|unknown|cannot determine|can't determine|unable to determine|not possible to (tell|determine))"`; else "unparsable".
- `is_truncated(n_generated_tokens: int, max_new_tokens: int, ended_with_eos: bool) -> bool` = (not ended_with_eos) and n_generated_tokens >= max_new_tokens.

### logits.py (numpy only)
- `log_softmax(logits: np.ndarray) -> np.ndarray` (float64, stable).
- `logsumexp(x: np.ndarray) -> float`.
- `single_token_ids(encode: Callable[[str], list[int]], forms: Sequence[str]) -> tuple[list[int], list[str]]`: unique ids of forms that encode to exactly one token (order preserved), and the skipped forms.
- `first_token_signal(logprobs: np.ndarray, yes_ids, no_ids, abstain_ids=()) -> dict` with keys `logp_yes`, `logp_no` (logsumexp over the ids), `logp_abstain` (NaN if no ids), `z = logp_yes - logp_no`, `coverage = exp(logp_yes) + exp(logp_no)`.
- `p_yes_normalised(z: float) -> float` = 1 / (1 + exp(-z)).

### stats.py
- `ALPHA_GRID`: ascending array = 0.001..0.009 step 0.001, then 0.01..0.50 step 0.005 (rounded to 4 decimals, unique).
- `wilson_ci(k: int, n: int, alpha=0.05) -> tuple[float, float]` ((nan, nan) when n == 0); `wilson_halfwidth(p, n, alpha=0.05) -> float`.
- `cluster_bootstrap(df, cluster_col, stat_fn, n_boot=9999, seed=0, strata_col=None) -> np.ndarray`: resample whole clusters with replacement (within each stratum when `strata_col` is given, keeping the number of clusters per stratum), call `stat_fn(resampled_df)`; must run 9,999 resamples of a 400-row frame in under about 30 s (precompute row-index arrays per cluster; use `df.iloc`).
- `cluster_jackknife(df, cluster_col, stat_fn) -> np.ndarray` (leave one cluster out).
- `bca_interval(theta_hat, boot, jack, alpha=0.05) -> tuple[float, float, str]`: BCa with z0 = Phi^-1((#{boot < theta} + 0.5 #{boot == theta}) / B) and jackknife acceleration; falls back to the percentile interval with method "percentile" when z0 is infinite, all replicates are equal, or the acceleration denominator is non-positive; method "bca" otherwise.
- `ci_inversion_p(theta_hat, boot, jack, null=0.0, alpha_grid=ALPHA_GRID) -> tuple[float, str]`: smallest alpha in the grid at which the (1 - alpha) interval from `bca_interval` excludes `null`; returns (0.001, "<=0.001") if excluded at the first grid point, (1.0, ">0.5") if never excluded, else (alpha, "").
- `clustered_ci(df, cluster_col, stat_fn, n_boot=9999, seed=0, alpha=0.05, null=0.0, strata_col=None) -> dict` with keys `estimate`, `lo`, `hi`, `method`, `p_value`, `p_flag`, `n_clusters`, `n_rows`, `n_boot`.
- `mcnemar_exact(b: int, c: int) -> float` = min(1, 2 * BinomCDF(min(b, c); b + c, 0.5)); 1.0 when b + c == 0.
- `holm(pvals: Sequence[float]) -> list[float]` (step-down, monotone, capped at 1; NaN stays NaN and is excluded from m).
- `cohen_kappa(a: Sequence, b: Sequence) -> float` (NaN when expected agreement is 1).
- `auroc(pos_scores, neg_scores) -> float` (Mann-Whitney, ties 0.5).
- `design_effect(y, clusters) -> dict` with `icc` (one-way ANOVA estimator, unequal cluster sizes), `mean_cluster_size`, `deff = 1 + (mean_cluster_size - 1) * icc`, `n_clusters`.
- `power_two_proportions(p1, p2, n1, n2, alpha=0.05) -> float` (normal approximation, pooled-variance critical value; must give 0.611 for 0.50 vs 0.70 at 60/60, 0.816 for 0.50 vs 0.75 at 60/60, 0.828 for 0.50 vs 0.70 at 100/100, each to 3 decimals).

### fileio.py
- `sha256_file(path) -> str`; `sha256_bytes(b: bytes) -> str`.
- `read_jsonl(path) -> list[dict]`; `append_jsonl(path, record: dict) -> None` (one line, UTF-8, flush); `done_keys(path, key_fields: Sequence[str]) -> set[tuple]` (for resume).
- `write_csv(df, path)` (UTF-8, no index, `\n` line endings, stable column order) and `require_columns(df, columns: Sequence[str], name: str) -> None` (raises ValueError naming the missing columns).
- `utc_now() -> str` (ISO 8601 with "Z").

### imghash.py (kernel-side; numpy, scipy, pillow only)
- `load_gray(path_or_image, size: int | None = None, draft: bool = False) -> np.ndarray` (uint8; JPEG `draft("L", ...)` decode when `draft=True`).
- `phash_bits(img: np.ndarray, hash_size=16, highfreq_factor=4) -> np.ndarray[bool]` (DCT-II of the (hash_size*highfreq_factor)^2 grayscale resize, top-left hash_size x hash_size block, median threshold, same algorithm as `imagehash.phash`).
- `dhash_bits(img, hash_size=16) -> np.ndarray[bool]`.
- `bits_to_hex(bits) -> str`, `hex_to_bits(hex_str, n_bits) -> np.ndarray[bool]`, `pack(bit_rows) -> np.ndarray[uint8]` (n x n_bits/8).
- `hamming(q_packed: np.ndarray, db_packed: np.ndarray, chunk: int = 4096) -> np.ndarray[int]` (popcount via a 256-entry lookup table; shape n_q x n_db or 1-D for one query).
- `thumb(img, side=128) -> np.ndarray[float32]`; `ncc(a, b) -> float` (zero-mean normalised cross-correlation in [-1, 1]).

### matching.py (kernel-side)
- `@dataclass MatchConfig`: `margin_bits: int = 16`, `cut: int | None = None` (None = automatic Otsu valley on the best-distance histogram, capped at 64 of 256 bits), `ncc_min: float = 0.95`, `ncc_margin: float = 0.02`, `topk_ncc: int = 5`, `use_dimension_key: bool = True`.
- `candidate_index(nih_meta) -> dict[tuple[int, int], np.ndarray]` (orig size -> row indices).
- `auto_cut(best_distances: np.ndarray, n_bits=256) -> int`.
- `match_all(ranzcr_hashes: pd.DataFrame, nih_hashes: pd.DataFrame, nih_meta: pd.DataFrame, config: MatchConfig, ncc_fn: Callable[[str, str], float] | None = None) -> tuple[pd.DataFrame, dict]` returning S3 and a stats dict. `ranzcr_hashes` has `uid`, `ranzcr_patient_id`, `jpg_w`, `jpg_h`, `ph` (hex); `nih_hashes` has `nih_filename`, `ph` (hex). Rule: candidates = NIH images whose (orig_w, orig_h) equals (jpg_w, jpg_h) when the key is on and the bucket is non-empty, else all NIH images ("global"); accept by margin when d1 <= cut and d2 - d1 >= margin_bits and (ncc_fn is None or ncc1 >= ncc_min); otherwise, when ncc_fn is given and d1 <= cut, compute NCC for the top-k candidates and accept the best by NCC when ncc_best >= ncc_min and ncc_best - ncc_second >= ncc_margin (route "ncc_tiebreak"); otherwise "ambiguous" (d1 <= cut) or "unmatched" (d1 > cut).
- `resolve_one_to_one(table) -> table`: when several uids claim one NIH file, keep the smallest d1 (then highest ncc1) and mark the rest "duplicate_conflict".
- `patient_consistency(table) -> table`: each ranzcr_patient_id must map to one nih_patient_id; matched rows that disagree with the patient's majority NIH patient become "patient_conflict".
- `calibration_report(table) -> dict` (histograms of d1 and d2 - d1, counts per status and route, match rate, ambiguous share within patient, cut used).

### ranzcr.py / nih.py (kernel-side)
- `ranzcr.presence(train_df) -> S1 frame`; `ranzcr.yield_table(presence_df) -> pd.DataFrame` with columns `metric`, `value` covering at least: n_images, n_patients, ett_present_images, ett_absent_images, patients_with_both_ett_states, ett_absent_line_images, ett_absent_line_patients, ett_absent_noline_images, ett_absent_noline_patients, ett_absent_ge2_patients, ett_absent_eq1_patients, ngt_incomplete_images.
- `nih.load_nih_meta(path) -> S2 frame`; `nih.locate_images(root: Path) -> dict[str, Path]` (filename -> path, scanning `images_*/images/*.png` and flat layouts).

### items.py
- `build_master_table(match_table, presence, nih_meta) -> S4 frame`.
- `select_core(master, seed=SEED, n_pairs=120, n_u_per_stratum=60, n_dev_pos=150, n_dev_neg=150) -> tuple[pd.DataFrame, dict]` returns S5 items (without smoke flags set yet) and a selection report. Rules: (1) pairs: eligible NIH patients have >= 1 ETT-present and >= 1 ETT-absent image; per selected patient choose A uniformly at random among ETT-present images, then B = the ETT-absent image of the same patient minimising |followup(B) - followup(A)| (ties: random); select n_pairs patients at random; if fewer than n_pairs eligible patients exist, rung "two_per_patient" (a second, disjoint A/B pair from patients with >= 2 present and >= 2 absent images), then rung "n80" if still short of 120 but at least 80; fewer than 80 raises. (2) U: patients outside pairs; ETT-absent images, one per patient; stratum "line" = n_other_lines >= 1, "noline" = n_other_lines == 0; if fewer than H1_SUBSTITUTE_MIN_PATIENTS noline patients exist, use "ge2" (>= 2 other lines) vs "eq1" (exactly 1) and record `h1_substitute = True`; view matching: both strata get the same number of AP and PA images (target AP share = the B arm's AP share, rounded, limited by availability); if a stratum cannot supply n_u_per_stratum, take all and record it. (3) DEV: patients outside pairs and U; one image per patient; 150 ETT-present ("pos") and 150 ETT-absent ("neg"). All patient sets disjoint by nih_patient_id.
- `assign_item_ids(items, seed) -> items` (10-char hex from a seeded RNG, unique).
- `mark_smoke(items, n=20, seed) -> items` (random DEV items, balanced 10 pos / 10 neg).
- `pod_items(items) -> S6 frame`; `nih_filelist(items) -> pd.DataFrame` with the single column `nih_filename`.
- `reading_lists(items, image_dir, n_a=30, seed) -> tuple[reading_list, reading_key]` (all B, all U, n_a random A; one random order; read_ids 8-char uppercase hex).
- `retest_list(reading_list, n=100, seed) -> pd.DataFrame` (random subset, new random order, same read_ids).
- `selection_report(...)` keys: counts per arm and stratum, rung, h1_substitute, view shares, follow-up gap summary, direction counts, n eligible patients, sha256 of every written file.

### metrics.py
Definitions per model x prompt over parsable items (parsed in {"yes","no","abstain"}); unparsable and truncated items are excluded from denominators and reported as rates:
- `hit_A = yes_A / n_A`; `FAR_B = yes_B / n_B`; `FAR_U = yes_U / n_U`; `ba = (yes_A / n_A + no_B / n_B) / 2`; `yes_rate = (yes_A + yes_B + yes_U) / (n_A + n_B + n_U)`; `abstention = (abstain_A + abstain_B + abstain_U) / (n_A + n_B + n_U)`; `ER_t = (correct - c * wrong) / n` over A, B and U with c = t / (1 - t), correct = yes on A or no on B/U, wrong = no on A or yes on B/U, abstain scores 0; `persistence = P(yes on B | yes on A)` over pairs with A parsed yes and B parsable; `pair_correct = P(yes on A and no on B)` over pairs with both parsable.
- `summarise(df, model_key, prompt_id, exclusions=None, unparsable_as_wrong=False) -> dict` with each rate, its n, its Wilson CI, unparsable_rate, truncated_rate, ER_0.9 (and ER_0.5) with a patient-clustered BCa CI, AUROC(A vs B) and AUROC(A vs U) of z (P0 only) with clustered BCa CIs, and the H0 class (P0 only).
- `h0_gate(ba, hit_a, yes_rate) -> str` in {"NO-GO", "GO", "CONDITIONAL"} using `constants.GATE` in the PROTOCOL order.
- `blind_rows(df, model_key) -> pd.DataFrame`: per prompt and condition (BLIND-text, BLIND-grey): `p_yes_norm = sigmoid(z)`, `coverage`, `greedy` (parsed answer); no interval.

### hypotheses.py
- `h1(df, model_key, n_boot, seed) -> dict`: under P0 on U; estimate = yes-rate(line or ge2) - yes-rate(noline or eq1); `clustered_ci(..., cluster_col="nih_patient_id", strata_col="stratum")`.
- `h3(df, model_key, rung) -> dict`: over B and U items parsable under both P1 and P2hi; abstain indicator; b = abstain under P2hi only, c = abstain under P1 only; estimate = (b - c) / n; p = `mcnemar_exact(b, c)`; under rung "two_per_patient" the primary is the clustered BCa CI of the paired difference with CI-inversion p, and McNemar on one random B per patient is a check; report n dropped.
- `h5(arm1_result) -> dict` (from arm1: one-sided shuffled-expert permutation p, real gain >= permuted gains; clustered CI descriptive).
- `holm_family(results: list[dict]) -> list[dict]` adds `p_holm`; a family = the per-model primaries of one hypothesis.
- PLUS `h2(df, model_a, model_b, n_boot, seed) -> dict` (paired differences in 1 - FAR on B and U and in BA on A and B, clustered by patient).

### arm1.py
- `expert_logodds(s, eps=1e-6)`; `fuse(z, s, beta, gamma=5.0)` = z + beta * clip(logodds(s), -gamma, gamma); yes iff fused > 0.
- `balanced_accuracy(yes: np.ndarray[bool], y: np.ndarray[int]) -> float`.
- `youden_tau(s_dev, y_dev) -> float` (midpoint threshold maximising TPR - FPR).
- `select_beta_dev(z_dev, s_dev, y_dev, grid=BETA_GRID, gamma=GAMMA) -> float` (ties -> closest to 1).
- `shuffled_expert_test(z, s, y, beta=1.0, gamma=5.0, n_perm=9999, seed=...) -> dict`: observed gain = BA(fused) - BA(z > 0); null = gains with s permuted across the same images; p = (1 + #{null >= observed}) / (1 + n_perm).
- `constant_bias_curve(z, y, b_grid=np.linspace(-10, 10, 401)) -> pd.DataFrame` (b, yes_rate, ba) and `matched_bias(z, y, target_yes_rate, b_grid) -> dict` (b*, yes_rate, ba; ties -> smallest |b|).
- `disagreement_abstain(z, s, y, tau) -> dict` (coverage, risk among answered, n).
- `oracle_rows(z, y, gamma) -> dict` (oracle label BA = 1.0 reference; clipped-oracle BA with s = y; share of |z| > gamma).
- `run_arm1(test_df, dev_df, model_key, n_perm, n_boot, seed) -> dict`: rows greedy-text, beta0, fused (beta 1), dev-selected beta, detector-alone, shuffled-expert null summary, matched constant bias, disagreement-abstain (and PLUS oracle rows); each row has BA, hit_A, FAR_B, FAR_U (U scored with the same rule), yes-rate, n; plus the greedy-text vs beta0 agreement share.

### tables.py / figures.py
- `table1(match_stats, yield_df, selection_report, verification_summary, kappa, detector_agreement) -> pd.DataFrame`; `table2(summaries, blind) -> pd.DataFrame`; `table3(arm1_results) -> pd.DataFrame`; `write_table(df, stem, outdir)` writes `<stem>.csv`, `<stem>.md` and `<stem>.tex`. Cells format rates as `0.62 [0.53, 0.70] (n=118)`.
- `figure1(yes_rates: pd.DataFrame, blind: pd.DataFrame, out_stem, example_pair: dict | None = None)`: panel a = example A/B pair from two NIH PNG paths with student-provided arrow coordinates (placeholder text when None; never polylines); panel b = yes-rate per arm (A, B, U-line, U-noline) per model and prompt with Wilson error bars, BLIND rows as dashed reference lines; writes `<out_stem>.pdf` and `.png`.

### scripts
- `scripts/select_items.py --match data/match_table.csv --train data/ranzcr/train.csv --nih data/nih/Data_Entry_2017.csv --images-dir data/images --out items/` -> items.csv, pod_items.csv, nih_filelist.csv, reading_list.csv, reading_key.csv, retest_list.csv, selection_report.json.
- `scripts/parse_runs.py --runs runs/<run_id> --out runs/parsed_results.csv`.
- `scripts/verify_join.py --items items/items.csv --key items/reading_key.csv --verified items/verified_items.csv [--retest items/retest.csv] --out items/exclusions.csv --summary results/verification_summary.json`.
- `scripts/analyze.py --items ... --parsed ... --detector ... --exclusions ... --match-stats ... --yield ... --selection ... --verification ... --out results/` -> table1-3, figure1, hypotheses.json, numbers.json (every number the report may quote, keyed by stable names).
- `scripts/synthetic_dryrun.py --out runs/synthetic/` builds synthetic S5/S8/S9/S11 data with known effects and runs the whole analysis; used as an end-to-end test.

## 4. Kaggle side

Three private CPU kernels, internet off, GPU off (no phone verification needed): `<user>/res201-hash` (yield table from train.csv, dimension-key test on 200 JPEGs, NIH perceptual hashes per folder written as `nih_hashes_<folder>.csv` with try/except per folder and exit code 0), `<user>/res201-match` (inputs: competition, NIH dataset, res201-hash output; RANZCR hashes computed in-session under `/kaggle/temp` and never written to `/kaggle/working`; outputs `match_table.csv`, `match_stats.json`, `calibration.png`, `spotcheck_XX.png` grids of 10 random matched pairs each, 60 in total, JPEG left and PNG right at 256 px), `<user>/res201-export` (inputs: NIH dataset and the private dataset `<user>/res201-nih-filelist` holding `nih_filelist.csv`; output `selected_nih_png.zip` plus `manifest.csv` with sha256). Kernel sources live in `kaggle_kernels/<name>_kernel/kernel_main.py`; `kaggle_kernels/build_kernels.py --username <u>` bundles the needed `cxrdev` modules (strips their `from cxrdev...` imports) and the main into `kaggle_kernels/build/<slug>/kernel.py` with `kernel-metadata.json` (`is_private: true`, `enable_internet: false`, `enable_gpu: false`, the right `competition_sources`, `dataset_sources`, `kernel_sources`). Input paths are discovered by globbing under `/kaggle/input` (do not hard-code mount names). `kaggle_kernels/run_kernel.py` pushes, polls status and downloads outputs into `data/kaggle/<slug>/`; `kaggle_kernels/make_filelist_dataset.py` creates or versions the private filename-list dataset.

## 5. Pod side

`pod/run_vlm.py --model-key <key> --items pod_items.csv --images-dir <dir> --out <jsonl> --run-id <id> [--smoke-only] [--blind] [--limit N] [--signal auto|first_token|full_sequence]` loads the model in bf16 (`AutoModelForImageTextToText`, `AutoProcessor`), builds chat messages with the model's own template and no system prompt, generates greedily (`do_sample=False`, `repetition_penalty=1.0`, `max_new_tokens=16`, `output_logits=True`, `return_dict_in_generate=True`), takes the raw first-step logits (never the processed scores), computes the S7 record with `cxrdev.logits` and `cxrdev.parse`, asserts the visual-token count (`image_grid_thw` for Qwen families: 1,024 for the hero at 1024x1024 with min = max pixels = 1,048,576; 256 image tokens for MedGemma), resumes from existing JSONL keys (item_id, prompt_id, condition), and writes one line per record. Smoke mode runs the 20 smoke items under P0, checks median coverage >= 0.90 and writes `smoke_report.json`; when coverage fails, `--signal auto` switches to the full-sequence score (teacher-forced log-likelihood of "Yes" vs "No"). `pod/run_medsiglip.py` writes S9. `pod/setup_pod.sh`, `pod/queue.sh` (smoke, then full runs, then MedSigLIP, with logs and a done marker), `pod/requirements-pod.txt`, `pod/README.md` (deploy, upload and download over SSH on exposed TCP, stop the pod).

## 6. Report side

Markdown sections in `report/sections/` with pandoc-style citations `[@key]`; keys are `firstauthorYEARfirstword` in lowercase (for example `kalai2026evaluating`, `whitehead2022reliable`, `tang2021clip`); numbers that come from results are written as `[[NUM:key]]` placeholders and facts the student must supply as `[[TODO:...]]`. `report/references.bib` holds the BibTeX entries.

## 7. Changes after the build (2026-11-02, before any study model output)

These supersede the text above where they differ (PROTOCOL v0.4 and its section 11 entries are the authority).
- Reading list (S10): every A image is read (`constants.N_READ_A` = 1,000,000; callers take min with the number of A), so the list is all A + B + U (360 at the standard n). `select_items.py` also writes `reading_list.meta.json` and `retest_list.meta.json` (the seeds) and `practice_list.csv` (10 dev-pool images, 5 with ETT and 5 without, own read ids, never a test image; seed master x 100 + 50).
- Exclusions (S12): skips and flags are applied before the label-disagreement check; an A image flagged "tracheostomy" is excluded with reason `flag_tracheostomy` (RANZCR labels tracheostomy tubes as ETT). Disagreements, the re-match list, agreement and kappa use answered, unflagged images; `agreement_all_read` keeps the old all-images figure. `verify_join.py` records the retest gap from the first read's `saved_utc` and adds a warning when it is under 48 h.
- Item selection: `select_items.py --exclude-filenames PATH` removes listed NIH images (one per line, `#` comments) before any selection; the real run passes `results/probes/probe_seen_filenames.txt` (11 images with model or detector outputs from software tests). The selection report records the file's sha256, `n_listed` and `n_present_in_matched_pool`, never the names. `select_items.py` reads train.csv with `ranzcr.read_train_csv` (ids stay strings).
- H1: the contrast comes from the selection report; under the substitute (active on the real data: 23 no-line patients) it is U(ge2) vs U(eq1), and no-line items are descriptive only (`n_descriptive`). H1 is a test under every rung, including n80. H3 uses the clustered analysis whenever any patient contributes two pairs (counted from pair ids).
- S9 gains a last column `dtype`; analyze writes `run.medsiglip.dtype`. `analyze.py --run-dir` reads `run.gpu_name` from the run folder; `match.dimkey_share_found` and `match.dimkey_used` come from `match_stats.json`.
- `numbers.json`, `hypotheses.json`, `metrics.json` and `verification_summary.json` are strict JSON with null for undefined values; `build_report.py` renders null as "n/a".
- `parse_runs.py` skips `smoke/*.smoke.jsonl` and `*.prev.jsonl` in a run folder unless `--include-smoke` is given.
- End-to-end test on real NIH images: `scripts/e2e_real_nih.py`, `tests/test_e2e_real_nih.py` (slow), write-up in `docs/E2E_REAL_NIH.md`.
- OSF: not used (student decision); "registered" in code comments means "fixed in the protocol before any model output", marked by the dated lock-in commit.
