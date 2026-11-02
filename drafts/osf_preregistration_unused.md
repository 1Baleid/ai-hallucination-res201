> **Not used.** The student decided on 2026-11-02 to skip the OSF preregistration; the protocol is frozen by a dated git commit instead (PROTOCOL.md section 11). Kept only in case that decision changes.

# OSF Preregistration: LongiDev-CXR Stage 1

Template: OSF Preregistration. This text is taken from the frozen protocol (`PROTOCOL.md` v0.4, Sections 2 to 8, including the change-log entries of 2026-11-02: the one-sided H5 permutation p-value; every A image read and tracheostomy-only A images excluded; flag exclusions before label-disagreement checks; the 256-bit perceptual hash and the dimension-key rule; six yes and six no surface forms; one primary and one secondary endpoint for PLUS H2; the clustered H3 analysis whenever any patient contributes two pairs; U kept at 60 + 60 under every rung). Before submission, fill every yield-table placeholder (NUM keys that begin with `yield.`, each named after a metric in the Day-1 `yield_table.csv`) and every TODO item, and delete this paragraph. References are given in plain text because the OSF form does not render citation keys.

## Study information

### Title

LongiDev-CXR Stage 1: device-existence hallucination of small open vision-language models on same-patient chest radiograph pairs, with penalised abstention and detector-gated answers

### Authors

Abdullah Baleid

### Description

Small open vision-language models are often asked whether a device is present on a chest radiograph. This study tests whether two such models answer the question "Is an endotracheal tube present in this chest X-ray?" from the image or from a prior that intensive-care radiographs carry tubes. The images are public NIH ChestX-ray14 radiographs (NIH Clinical Center; Wang et al., CVPR 2017). The labels are RANZCR CLiP labels (43 trained labellers: radiologists, radiology trainees, hospital medical officers), described by Tang et al. (Scientific Data 2021) and linked to the NIH images by image content.

The benchmark has three arms. A is a radiograph with an endotracheal tube (ETT). B is a radiograph of the same patient without one, so each A/B pair is a within-patient contrast. U is an ETT-absent radiograph from another patient, split by whether another line or tube is labelled. Each test image receives three prompts: a binary question (P0), the same question with an "I don't know" option (P1), and P1 preceded by a stated penalty for wrong answers (P2hi). Two blind conditions (no image; a uniform grey image) measure the answer prior. A detector-gated rule (Arm 1) adds a clipped zero-shot detector log-odds to the model's first-token yes/no log-odds and is compared with controls. The CORE models are Qwen3-VL-2B-Instruct (general) and MedGemma 1.5 4B (medical). No model output had been generated or seen at the time of registration.

### Hypotheses

One primary test per hypothesis and model. A Holm family is the per-model primaries of one hypothesis over the two CORE models.

- **H0, floor gate (classification, no test).** Per model, under P0, after exclusions, evaluated in this order: NO-GO if balanced accuracy (BA) < 0.60, or yes-rate < 0.10 or > 0.90 (a constant-answer policy); otherwise GO if BA >= 0.70 and hit_A >= 0.70 and 0.25 <= yes-rate <= 0.75; otherwise CONDITIONAL. Point estimates are reported with intervals beside them, and the class is reported as a finding either way.
- **H1, co-occurrence prior.** Prediction: under P0, the yes-rate is higher on U images with at least one other labelled line (NGT, CVC or Swan-Ganz) than on U images with none. Primary test: the difference in yes-rate (line minus no-line), with a patient-clustered BCa bootstrap 95% interval and a two-sided p-value by interval inversion. Registered substitute: if the no-line stratum has fewer than 30 patients, the contrast is U with two or more other labelled lines minus U with exactly one (same test), and the no-line stratum is descriptive. Secondary: FAR_B versus FAR_U, persistence, B strata, blind P(yes).
- **H3, penalty effect.** Prediction: the stated penalty raises abstention. Primary test: the exact two-sided McNemar test on the abstain indicator, P2hi versus P1, over B and U items parsable under both prompts, per model; the number of items dropped is reported. Whenever any patient contributes two pairs (the two-pairs-per-patient rung, or an 80-pair rung that needs second pairs), the primary becomes the patient-clustered BCa interval of the paired difference (p-value by interval inversion), with McNemar on one randomly selected B per patient (and every U item) as a check. Secondary: ER_0.9 versus P0 accuracy; FAR_B by prompt.
- **H5, detector-gated answers (Arm 1).** Prediction: fusing the detector score raises balanced accuracy. Primary test: the paired difference in BA on A and B between the fused rule (beta = 1) and the beta = 0 logit rule (yes if z > 0), tested by a one-sided shuffled-expert permutation test with 9,999 permutations: p = (1 + #{permuted gain >= observed gain}) / (1 + 9,999). The patient-clustered BCa interval of the gain is descriptive. Secondary: FAR_B, FAR_U, hit_A; the constant-bias curve at matched yes-rate; the greedy-text answer as a reported row.
- **H2, general versus medical (PLUS tier; tested only if the PLUS runs take place).** Primary: the CheXOne minus Qwen2.5-VL-3B-Instruct paired difference in the hallucination-free rate (1 - FAR over B and U, on items parsable under both models), under P0, two-sided, with a patient-clustered BCa interval and a p-value by interval inversion. Secondary: the paired difference in BA on A and B, with the same interval. PLUS results form their own family and never change a CORE result.
- **STRETCH (only if run; reported as registered here).** H6 retrieval (relevant versus greedy; mismatched versus greedy); H7-lite (256 versus 1,024 visual tokens, McNemar on correctness, clustered bootstrap under the two-pairs rung); H8 (gate pass rate; FAR on edited images beside FAR_B and FAR_U, intervals only).

## Design plan

### Study type

Other: a computational evaluation of pretrained models on existing, public, de-identified radiographs. There are no human participants. Prompt, image condition and decision rule vary within items.

### Blinding

The item list sent to the GPU machine holds random 10-character item ids, NIH filenames, the prompt set and a smoke-test flag; no arm, label or patient field reaches the models' side. The student's verification reading is blind to arm and to model output: every test image (all A, all B and all U) appears in one random order under read ids unrelated to item ids, with the arm hidden. The reading file is saved with its SHA-256 and a UTC timestamp before the join to arms and before any model output is opened. The smoke tests use development-pool images only, so no test-item output exists before the read.

### Is there any additional blinding in this study?

The analysis code is written and tested end to end on synthetic data with known effects before any real output is opened. The analyst is not blind to arm when the registered analysis runs.

### Study design

Within-item design: model (two CORE models) by prompt (P0, P1, P2hi) by arm (A, B, U), plus the blind conditions (BLIND-text and BLIND-grey, one deterministic generation per model and prompt) and a development pool run under P0. A and B form same-patient pairs (one pair per patient). U is stratified for H1 and matched on view position (AP/PA). Arm 1 applies decision rules to the cached P0 first-token log-odds of the same items, with controls on the same arrays. Arm 1 needs no extra model generations beyond the development-pool P0 run.

### Randomization

There is no random assignment of a treatment; prompts and conditions are applied to every item. Every random choice uses a fixed seed derived from master seed 20261012 as seed = master x 100 + purpose offset: `pairs` 2026101201; `u` 2026101202; `dev` 2026101203; `ids` 2026101204; `reading` 2026101205; `retest` 2026101206; `smoke` 2026101207; `bootstrap` 2026101208; `permutation` 2026101209; `spotcheck` 2026101210; `model` 2026101211. Random choices: the selected pair patients, image A within a patient, ties for B, the U and development draws, item ids, the reading order and read ids, the retest subset and its order, the smoke items, bootstrap resamples, permutations and the spot-check pairs. Decoding is greedy, so generation involves no sampling.

## Sampling plan

### Existing data

Registration prior to creation of data.

### Explanation of existing data

The images (NIH ChestX-ray14) and labels (RANZCR CLiP) are public and already exist. The study's data are the model outputs, the detector scores and the student's verification reading. At registration, no model output had been generated or seen on any study item, and the verification reading had not started. Software tests before registration produced model or detector outputs on 11 public NIH images (listed in `results/probes/probe_seen_filenames.txt`); those 11 images are excluded from item selection. The only data-derived information inspected before registration was label counts from the Day-1 yield table (images and patients per ETT state, per other-line stratum and per rung condition). Those counts fixed the rung and the H1 contrast below. At registration, the matching statistics, the 60-pair spot check and the item lists did not exist yet.

### Data collection procedures

Day numbers refer to the execution plan: Day 1, yield table and hashing; Day 2, matching, item selection and this registration; Day 3, smoke tests and model runs; Day 4, the verification reading, then the join to arms and the analysis; the retest follows at least 48 h after the reading.

1. Sources and terms. NIH ChestX-ray14 PNGs (1024 x 1024) are used under the release's attribution requirements (NIH Clinical Center as data provider; Wang et al., CVPR 2017; link to the NIH download site). RANZCR CLiP labels come from the Kaggle competition `ranzcr-clip-catheter-line-classification`, whose rules permit non-commercial academic research and education and forbid copying, redistribution and re-identification. ETT present = any of the ETT normal, borderline or abnormal columns (the dataset descriptor, Tang et al. 2021, states that all tracheostomy tubes were labelled in the "ETT normal" category; see Verification); other labelled lines = NGT, CVC, Swan-Ganz; an incompletely imaged NGT counts as present and is flagged. The NIH text-mined finding labels are covariates only; ground truth is never a detector output.
2. Matching (private Kaggle notebooks; the only linkage is between two public de-identified releases of the same radiographs, and no re-identification is attempted). A dimension key restricts the candidates to NIH images whose recorded original width and height equal the JPEG's size; the key is used only if at least half of 200 sampled JPEG sizes occur among the NIH original sizes, otherwise every NIH image is a candidate (and a JPEG whose size bucket is empty is compared with every NIH image); a 256-bit perceptual hash with a margin rule (best Hamming distance within an empirically calibrated cut and at least 16 bits ahead of the second best; the cut is set at the valley of the bimodal best-distance histogram, capped at 64 bits, and reported); confirmation by normalised cross-correlation of 128-pixel thumbnails (at least 0.95; an NCC tie-break over the five closest candidates requires a lead of 0.02); one-to-one assignment (when several JPEGs claim one NIH file, the closest keeps it and the others are excluded); each RANZCR patient maps to a single NIH patient (a match that disagrees with the patient's majority NIH patient is excluded). Ambiguous images are excluded and counted with their within-patient share. A visual spot check of 60 random matched pairs is reported. The competition's first-place team already used image hashing to find RANZCR images among the NIH radiographs, so the matching is not claimed as a contribution.
3. Item selection. Pairs: eligible patients have at least one matched ETT-present and one ETT-absent image; for each selected patient, A is drawn at random from the ETT-present images and B is the ETT-absent image nearest in NIH follow-up number (ties at random); direction and gap are logged. U: other patients, one ETT-absent image each, 60 with at least one other labelled line and 60 with none under every rung; the two strata hold the same numbers of AP and of PA images (AP share set to that of B where availability allows); a stratum with fewer than 60 patients takes all of them. Development pool: 300 further images (150 ETT present, 150 absent), one per patient, used only for the detector threshold, the beta sensitivity row and the smoke tests, never for a test statistic. All patient sets are disjoint by NIH patient id.
4. Model runs. Both CORE models run in bfloat16 with greedy decoding, `max_new_tokens` 16, repetition penalty 1.0, no system prompt, the model's own chat template and a fixed seed. Qwen3-VL-2B-Instruct uses `min_pixels = max_pixels = 1,048,576` (1,024 visual tokens, asserted); MedGemma 1.5 4B uses its fixed 896-pixel input (256 image tokens, asserted). The raw first-step logits give z = logsumexp over the yes-form token ids minus logsumexp over the no-form token ids (forms below) and coverage = P(yes forms) + P(no forms). A 20-item development-pool smoke test under P0 requires a median coverage of at least 0.90 per model; otherwise z is computed from the full-sequence log-likelihoods of "Yes" and "No" (two teacher-forced passes per item), and the switch is dated in the protocol change log.
5. Detector. MedSigLIP-448 zero-shot probability s of the first of the two texts below (softmax over the pair), at 448 pixels, on all 660 images. It is never an evaluated model and never excludes, flags or selects an item.
6. Verification. The student reads every test image, all A, all B and all U (360 images in the standard design), as described under Blinding, recording present, absent or skip with flags for ambiguous, tracheostomy (the only airway tube is a tracheostomy tube) and out-of-field, using a one-page ETT reference sheet. Every A image is read because the RANZCR CLiP labels put tracheostomy tubes in the "ETT normal" category, so an ETT-present image may hold only a tracheostomy tube, while the question asks about an endotracheal tube. At least 48 h after the read ends, 100 of the images are re-read blind in a new random order (Cohen's kappa). No second reader is available; this is stated as a limitation, and the reading is reported as agreement with the published labels.
7. Parsing. Normalise curly quotes to straight quotes, remove `*`, `_` and backticks, collapse whitespace and strip. Stage 1: case-insensitive `re.match` with the first pattern below gives yes or no. Stage 2: otherwise, case-insensitive `re.search` with the abstain lexicon below gives abstain. Stage 3: otherwise unparsable. An output is truncated if it ends without an end-of-sequence token after `max_new_tokens`. One hundred outputs are checked by hand; no LLM judge is used.

```text
^\W*(yes|no)\b
```

```text
(cannot tell|can't tell|don't know|do not know|not sure|unsure|unknown|cannot determine|can't determine|unable to determine|not possible to (tell|determine))
```

Surface forms (from `cxrdev/constants.py`; only forms that encode to a single token are used, and the id lists are logged):

```python
YES_FORMS = ("Yes", "yes", " Yes", " yes", "YES", " YES")
NO_FORMS = ("No", "no", " No", " no", "NO", " NO")
```

Detector texts:

```text
a chest x-ray with an endotracheal tube
a chest x-ray without an endotracheal tube
```

### Sample size

Design: 120 A/B pairs (240 images), 120 U images (60 with two or more other labelled lines, 60 with exactly one, under the registered H1 substitute) and 300 development images (150/150): 660 images, of which 360 are test images. Per model: 360 test images x 3 prompts, 300 development images x P0, and 6 blind generations (BLIND-text and BLIND-grey x 3 prompts). Verification: all 360 test images, of which 100 are re-read.

Yield table (RANZCR labels alone, before matching; upper bounds for the matched yield): 30,083 images of 3,255 patients; 8,457 images (28.1%) with an ETT; 753 patients with both ETT states (568 with at least two of each); 3,178 patients with an ETT-absent image carrying another labelled line; 23 with an ETT-absent image carrying none; 540 with two or more other lines and 3,110 with exactly one. Central lines are labelled on 97.5% of images (counted locally on 2026-11-02 with `cxrdev.ranzcr.yield_table`).

Registered rung: standard (120 pairs, one per patient), because 753 patients are eligible. Registered H1 contrast: the substitute, U with two or more other labelled lines versus U with exactly one, because only 23 patients have an ETT-absent image with no other labelled line (fewer than 30); the few no-line images are reported descriptively.

Rungs, decided from the yield table before any model output: if fewer than 120 eligible patients exist, a second, disjoint A/B pair is taken from patients with at least two ETT-present and two ETT-absent images (analysed with patient clustering); if still short, n = 80 pairs. The U arm stays at 60 + 60 under every rung.

### Sample size rationale

The sample size was set by the student's reading time (360 images at 25 to 35 s each) and the compute budget, not by a power target. The power note states what it can detect (normal approximation, two-sided alpha = 0.05). With 120 pairs, a Wilson 95% interval at p = 0.5 has a half-width of about 0.09. Two U strata of 60 detect a difference of 0.25 with about 80% power (0.20 with about 60%). The U strata stay at 60 and 60 under every rung, so the H1 power does not change under the 80-pair rung. At n = 200 (PLUS), the half-width is about 0.07, and two strata of 100 detect 0.20 with about 83% power.

### Stopping rule

There is no optional stopping and no interim look at model outputs. Every registered item is run once per model and prompt. The item lists and their SHA-256 hashes are fixed before any model run. After the Day-3 run queue, no registered item is cut (see Other).

## Variables

### Manipulated variables

Prompt (within item). The device phrase is the endotracheal tube in every CORE item. The strings are exact; the confidence-target sentence of P2hi and P2lo is Kalai et al.'s instruction (Why Language Models Hallucinate, arXiv 2509.04664 v1, Section 4.2; version of record: Nature 2026, DOI 10.1038/s41586-026-10549-w) with t substituted as a percentage. P0 is the default template of Lotfinia et al. 2026 (arXiv 2606.17710).

P0 (CORE):

```text
Is an endotracheal tube present in this chest X-ray? Answer with a single word: Yes or No.
```

P1 (CORE):

```text
Is an endotracheal tube present in this chest X-ray? Answer with yes, no, or I don't know.
```

P2hi (CORE; t = 0.9, penalty 9):

```text
Answer only if you are > 90% confident, since mistakes are penalized 9 points, while correct answers receive 1 point, and an answer of "I don't know" receives 0 points. Is an endotracheal tube present in this chest X-ray? Answer with yes, no, or I don't know.
```

P2lo (PLUS only; t = 0.5, penalty 1):

```text
Answer only if you are > 50% confident, since mistakes are penalized 1 point, while correct answers receive 1 point, and an answer of "I don't know" receives 0 points. Is an endotracheal tube present in this chest X-ray? Answer with yes, no, or I don't know.
```

Image condition: image; BLIND-text (the prompt with no image); BLIND-grey (a 1024 x 1024 8-bit PNG with every pixel at 128); BLIND-demo (PLUS: the question preceded by a sentence giving view, age and sex from the NIH metadata).

Arm 1 decision rule on the same P0 arrays: z' = z + beta x clip(log(s / (1 - s)), -5, 5), answer yes if z' > 0; beta = 0 (baseline), beta = 1 (primary, fixed in advance), beta from {0.5, 1, 2} chosen by BA on the development pool (sensitivity; ties to the value closest to 1); detector alone (s > tau, tau at the Youden point on the development pool); shuffled expert (s permuted across the same A and B images, 9,999 permutations); constant bias (z + b for 401 values of b from -10 to 10, read where its yes-rate on A and B is closest to the fused rule's; ties to the smallest |b|); disagreement-abstain (abstain when the sign of z differs from the detector's call). PLUS: "oracle label" (answer = RANZCR label; BA = 1 by construction, reference only) and "clipped oracle" (s set to the label) rows with the share of items with |z| > 5.

### Measured variables

Per generation: raw output, parsed answer (yes, no, abstain, unparsable), truncation, number of generated tokens, z, coverage, visual-token count, time per item, model revision and library versions. Per image: MedSigLIP probability s; the student's reading (present, absent or skip; flags for ambiguous, tracheostomy and out-of-field; seconds); RANZCR labels (ETT, NGT, CVC, Swan-Ganz; ground truth); NIH metadata (patient id, follow-up number, view position, age, sex) as covariates. Per pair: direction and follow-up gap.

### Indices

Over parsable items (n counts yes, no and abstain answers): hit_A = P(yes | A); FAR_B = P(yes | B); FAR_U = P(yes | U); BA = (P(yes | A) + P(no | B)) / 2; yes-rate = yes answers over A, B and U; abstention = abstain answers over A, B and U; persistence = P(yes on B | yes on A), over pairs with A answered yes and B parsable; pair-correct = P(yes on A and no on B), over pairs with both parsable; Effective Reliability ER_t = (correct - c x wrong) / n over A, B and U with c = t / (1 - t) (c = 9 at t = 0.9, c = 1 at t = 0.5) and abstentions scoring 0 (Whitehead et al., ECCV 2022, Eq. 6, binary-accuracy case); AUROC of z for A versus B and for A versus U under P0, reported separately (over every retained item with a finite z, whatever its parse); blind P(yes) = 1 / (1 + exp(-z)) and the greedy answer per prompt; H0 class. Arm 1 rows: BA, hit_A, FAR_B, FAR_U (U scored by the same rule), yes-rate, n, and coverage with risk among answered items for disagreement-abstain; the agreement share between the greedy-text answer and the beta = 0 rule.

## Analysis plan

### Statistical models

No regression or mixed-effects model is fitted in Stage 1. Rates (hit_A, FAR, yes-rate, abstention, coverage) get Wilson 95% intervals. Balanced accuracy, ER_t, AUROC and every difference get a patient-clustered BCa bootstrap interval with 9,999 resamples (cluster = NIH patient id, so the A and B images of a pair resample together; strata kept for the U contrast), with acceleration from a cluster jackknife and the percentile interval as the documented fallback. Paired indicators on identical items use the exact McNemar test. The H5 inference is the shuffled-expert permutation test described under Hypotheses. The design effect, 1 + (mean cluster size - 1) x ICC with a one-way ANOVA ICC, is reported.

### Transformations

z is a log-odds difference of first-token probabilities and is used untransformed; blind P(yes) is its logistic transform. The detector probability s is converted to log-odds (after clipping s to [1e-6, 1 - 1e-6]) and clipped to [-5, 5] inside the fusion rule. Outcome rates are not transformed.

### Inference criteria

One primary test per hypothesis and model; alpha = 0.05. Bootstrap p-values are obtained by interval inversion: the smallest alpha on a grid from 0.001 to 0.5 at which the (1 - alpha) BCa interval excludes zero, using the same 9,999 resamples (reported as <= 0.001 or > 0.5 at the ends of the grid). Holm's step-down adjustment is applied within each family (the per-model primaries of one hypothesis); raw and adjusted p-values are reported beside the 95% interval. H1 and H3 are two-sided; H5 is one-sided. H0 is a classification and has no test. Cells with n < 30 receive an interval and no test.

### Data exclusion

1. Matching: ambiguous, unmatched, duplicate-conflict and patient-conflict images never enter selection; they are counted.
2. Verification, in this order: items marked skip and items flagged ambiguous, tracheostomy or out-of-field are excluded first, before any comparison with the label; only then is a reading that contradicts the RANZCR label treated as a disagreement, first as a matching error (the patient is re-matched) and, if it persists, as label noise (the item is excluded and counted). Exclusions come only from the reading and from matching re-checks; the detector never excludes.
3. Outputs: unparsable and truncated outputs are excluded from the denominators of hit_A, FAR_B, FAR_U, BA, ER_t and the abstain indicator and reported as their own rate per model, prompt and arm; a sensitivity row counts them as wrong. H3 uses only items parsable under both P1 and P2hi and reports the number dropped. A run with more than 10% unparsable outputs under P0 is flagged.
4. Pair-level indices (persistence, pair-correct) use pairs in which both images remain.

### Missing data

There is no imputation. Generations interrupted on the GPU machine are resumed from the logged keys, so every registered item x prompt is attempted. A model that cannot be run is reported as "not run". A stratum below its target takes all available patients, and cells below n = 30 receive an interval and no test.

### Exploratory analysis

Everything not listed as a primary test is exploratory or secondary: FAR_B versus FAR_U, persistence and pair-correct; B strata by other lines; blind P(yes) and greedy answers; ER_0.9 by prompt; AUROC of z; the first-token abstention log-probability; agreement of MedSigLIP with the RANZCR labels per arm and its development-pool AUROC; the unparsable-as-wrong sensitivity row; follow-up gap and direction; view, age and sex; the design effect. PLUS (only if run): P2lo, BLIND-demo, a MedGemma system-prompt sensitivity row, n = 200 pairs and 200 U, NGT, CVC and Swan-Ganz questions as RANZCR-label-only rows, same-state control pairs (60 A-A' and 60 B-B'), CheXzero as a second automatic verifier, and a risk-coverage curve from |z| with the verbal abstention point.

## Other

### Pre-registered fallback branches

- No usable pair list by noon of Day 3: RANZCR-direct pairs from the competition labels alone (B random, direction "unknown"); Qwen3-VL-2B on a Kaggle T4 in fp32 on the JPEGs resized to 1024 x 1024 inside Kaggle; MedSigLIP on Kaggle; MedGemma when the matching lands, otherwise "not run"; the matching is reported as partial.
- GPU machine unavailable: Qwen3-VL-2B on a Kaggle T4 in fp32 on the selected NIH PNGs; MedGemma deferred or "not run".
- Qwen3-VL-2B at NO-GO: the floor table is the headline; H1 and H3 run on MedGemma and on Qwen3-VL-2B's prior rows; Arm 1 is reported from the floor.
- Abstention never triggered: reported as the finding.
- MedSigLIP development AUROC near 0.5: Arm 1 is still run and reported as a null beside the AUROC.
- PLUS only: if CheXOne fails to load, NV-Reason-CXR-3B replaces it; if both fail, H2 is dropped with one sentence.

### Cut decisions

Before this registration only: n from 120 to 80 pairs, and the development pool from 300 to 200 or 150, recorded as the registered n. After the Day-3 run queue, no registered item is cut: PLUS and STRETCH items go first, then the development-selected beta sensitivity row, then nothing else. BLIND-grey is kept. P1 is never cut, because it is the H3 comparator.

### Data terms and ethics

RANZCR files (labels, JPEGs, identifiers, polylines and anything derived from them row by row) stay in private Kaggle notebooks and on the student's machine; the GPU machine holds only NIH PNGs and prompt text; nothing RANZCR-derived goes to the GPU machine, the code repository or OSF, and only aggregates are released. MedGemma and MedSigLIP are used under the Health AI Developer Foundations terms, with no clinical use and no redistribution of weights. PLUS models: Qwen2.5-VL-3B-Instruct under the Qwen Research License and CheXOne under CC BY-NC 4.0, evaluation only. No online LLM API and no PhysioNet data are used. No IRB determination was requested. The study uses only public, de-identified data released for research; no person is contacted, no identifiable information exists in either release, and the only human input is the student's own reading of the public images. Outputs are research measurements and are not for clinical use.

### Deviations

Any change after this registration is dated in the protocol change log with its reason and reported in the paper.
