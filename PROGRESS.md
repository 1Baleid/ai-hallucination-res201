# PROGRESS LOG: RES201 project (object hallucination in multimodal LLMs, chest X-ray devices)

Purpose: a dated record of what was done, what was decided and why, what changed, and what came out of it, written so it can be lifted into the final report (methods, timeline, decisions, limitations). One entry per working day. Keep entries factual; put numbers, file names and sources, not intentions. Newest entry last.

Conventions: dates are absolute (YYYY-MM-DD, Arabia Standard Time). "Agent work" means tasks executed by Claude Code (multi-agent workflows); "human work" means tasks done by the student. Files referred to live in this project folder.

---

## 2026-10-12 (Monday): project framing, literature sweeps, strategy v1, domain pivot, planning for the nine-day study

### Inputs received
- Course rules (URO email): RES201 is a team-based research course; the individual paper-format report is the graded deliverable; four outcome pathways; deadline defined as Week 13 (the registrar's Term 261 calendar puts Week 13 at 2026-11-08 to 11-12; classes began 2026-08-19; finals 2026-12-13 to 12-24).
- Student component title: "Benchmarking and Mitigating Object Hallucination in Multimodal Large Language Models Using Synthetic and Real Visual Data".
- Advisor reading list (7 items, all text-only LLM hallucination): LLM-Check (Sriramanan et al., NeurIPS 2024); Huang et al. 2023 survey; Xu et al. 2024 "Hallucination is Inevitable"; Kalai et al. 2025 "Why Language Models Hallucinate" (OpenAI; Nature 2026); HalluLens (Meta 2025); two GitHub topic pages.
- Umbrella program listing (received in the evening): "Evaluating VLMs/LLMs Hallucination in Domain-Specific Tasks", SDAIA-KFUPM JRCAI funded, project manager Dr. Muzammil Behzad: analyze hallucination in open-source VLMs on domain-specific QA (healthcare or education); detect, categorize, reduce via grounding and retrieval-based techniques.
- Student constraints: one GTX 1060 6 GB on a second machine; cloud GPUs only after a money estimate; no assigned sub-task; no advisor models or datasets; execution will mostly be done by agentic AI; no peer available for label audits.
- Deadline clarification (late evening): the FINAL report must be submitted about 2026-10-21; it must contain completed work with results; it may be scoped to what is doable in nine days; the plan document must be short.

### Agent work done
1. Literature and feasibility sweep, general domain (12 agents, about 1,100 web lookups, primary sources opened for every item marked verified). Output: `research/01_literature_and_feasibility_digest_2026-10-12.md` (about 100 items with URLs). Angles: benchmarks, mitigation, synthetic-data tooling, detection transfer from the reading list, newest 2025-2026 small models, compute and prices, venues; gap fills on the resolution/token-budget confound, an Arabic angle, statistics, and where hallucination originates.
2. Strategy v1 designed by a judge-panel workflow (4 drafts, 3 judges, synthesis, 4 adversarial skeptics, final revision; 13 agents). Output: `STRATEGY.md` v1 (general-domain, COCO-based "TriPOPE" design) and `research/02_strategy_design_log_2026-10-12.md`. Archived later as `research/STRATEGY_v1_general_domain_2026-10-12.md`.
3. Domain-fit sweep after the umbrella listing arrived (12 agents, about 1,260 lookups). Output: `research/03_domain_fit_digest_chest_xray_2026-10-12.md`. Angles: medical hallucination benchmarks, accessible chest X-ray datasets with object-level annotations, synthetic and edited CXR tools, open medical VLMs at 2-4B, grounding and retrieval mitigation, the education alternative, advisor fit and ethics; gap fills on RANZCR provenance, floor effects for small models on devices, edited-arm validity, and references in the advisor's own paper.
4. Strategy v2 design (3 drafts, 3 judges, synthesis, 4 adversarial skeptics: novelty, feasibility, rigor, advisor-and-ethics). The final reviser agent overran the 64,000-token output cap twice, so the synthesis and the four must-change lists were condensed by hand into `STRATEGY.md` v2 (compact reference: three stages, precedents table, Stage 2 steps, Stage 3 training track, budget, risks). Raw outputs: `research/04_strategy_v2_design_log_chest_xray_2026-10-12.md`, part 1.
5. Two supplementary designs (one workflow): Track C, a training-based mitigation track (seven recipes with code, licence and compute checks, then a correcting check that reframed the two load-bearing recipes and fixed stale tooling facts), and Track A, a 10-day proposal-grade deliverable (superseded by the results-report reading; its feasibility-evidence pieces were reused). Raw outputs: research/04, part 2.
6. Nine-day executed-study design (Stage 1: tiers, day by day, runs, items, prompts, metrics, mitigation, statistics, outline, cost, risks, deferrals) plus a feasibility check that recounted the hours (41-54 h for a student coding alone; about 30 student-hours under agent execution), corrected the item counts (360 test images, not 450), added the dev-pool P0 run that Arm 1's beta selection needs, stratified U for H1, added a Day-3-noon kill switch with a RANZCR-direct fallback, front-loaded the writing, and removed the peer audit. Raw outputs: research/04, part 3. Condensed into `STAGE1_PLAN.md` (the to-do list) and `PROTOCOL.md` v0.1 (the frozen protocol text).
7. Three-lens adversarial review of `STAGE1_PLAN.md` and `PROTOCOL.md` (executability; statistics and protocol validity; ethics, terms, wording and novelty) run as workflow wf_41b38c0e-035 (interrupted by a device restart, resumed); no fatal defect in any lens; all accepted fixes applied the same evening (PROTOCOL.md v0.3, STAGE1_PLAN.md v1.1, the two drafts, HANDOFF.md, STRATEGY.md) and recorded under "Changes" below.
8. Continuity files written or updated: `HANDOFF.md` (state, decisions, files, pending work, how to resume), `PROGRESS.md` (this file), `drafts/emails_day0.md` (instructor, advisor and IRB email drafts), and the persistent memory directory used by Claude sessions.

### Files produced today
- `STRATEGY.md` v2 (reference), `STAGE1_PLAN.md` (to-do list), `PROTOCOL.md` v0.1 (frozen protocol draft), `drafts/emails_day0.md`, `drafts/irb_protocol_summary.md`, `HANDOFF.md`, `PROGRESS.md`.
- `research/01_literature_and_feasibility_digest_2026-10-12.md`, `research/02_strategy_design_log_2026-10-12.md`, `research/03_domain_fit_digest_chest_xray_2026-10-12.md`, `research/04_strategy_v2_design_log_chest_xray_2026-10-12.md`, `research/STRATEGY_v1_general_domain_2026-10-12.md` (archive).

### Key verified facts established today (citable; sources in research/01 and research/03)
- Registrar calendar: Term 261 Week 13 = 2026-11-08 to 11-12 (early registration listed for weeks 12-14 = Nov 7-15).
- The RANZCR CLiP radiographs (30,083 images, 3,791 patients, 50,612 image-level labels, 17,999 polylines for ETT/NGT/CVC) are NIH ChestX-ray14 images re-annotated by 43 labellers (radiologists, radiology trainees, hospital medical officers), about 30% double-read, about 15% discrepancy on re-review (Tang et al., Sci Data 2021). No public UID-to-NIH mapping exists; it must be built by image dimensions plus perceptual hashing with a margin rule.
- NIH ChestX-ray14: 112,120 PNGs at 1024x1024, 30,805 patients; "no restrictions" on image use with three attributions (Box link, Wang et al. CVPR 2017, NIH Clinical Center acknowledgement).
- MedGemma 1.5 4B model card lists ChestX-ray14 only as an evaluation set (MIMIC-CXR is in training); the HAI-DEF terms are gated click-through.
- Object-CXR mirror (github.com/hlk-1135/object-CXR) exposes a Google Drive folder: 10,000 frontal CXRs from about 300 Chinese township hospitals, 5,000 with foreign objects annotated by 12 radiologists; support devices excluded; terms "free to download for scientific research and non-commercial usage".
- Small 2-4B generalist VLMs have no published device-existence numbers on chest X-rays; the nearest evidence (ReXVQA, 4-option MCQ) shows Phi-3.5-vision at chance on tubes/lines and Qwen2.5-VL at 65%; MedGemma-4B at 83.9%. A floor-effect pilot is therefore mandatory before fixing the primary device class.
- Plain contrastive decoding (VCD) is null-to-negative on chest X-ray VQA in three independent sources (MedHEval, Med-VCD, VGS-Decoding), so mitigation is built around grounding (detector-gated answers, region crops) and retrieval, with controls.
- Direct 2026 precedents that narrow novelty claims: Guzel et al. (Diagnostics 2026; four devices on RANZCR with closed models, only 3 device-absent cases), TAIX-VQA (Moll et al., ML4H 2025, arXiv 2510.11196; ETT, CVC, pacemaker and chest-tube yes/no questions on ICU radiographs, run on MedGemma-4B), McLaughlin et al. 2026 (Gemma-3-4B vs MedGemma-4B with refusals), PSF-Med 2026 (CXR paraphrase flip rates), Xiong et al. 2026 (RadEdit removal with a background-edit control), Lotfinia et al. 2026 (image swap/occlusion/removal with a support-device question at 224 px, MIT code).
- The advisor's paper (Alshahrani and Behzad, arXiv 2606.13211) inherits its headline "general beats medical (76.6% vs 51.3%)" from Kim et al. 2025 on the text-only Med-HALT benchmark; it also contains citation errors to flag before co-authoring ([45] mislabels MedHallBench; "CXR-VisHal" is a MedHEval subset; Med-HallMark has 6,341 samples; the DREAM PMCID is PMC12866389).
- Compute: on the GTX 1060, PyTorch 2.14 cu126 is the last supported build, fp16 runs at 1/64 rate, bitsandbytes NF4 works but 8-bit does not; fp16 on a Kaggle T4 is unsafe for Gemma-family and Qwen2-VL models; RunPod secure RTX A5000 $0.27/h (network volumes are secure-cloud only), community RTX 3090 $0.22/h.
- Verified late evening (web reads by the agent): Kalai et al. version of record is Nature 653(8116):1047-1051, 2026, DOI 10.1038/s41586-026-10549-w (Crossref); the MedGemma 1.5 4B card names MIMIC-CXR, Chest ImaGenome and other sets in its training sentence and lists ChestX-ray14 without a use marker in its data-ownership section and as a CXR14 row in its evaluation table, with no mention of RANZCR, 896x896 input to 256 tokens, BF16, HAI-DEF terms; the MedGemma 1.5 technical report (arXiv 2604.05081) lists ChestX-ray14 only as an evaluation set; the CheXOne paper (arXiv 2604.00493, Section 4.1) lists 30 training sources including ChestXray14 and Object-CXR but not RANZCR CLiP, so the device labels of the test images are not in CheXOne's training list; the Lotfinia et al. repository's `data_loader/build_utils.py` default prompt is "Is {display} present in this chest X-ray? Answer with a single word: Yes or No.", identical in form to P0; the archived 2021 Kaggle rules for RANZCR CLiP permit academic research and prohibit copies, redistribution and re-identification (quoted in PROTOCOL.md section 9).

### Decisions taken today and their justification
- Domain: chest X-ray (healthcare), not education. Reason: verified fit with the advisor's lab and paper; a fully credential-free data path exists; a dense 2026 literature to position against; the education side has no advisor footprint.
- Objects: medical devices (endotracheal tube first; nasogastric tube second; CVC/PICC only as a hard resolution class; pacemaker and foreign objects as reserves), not pathologies. Reason: discrete, verifiable by a trained non-expert, labelled with geometry in RANZCR, nearly absent from hallucination benchmarks, and not learnable from NLP report labels.
- Negative stimulus: the same patient's other frontal study with the device absent (edit-free "within-patient contrast"), plus matched unpaired negatives and text-only prior rows. Reason: no inpainting artefacts; no hallucination benchmark has used longitudinal device pairs; the ICU co-occurrence prior becomes the hypothesis rather than a nuisance.
- Mitigation family: detector-gated answers (CCD-style expert fusion on cached logits) and similar-case retrieval, each with the controls the literature lacks (detector-alone, shuffled-expert, constant-bias, kNN-vote-alone, discordant retrieval); plain VCD only as a reference row.
- Verifier independence: ianpan/cxr-lines-and-tubes is in-sample on RANZCR and may only generate masks; MedSigLIP and CheXzero are the automatic verifiers and are never evaluated models.
- Wording: "RANZCR CLiP labels (43 trained labellers)" not "radiologist labels"; "within-patient contrast" not "counterfactual"; "to our knowledge" on every novelty claim; "not for clinical use" on any edited image.
- Ethics sequence: IRB determination request to IRB@kfupm.edu.sa before any processing (NCBE Implementing Regulations v3 Article 10.19 for expedited review); NCBE certificate; model inference on public de-identified data proceeds while any rater session waits.
- Scope: a nine-day executed study (tiered by available hours) whose report exists with results on 2026-10-21; the longer program (synthetic probes, retrieval and region arms, full panel, reliability block, training with cluster access) is reference material.

### Changes to earlier plans
- Strategy v1 (general-domain COCO benchmark with decoding-time mitigation) was superseded the same day after the umbrella listing arrived; kept only for its protocol and statistics discipline.
- The second-rater audit (two hours of a peer's time) was replaced by automatic-verifier agreement plus a student intra-rater retest after 48 hours, because no peer is available.
- The "Week 13" schedule (report v1 by 2026-11-08) was replaced by the nine-day schedule (final report about 2026-10-21).
- The Stage 1 design was corrected before adoption: item counts fixed to 120 pairs + 120 U + 300 dev (360 test images, 660 PNGs with the dev pool); U stratified 60/60 by other labelled lines and matched on View Position so H1 is testable; the dev-pool P0 run added for both models; Arm 1's beta fixed at 1 a priori (dev-selected beta as a sensitivity row); verification order fixed (student reading saved and timestamped before any model output is opened; verifier agreement reported only, never used for exclusion); the 60-item retest raised to 100 items; the PLUS "peer audit" replaced by "advisor-provided lab rater, planned without"; the GTX 1060 taken off the critical path (MedSigLIP runs on the pod); writing front-loaded (Sections 1, 2, 4, Appendix B from Day 1); Day-3-noon kill switch to a RANZCR-direct Kaggle fallback; the report target set to 8-9 pages for CORE.
- The pod file schema was changed so that no arm, pair or label information reaches the pod (opaque item ids and NIH filenames only; labels joined locally), to keep RANZCR-derived data off rented machines until the Kaggle rules have been read.
- The mapping was dropped from the contribution list (Kaggle solutions already hashed RANZCR images against NIH); the yield table and the NIH-filename item list remain artefacts.
- Statistics review (workflow wf_41b38c0e-035, statistics lens; no fatal defect, ten majors, five minors, all applied as `PROTOCOL.md` v0.2 and matching edits to `STAGE1_PLAN.md`): P1's abstain option changed to "I don't know" so that P2-hi differs from P1 only by the penalty sentence; the cached first-token signal defined as a logsumexp over the yes and no surface-form ids with a coverage assertion (median yes+no mass >= 0.9 in the dev-pool smoke test) and a full-sequence fallback; a two-stage parser with an abstain lexicon and apostrophe normalisation (the old anchored regex missed "I cannot tell"); unparsable and truncated outputs excluded from denominators and reported as a rate, with "counted as wrong" as a sensitivity row; the H5 baseline set to the beta = 0 logit rule with one primary inference (the 9,999-permutation shuffled-expert p); the H0 gate made exhaustive (NO-GO, then GO, else CONDITIONAL; yes-rate over A, B and U under P0); Holm given a CI-inversion bootstrap p and a named family; the Day-4 read made blind to arm (random order, opaque ids) with smoke-test items drawn from the dev pool; "drop P1" removed from the cut order (it would have killed H3); the power note corrected (two strata of 60 detect 0.25 at 80% power, not 0.20; 100/100 detect 0.20 at 83%); BLIND-text and BLIND-grey rows reported as single deterministic generations without intervals; McNemar replaced by the clustered bootstrap under the two-pairs rung; oracle rows split into "oracle label" and "clipped oracle"; AUROC split into A-vs-B and A-vs-U; Whitehead et al. 2022 cited for Effective Reliability; the Kalai rubric strings located in the Nature article's Extended Data Table 2.
- Executability review (same workflow; no fatal defect): the image budget corrected from 540 to 660 PNGs (120 pairs are 240 images); a Restricted RunPod API key and SSH over exposed TCP (official PyTorch template) added to Days 1 and 3 so the agent can deploy, transfer and stop the pod; a blind-reading tool (`read_tool.py`) and the ETT reference sheet scheduled on Day 3; the transfer of the selected NIH PNGs specified (private Kaggle filename-list dataset, a second private notebook that copies the 660 files and renders the 60 spot-check pairs as a grid inside Kaggle); the cut order rewritten as dated decisions (n and dev pool only before the OSF registration; nothing registered cut after the Day-3 queue); the CheXOne pair set to 512x512 (its training setting); the Kaggle secret attachment made an explicit UI step; the data-boundary sentence unified ("stays on Kaggle and the student's machine, never on the pod"); the Day-1 done-when now includes the no-labelled-line patient count with a registered substitute H1 contrast; HANDOFF's panel bullet aligned to the two-model CORE; the 48-hour retest clock anchored to a logged 14:00 read deadline; the pHash shards made fault-tolerant; the P0 check closed as verified.
- Ethics, terms and wording review (same workflow; no fatal defect): the IRB request reframed as a determination request with the supervisor in copy as PI of record, a stated start date and a pause rule for the student's reading sessions, and the KFUPM IRB forms reserved if asked; the NCBE certificate moved before the Day-4 reading (registration Day 0, course Days 1-3); the IRB attachment corrected to describe the RANZCR-to-NIH linkage precisely and the reading's use honestly; the archived 2021 Kaggle rules 7.A and 7.B quoted into PROTOCOL.md with privacy and working-copy rules (private notebooks and outputs; the RANZCR pHash shard never leaves Kaggle); the MedGemma contamination wording unified against the MedGemma 1.5 technical report (arXiv 2604.05081; no asterisk claim); the second and third "to our knowledge" claims narrowed (TAIX-VQA already asks the ETT question of MedGemma-4B and Lotfinia 2026 already has blind-image rows and a support-devices block; CCD already fuses MedSigLIP logits with a random-expert test), the first claim qualified as credential-free with Chest ImaGenome, Medical-Diff-VQA, MC-CXR and MI-CXR named; CheXOne's licence cited from the Hub card (CC BY-NC 4.0) and run in instruct mode at 16 tokens; the NV-Reason-CXR-3B licence added; the HAI-DEF obligations restated (attribution on outputs, no Notice, no Clinical Use); "label-clean" replaced by a descriptive statement; the Kalai prompt described as "t substituted as a percentage", never "verbatim"; Guzel cited by title and DOI; the TAIX-VQA "634 images" figure dropped as unsupported.
- Kalai et al. is cited as the Nature 2026 version of record ("Evaluating large language models for accuracy incentivizes hallucinations", Nature 653(8116):1047-1051, online 2026-04-22, DOI 10.1038/s41586-026-10549-w; Crossref record verified 2026-10-12) together with the preprint id arXiv 2509.04664 ("Why Language Models Hallucinate"); the confidence-target instruction is quoted from the preprint until the Nature text is checked through the KFUPM library.

### Money and compute used today
- $0. All work today was literature, design and verification by web lookups; no GPU time was used.

### Open items carried to 2026-10-13
- Day 0 human steps (student): send the instructor and advisor emails (`drafts/emails_day0.md`); queue the IRB request for 08:00 Oct 13 and log the send time in `PROTOCOL.md` section 10; Kaggle late-join of ranzcr-clip-catheter-line-classification with the rules accepted, the Rules tab screenshotted and the Data Access/Use and Data Security clauses pasted into `PROTOCOL.md` section 9; Kaggle phone verification; HAI-DEF accepted on google/medgemma-1.5-4b-it and google/medsiglip-448; Hugging Face read token; `kaggle.json` and the HF token placed where the agent can use them. Tonight also: register on ncbeportal.kacst.gov.sa (course on Days 1-3, before the Day-4 reading). Day 1: RunPod account with $10 credit, a Restricted API key, the agent's SSH public key and the 50 GB Secure Cloud volume; attach the HF token as a Kaggle secret in the notebook UI after the agent's first push; OSF account.
- Day 0-1 agent steps: mapping notebook with `kernel-metadata.json`; Section 2 draft; repo skeleton; then Day 1 of `STAGE1_PLAN.md` (yield table, dimension-key test, sharded pHash pass).
- Apply the three-lens review findings (workflow wf_41b38c0e-035) to `STAGE1_PLAN.md` and `PROTOCOL.md` if not yet recorded here.
- Confirm on Day 0: the live Kaggle Rules text against the archived 7.A and 7.B quotes in `PROTOCOL.md` section 9 (screenshot filed). The P0 wording, the MedGemma card and the CheXOne training list were verified by the agent on 2026-10-12.

---

## Template for the next entries

## YYYY-MM-DD (Weekday): one-line summary
### Done (agent work / human work)
### Results and numbers (with file names)
### Decisions and changes (what, why)
### Problems and how they were handled
### Money and compute used
### Next


---

## 2026-10-20: GitHub repository preparation

- Agent work: reviewed the 12 existing Markdown documents for version control; no evaluation scripts, datasets, model outputs, or completed experimental results are present in this workspace.
- Added `README.md` as a document index and `.gitignore` to exclude credentials, local environments, research datasets, model weights, and derived working files.
- Prepared the project for a private GitHub repository under the local authenticated account `1Baleid`, named `ai-hallucination-res201`, with `main` as the initial branch. The repository name was corrected from RES299 to RES201 at the student's request; the local Git remote was updated accordingly.
- This session changes repository packaging only; no model inference, data processing, or compute spending was performed.

## 2026-11-02 (Monday): execution starts; code build; first real-data calibration

### Decisions and changes (what, why)
- Status at the start of the day: planning only. Nothing had been executed in October, so the plan's October calendar lapsed. The repository had been pushed to the private GitHub repo `1Baleid/ai-hallucination-res201` on 2026-10-20.
- Student decision: "ignore dates for now, focus on finishing it". Calendar dates in `STAGE1_PLAN.md` are on hold; the day order still holds.
- Student decision: no contact with the advisor. Removed the advisor email, the advisor copy and PI-of-record naming on the IRB request, the draft review, the citation memo and any lab rater (`STAGE1_PLAN.md` Day 0, 8, 9 and section 7; `PROTOCOL.md` sections 10 and 11; `drafts/`).
- Clarification recorded in `PROTOCOL.md` section 11: the H5 shuffled-expert permutation p is one-sided (real-expert gain compared with permuted-expert gains).

### Done (agent work)
- Checked this machine: Python 3.13, no GPU, no Kaggle or Hugging Face credentials, an empty RunPod key. Created the project environment `.venv` with the analysis stack and a CPU model stack (torch 2.15 CPU, transformers 5.22.1); versions in `docs/venv_freeze_2026-11-02.txt`; requirement files `requirements-local.txt` and `requirements-cpu-smoke.txt`.
- Wrote the code contract `docs/PIPELINE_SPEC.md` (layout, data schemas, function signatures, data-boundary and wording rules) and the fixed inputs `cxrdev/constants.py` and `cxrdev/prompts.json` (prompt strings from PROTOCOL v0.3).
- Launched the build workflow (run id wf_3868cc12-a66): core statistics library, Kaggle kernels and image matching, item selection, analysis and tables, GPU-pod runner, blind-reading tool, and drafts of report Sections 1 to 4, Appendix B and the OSF text. Results are recorded below when it finishes.
- Wrote `docs/HUMAN_STEPS.md` (Kaggle, Hugging Face, RunPod, blind reading; no advisor steps) and `scripts/build_report.py` (assembles the report, fills result placeholders, merges bibliographies, checks wording) with 5 passing tests.
- Obtained public NIH ChestX-ray14 data without any login, stored in the git-ignored `data/` folder: the first image archive from NIH's Box (4,999 PNGs, 1,335 patients) and the metadata file `Data_Entry_2017_v2020.csv` from a public Hugging Face mirror (112,120 rows, 30,805 patients; 67,310 PA and 44,810 AP). The Google Cloud copy of the dataset needs authentication and was not used.

### Results and numbers
- NIH original image sizes: 10,471 distinct sizes; 8,882 sizes occur only once, so the size key alone identifies those images; the largest size group holds 37,153 images, which the perceptual hash must separate.
- Matching calibration on real radiographs (300 trials, 150 patients with two or more films; a simulated RANZCR copy made by resizing the NIH image to its original size and saving it as JPEG; 256-bit perceptual hash):

| Measure | JPEG quality 95 | JPEG quality 75 |
|---|---|---|
| Copy to its own source, median / max distance (bits) | 0 / 4 | 0 / 6 |
| Copy to the same patient's nearest other film, min distance | 44 | 44 |
| Copy to the nearest of 200 unrelated films, min distance | 62 | 60 |
| Smallest margin, sibling minus source | 44 | 42 |

  The planned margin rule (16 bits) has wide headroom on real films, so sibling films of the same patient are not a practical source of mismatches. Caveat: the real RANZCR conversion settings are unknown; the matching kernel recalibrates on the real data.

- Real-model probe on this laptop's CPU (Qwen3-VL-2B-Instruct, revision 89644892, bf16, transformers 5.22.1; two real NIH radiographs, one AP and one PA, plus no-image rows; log and script in `results/probes/`). This is a feasibility check, not a study result:

| Check | Outcome |
|---|---|
| Visual tokens for a 1024x1024 X-ray with min = max pixels = 1,048,576 | exactly 1,024 (grid 1 x 64 x 64) on both images |
| Yes/No surface forms | all twelve forms are single tokens |
| First-word mass on Yes/No, plain prompt (P0) | 1.00 on both images (coverage check passes) |
| Same, "I don't know" prompt (P1) | 0.999 on both images |
| Same, penalty prompt (P2hi) | 0.79 and 0.63; the rest moves to "I" (the start of "I don't know") |
| Answers | parsed cleanly ("No" / "no" on both images under all prompts) |
| No-image row, plain prompt | answered "Yes" with P(yes) 0.68, the prior the blind rows are meant to expose |
| CPU time | about 5 to 6.5 minutes per image question in bf16, so the study must run on the rented GPU |

### Money and compute used
- $0. Local CPU only.

### Explainer video (same day, at the student's request)
- Made a narrated plain-language explainer video of the project, `docs/video/project_explainer.mp4` (about 8.4 minutes, 1080p, English subtitles and four chapters: the project, where we are now, is it tangible?, what's next), with the subtitles also as `docs/video/project_explainer.srt`. The content is the two plain-language status messages from this session (project summary, status, the student's steps, the tangibility assessment with its weaknesses and realistic expectations).
- Built by `docs/video/make_explainer.py` (Pillow frames, the offline Windows voice "Microsoft Zira Desktop", ffmpeg); rerun it to regenerate after edits. It shows one real NIH film (`00000065_000.png`); the tube in the "tube in / tube out" scene is drawn on for illustration and labelled as such.
- $0; local CPU only.

### Kaggle access (same day; student step 1 of `docs/HUMAN_STEPS.md` done)
- The student accepted the RANZCR CLiP competition rules (Kaggle reports `userHasEntered = True`) and created a new-style API token, saved by the agent as `C:\Users\messi\.kaggle\access_token` (outside the repository). Checked with the Kaggle tool: the competition file list loads.
- The live Rules text (copied by the student 2026-11-02) matches the archived quotes in `PROTOCOL.md` section 9 word for word ("DATA ACCESS AND USE", 7.A, 7.B); section 9 updated. Two clauses noted for the report: A.2 forbids human rating of the competition's test set (we use train images only), and A.3 asks for sponsor approval of public communications about participation in the competition (this study is research use of the data, not a competition entry; noted as a limitation, no action).

### Code build results (agent work, continued)
- The build workflow (wf_3868cc12-a66) was interrupted by a restart of the agent session; six builders had finished, and two follow-up agents finished the analysis and pod-runner work. Every module now has its own passing tests:

| Part | What it does | Tests |
|---|---|---|
| Core library (`cxrdev/prompts.py`, `parse.py`, `logits.py`, `stats.py`, `fileio.py`) | prompt strings, answer parser, first-word yes/no signal, Wilson and clustered bootstrap intervals, McNemar, Holm, kappa, AUROC, power | 145 passed |
| Kaggle side (`cxrdev/imghash.py`, `matching.py`, `ranzcr.py`, `nih.py`, `kaggle_kernels/`) | perceptual hashing, RANZCR-to-NIH matching, three private CPU kernels and their bundler | 89 passed |
| Item selection (`cxrdev/items.py`, `scripts/select_items.py`) | pairs, unpaired negatives, dev pool, opaque ids, pod list, reading lists | 67 passed |
| Analysis (`cxrdev/metrics.py`, `hypotheses.py`, `arm1.py`, `tables.py`, `figures.py`, `scripts/`) | metrics, H0/H1/H3/H5, detector-gated answers and controls, Tables 1-3, Figure 1, end-to-end synthetic dry run | 77 passed |
| GPU-pod runner (`pod/`) | model runs with cached first-word signal, MedSigLIP scores, setup and queue scripts, README | 93 passed, 1 opt-in skipped |
| Reading tool (`tools/`) and `docs/ETT_reference_sheet.md` | blind reading and retest with saved fingerprints | 94 passed |
| Report drafts | Sections 1-4, Appendix B, `osf/preregistration.md`, bibliography | wording and citation checks pass |

- Whole suite after the follow-up work: 569 passed. The full-size analysis (9,999 resamples and permutations, two models) takes about 2 minutes on this laptop.
- The pod runner was run for real on this CPU through its own command line on one NIH image: 1,024 visual tokens, answer "No", first-word yes/no coverage 0.99998, same token ids as the probe.
- Decision (PROTOCOL v0.4, before any data exist): every tube-present image is read (360 reads instead of 270) and images whose only airway tube is a tracheostomy are excluded, because the RANZCR CLiP descriptor says tracheostomy tubes were labelled "ETT normal". It adds about 45 minutes to the student's reading. Smaller v0.4 clarifications: 256-bit hash wording, the dimension-key rule, six yes/no word forms, one primary endpoint for the optional general-vs-medical test, when the clustered abstention test applies, and the unpaired arm staying at 60 + 60.
- Running at the time of writing: an integration agent (cross-module fixes and an end-to-end run on real NIH images with simulated RANZCR copies) and a report reviewer (consistency of the drafts with the protocol and code, bibliography keys).

### Hugging Face access (same day; student step 2 of `docs/HUMAN_STEPS.md` done)
- The student accepted the Health AI Developer Foundations terms on `google/medgemma-1.5-4b-it` and `google/medsiglip-448` and created a read token (`res201`, account Baleid); the agent saved it with `hf auth login` to `C:\Users\messi\.cache\huggingface\token` (outside the repository). Access check passed on both gated models.

### RunPod access (same day; student step 3 of `docs/HUMAN_STEPS.md` partly done)
- The student added credit (balance $10.11, auto-pay off) and created an API key (`res201`, read/write); the agent saved it in `C:\Users\messi\.runpod\config.toml` (outside the repository) and checked it with a read-only API query. Spend rate $0/h. One old pod from 2026-07-30 (`incredible_azure_bee`, stopped, no GPU, no volume) exists on the account; left untouched. The 50 GB network volume is deliberately not created yet (it bills about $3.50 per month while idle); the agent creates it when the GPU runs are ready, and the SSH key is added then.

### OSF preregistration skipped (same day; student decision)
- The student asked why the OSF registration was needed. Answer given: it is optional (not a course requirement); it only adds credibility by time-stamping the plan before results. The student chose to skip it.
- Replacement: the protocol is frozen by a dated git commit before any model output. Report Section 4.9 is now "Pre-specification" and says the study was not publicly preregistered (a TODO holds the freeze commit hash and date). In the report "registered" became "pre-specified"; in `PROTOCOL.md`, `STAGE1_PLAN.md` and code comments "registered" now means "fixed in the protocol before any model output" (`PROTOCOL.md` section 11).
- Files changed: `PROTOCOL.md` (status line, section 6 and 8 headings, cut-decision wording, data-terms line, new section 11 entry), `STAGE1_PLAN.md` (Day 1 OSF account and Day 2 registration removed; Day 2 ends with the freeze commit), `STRATEGY.md`, `HANDOFF.md`, `docs/HUMAN_STEPS.md` (step 2b removed), `docs/PIPELINE_SPEC.md`, report Sections 3 and 4, `kaggle_kernels/README.md`, and comments in `cxrdev/items.py`, `scripts/select_items.py`, `scripts/e2e_real_nih.py`, `tests/test_items.py`, `results/probes/probe_seen_filenames.txt`. `osf/preregistration.md` moved to `drafts/osf_preregistration_unused.md` with a "not used" note.
- Explainer video: point 3 ("the plan is publicly time-stamped on OSF") replaced by "the plan is locked before any results"; `docs/video/project_explainer.mp4` and `.srt` regenerated.
- Unchanged: the 11 probe-seen NIH images stay excluded, so no study item has a model output before the freeze.
- Checks: `tests/test_items.py` and `tests/test_build_report.py` pass.
- Next (agent): the protocol-freeze commit, then the Kaggle image-matching run. Nothing waits on the student until the blind reading.
- Money: $0.

### Kaggle hash run (same day; agent work, first real-data compute)
- Private kernel `abdullahbaleid/res201-hash`, pushed from the bundle built by `kaggle_kernels/build_kernels.py` (copy of the pushed bundle in the git-ignored `data/kaggle/pushed/`); outputs in the git-ignored `data/kaggle/res201-hash/`.

| Measure | Result |
|---|---|
| NIH images hashed (256-bit pHash and dHash) | 112,120 of 112,120, 0 errors, all 12 folders |
| Run time | 28 minutes on 4 CPU cores (Kaggle's Python 3.13.15, pandas 2.3.3, numpy 2.1.3) |
| RANZCR JPEG sizes found among NIH original sizes | 200 of 200 sampled (share 1.0); median 21,961 NIH candidates per size, so the hash does the real separation |
| Yield table | identical to the local count (30,083 images, 3,255 patients, 753 patients with both ETT states) |

- The size test confirms the descriptor's statement that RANZCR enlarged each NIH image to its original resolution, so the size key is used in matching.
- Software checks on real NIH images and the real gated models (CPU, this laptop): MedGemma 1.5's processor gives exactly 256 image tokens per X-ray, single-token yes/no forms and the right end-of-turn token; MedSigLIP scored 8 NIH films through `pod/run_medsiglip.py` (about 40 s per image on CPU; it needed `sentencepiece` and `protobuf` locally, which the pod requirements already list). These 8 images and the 3 Qwen probe images are excluded from the study (`results/probes/probe_seen_filenames.txt`).
- Money: $0 (Kaggle CPU is free).

### Integration and end-to-end test on real NIH images (same day; agent work)
- An integration agent closed the gaps between modules and applied the v0.4 decisions in code: all A images read (360), tracheostomy-flagged A images excluded before any disagreement check, the 11 test-touched NIH images excluded through `select_items.py --exclude-filenames`, H1 using the substitute contrast under every rung, H3 clustered whenever a patient has two pairs, a practice list of 10 dev images for the reading tool, strict JSON outputs, and a MedSigLIP precision column. Details: `docs/PIPELINE_SPEC.md` section 7.
- End-to-end run on real data (`scripts/e2e_real_nih.py`, write-up `docs/E2E_REAL_NIH.md`): 1,510 real NIH PNGs from 400 patients, 595 simulated RANZCR copies (resized to each image's real original size, JPEG quality 95), 30 of them with no source in the set; bundled Kaggle kernels run locally, then item selection, simulated model outputs, verification, analysis and report build.

| Check | Result |
|---|---|
| Matching accuracy | 565 of 565 copies matched to the right NIH file; 0 wrong; 30 of 30 sourceless copies unmatched |
| Distance to the true source (256 bits) | 0 to 12 bits (median 2) |
| Distance for sourceless copies | 46 to 120 bits (median 80); automatic cut 29 bits |
| Margin to the second candidate | at least 44 bits |
| NCC of true matches | 0.9967 to 0.99999; the tie-break was never needed |
| Downstream | tables, figures and report built; the 3 tracheostomy-flagged A images excluded for that reason, not as disagreements; no RANZCR id, read id, item id or filename in the results |

- Report fixes: the U-items paragraph now describes the substitute contrast with its own counts; the Kaggle rules-check and gated-terms dates (2 November 2026) filled in; the analysis writes all four U-stratum counts so the report resolves under either design.
- Whole test suite: 595 passed (including the real-NIH end-to-end test).
- Money: $0.
- Next (agent, after the student's go-ahead): the dated lock-in commit and push to the private GitHub repo, then the Kaggle matching run, item selection, image export, and the GPU runs.

### Meeting-briefing video (same day, at the student's request)
- Made a second narrated video, `docs/video/advisor_briefing.mp4` (about 7.5 minutes, 1080p, English subtitles also as `docs/video/advisor_briefing.srt`), to prepare the student to explain the project to Dr. Behzad. Seven chapters: the one-breath pitch; the idea (problem, why tubes, the suspicion and the 68% no-picture hint, the three kinds of test, the two models, the two fixes); the data puzzle (NIH 112,120 images, RANZCR 30,083 labelled copies, fingerprint matching); progress (the 11-step table, 753 eligible patients against 120 needed, $0 spent); the student's decisions; likely questions with answers; and the open due-date question. The "what the other sessions are doing" part of the source message was left out as not relevant to the meeting.
- The video engine is now shared: `docs/video/videokit.py` holds the drawing helpers and the build code, and `make_explainer.py` and `make_briefing.py` hold each video's scenes. The fingerprint picture is a simplified 16 x 16 illustration made from a real NIH film, labelled as such; the explainer video itself was not changed.
- Money: $0.
- Updated the same day with the results already in hand: the matching trial run (565 of 565 simulated copies matched correctly, 0 wrong, 30 sourceless copies left unmatched; from the end-to-end test above), the Kaggle size check (200 of 200 sampled RANZCR images have an NIH original size), about 600 automatic checks (was 570), and "the software works on real X-rays" (Qwen answers cleanly; MedGemma's input and MedSigLIP checked) in place of "the AIs run correctly". Changed scenes: data, puzzle, step table, good news, results answer. Now about 8.3 minutes; rebuilt with `docs/video/make_briefing.py`.
- Wording fix on the puzzle narration: "a full trial run ... the same way the competition made its images" became "a trial run ... imitating the way the competition made its images", plus "the real images may differ a little, so the real run checks this again". Reason: the trial used 1,510 NIH images with simulated copies (JPEG quality 95 assumed; the real RANZCR conversion settings are unknown), so it supports the method but does not prove the real-data match rate. Rebuilt; 8.4 minutes.
