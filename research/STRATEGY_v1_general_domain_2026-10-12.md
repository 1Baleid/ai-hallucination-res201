# Research Strategy

## Benchmarking and Mitigating Object Hallucination in Multimodal LLMs Using Synthetic and Real Visual Data

Version 1.0, 12 October 2026. Prepared for the RES201 project (KFUPM, ICS).

How to read this document: Section 0 is the one-page version. Sections 1 to 4 explain the situation, the literature gap and the research questions. Sections 5 and 6 are the plan itself (phases and the ten major steps). Sections 7 to 12 are the operating rules: protocol, models, budget, schedule, report outline, risks. Section 13 records how the plan was built and what still needs checking. The supporting evidence (about 100 papers, tools and prices, with URLs) is in `research/01_literature_and_feasibility_digest_2026-10-12.md`. The design log with the adversarial review that shaped this plan is in `research/02_strategy_design_log_2026-10-12.md`.

---

## 0. Executive summary

**Working title.** Same Scene, Three Sources: Penalty-Scored Object-Existence Evaluation of 2-4B Open VLMs on Matched Real, Edited and Generated Images, with Bias-Controlled Mitigation.

**The idea.** Build TriPOPE, a small, reproducible benchmark in which every test item exists in three versions of the same scene: the real COCO photo (R), the same photo with one object cleanly removed (E), and a layout-matched generated twin at the photo's exact pixel size (S). Ask the same six yes/no object-existence questions on all three. Then do three things nobody has done together on 2026 small open vision-language models:

1. Measure whether the image source changes hallucination once the visual-token budget is identical, and whether a "yes" on a removed object comes from editing traces or from the language prior.
2. Score the answers under abstention-aware rules taken verbatim from Kalai et al. (the paper on your advisor's list), with an explicit "I don't know" option and a stated penalty, and compare verbal abstention with what the model's own logits and hidden states could have supported (the LLM-Check bridge).
3. Test whether training-free mitigation (VCD and a prior-contrast variant) and a released aligned checkpoint (SENTINEL) actually improve seeing, or only shift the yes/no bias, using the control experiments that two 2025-2026 critiques demand.

**Why it works as a report and as a paper.** Every hypothesis is pre-registered with one primary test, so each result is reportable whichever way it falls. The design is tighter than any published synthetic-vs-real object-hallucination study (pixel-identical arms, logged token counts, corrected labels, human audit, confidence intervals, paired tests), which is exactly what reviewers now ask for. It benchmarks and mitigates, with synthetic and real data, as the project title promises, and it makes three of the five papers on the reading list load-bearing rather than decorative.

**Cost.** Expected $10-25, hard ceiling $40: about 30-45 GPU-hours on a rented RTX 3090/4090 ($7-15), a small storage volume ($3-4), at most $2-5 of API calls. Your GTX 1060 does all image editing and generation unattended at night (25-40 hours). Kaggle's free tier covers triage and backup. No training anywhere.

**Timeline.** Six core weeks (Oct 12 to Nov 22) plus two buffer weeks. The schedule has a hard checkpoint at the end of Week 4 (Nov 8): a complete, submittable report v1. That matters because the KFUPM registrar's calendar puts Week 13 of this term at Nov 8-12, 2026. If your course deadline is literally Week 13, v1 is your submission; if you have until late November or next term, the full plan with appendices and extensions applies.

**First three actions.** (1) Get the exact RES201 due date and report template in writing. (2) Run the Day-1 precision and timing triage of the five core model runs on Kaggle (fp16 is known to crash bf16-trained models on T4s; the plan assumes you rent a native-bf16 pod for headline numbers). (3) Compute how many usable images the selection rules yield from POPE and RePOPE before generating a single image.

---

## 1. Situation and assumptions

### 1.1 What is fixed

- Course: RES201. The individual paper-format report is the graded deliverable. Conference or journal routes still require the individual report. You want the report tangible enough to push to a workshop or student research workshop afterwards.
- Project title: Benchmarking and Mitigating Object Hallucination in Multimodal Large Language Models Using Synthetic and Real Visual Data.
- Reading list (all text-only LLM hallucination): LLM-Check (NeurIPS 2024), Huang et al. survey (2023), Xu et al. "Hallucination is Inevitable" (2024), Kalai et al. "Why Language Models Hallucinate" (OpenAI 2025; also Nature 2026), HalluLens (Meta 2025), plus two GitHub topic pages.
- Compute: one GTX 1060 6 GB (Pascal) on another machine; willingness to rent cloud GPUs after a cost estimate; APIs in small amounts only.
- No assigned sub-task, no teammate split, no advisor models or datasets.

### 1.2 Deadline (verified on the registrar's calendar, 12 Oct 2026)

Term 261: classes began Wednesday Aug 19, 2026 (week 1); mid-term break Oct 20-22 (week 10); early registration Nov 7-15 (weeks 12-14), so week 13 is Nov 8-12; last day of classes Dec 10; finals Dec 13-24. The URO email defines the deadline as "Week 13". Two scenarios are built into the schedule:

- Scenario A, due Nov 8-12: submit report v1 produced at the end of Week 4 (steps 1-7 of the plan; arm S and the appendices may be thinner and are labelled preliminary).
- Scenario B, due later (late November, or Week 13 of the next term if you extend): submit report v2 with the validity appendix and gated extensions.

Confirm which applies in writing on day one.

### 1.3 Assumptions you can change

- Part-time effort of roughly 15-20 hours per week for six weeks, plus a second rater (a teammate or graduate student) for about 4 hours.
- A $40 spending cap is acceptable to you and your advisor.
- English prompts only this term; the Arabic/Najdi extension is a follow-up (Section 12).
- Analysis and benchmark construction, not model training, is the contribution. Training-based mitigation is evaluated through released checkpoints only.

---

## 2. What the literature says and where the gap is

The digest in `research/01` has about 100 verified items with URLs. The facts that shape this plan:

1. **The standard benchmarks are saturated or noisy.** POPE (9,000 yes/no questions over 500 COCO images) scores 86-90 for everything from 2B to 78B models; it rewards saying "yes"; RePOPE (2025) found 9.3% of its positive labels false and 13.8% ambiguous, and the corrections reorder models; HQM (2024-2026) measured its parallel-forms reliability at 0.36. CHAIR depends heavily on caption length (LLaVA-1.5 scores 20.6 at 64 tokens and 45.0 at 512). AMBER is the only LLM-free generative benchmark. HallusionBench is still discriminative but is mostly charts and illusions.
2. **Synthetic and edited images are the 2025-2026 frontier, but nobody pairs them per item.** ODE (CVPR 2025), AutoHallusion, HaloQuest, VHTest, Causal-HalBench (AAAI 2026), FREAK (ICLR 2026), CIPHER/OHC-25K (CVPR 2026), POPEv2, HII-DPO, DO-Bench and ReactBench all use generated or edited images, but none offers a real photo, an edited version and a generated twin of the same scene with identical questions, verified labels and identical pixel size. BEAF (ECCV 2024) and ROHE (NAACL 2025) did LaMa removals on COCO and are the direct precedents for the edited arm.
3. **The direction of the synthetic-vs-real effect is unresolved.** "AIGCs Confuse AI Too" (2024) found more hallucination on SD 1.5 re-renders; HaloQuest found slightly less on Midjourney images; ODE attributes part of COCO's advantage to contamination. All used 2023-2024 models.
4. **A confound nobody controlled: the visual-token budget.** Modern models use dynamic resolution. A square 512 px generated image and a 640x480 COCO photo receive 256 vs 1,792-3,328 tokens under InternVL tiling and 324 vs 391 under Qwen2.5-VL. POPE accuracy collapses below roughly 200 tokens. None of the synthetic-vs-real papers logged or matched token counts, and the two main harnesses (lmms-eval, VLMEvalKit) silently give the same image 391 vs about 1,302 tokens.
5. **Abstention is not scored in object hallucination.** POPE forces yes/no. Kalai et al. argue that accuracy-only scoring rewards confident guessing and propose explicit confidence targets with penalties. "Seeing without Looking" (2026) added an unpenalized "unknown" option to POPE on 4B+ models and found it rarely used; MM-AQA (2026) and Rep2Act (2026) showed VLMs rarely abstain although their internals know. No paper runs Kalai's verbatim penalty instruction as a dose-response on object existence, or crosses it with image source.
6. **Mitigation claims are under attack.** "The Mirage of Performance Gains" (2025) and "Rethinking the Effectiveness of Contrastive Decoding" (2026) showed that VCD-style gains on POPE come from a one-directional yes-bias and from sampling collapsing to greedy; a trivial output-layer bias matches them. "Does Playing it Safe Count as Faithfulness?" (Sept 2026) showed lower hallucination often means lower recall. These controls were never run below 7B, on a training-based method, or on synthetic inputs.
7. **Internal-signal detectors exist for VLMs but not at 2-3B and not across image sources.** Logit-lens internal confidence (ICLR 2025), HALP (EACL 2026), UniProbe (2026), GLSim (NeurIPS 2025), HalluPrism and "Anatomy of a Lie" (2026) use logits, hidden states or attention on 7B+ models. LLM-Check itself has one multimodal citation (HalluShift++). Nobody tests whether a selection threshold set on real images transfers to edited or generated twins.
8. **Nobody reports confidence intervals.** Among benchmark-construction papers (POPE, AMBER, HallusionBench, VHTest, HaloQuest, ODE, BEAF, ROHE) none reports intervals or paired tests, while 2026 methods papers (WEP, GHOST-Q) now do.
9. **2026 small models are untested on object hallucination under one protocol.** Qwen3-VL-2B (Oct 2025), Qwen3.5-2B (Feb 2026), InternVL3.5-2B (Aug 2025), Gemma 4 E2B (Apr 2026), LFM2.5-VL-3B (Aug 2026) and MiniCPM-V 4.6 (May 2026) are all open, fit 6-16 GB, and vendor hallucination numbers disagree with independent ones.

**The gap this plan fills.** Per-item matched real/edited/generated triplets at identical token budget, scored with abstention-aware rules, on 2026 2-4B models, with mitigation tested under bias controls, with the statistics reviewers now expect. Each piece has a nearest neighbour (named in Section 6); the combination has none.

---

## 3. Research questions and contributions

**RQ1 (image source at matched token budget).** On matched triplets, do 2025-26 small open VLMs answer object-existence questions differently by image source once visual-token counts are identical? Is a "yes" on a removed object explained by residual editing traces (compare LaMa, SD-inpainting and grey-mask removal of the same objects) or by the scene-conditioned language prior (text-only blind run, and the same model's false-acceptance on real-image adversarial negatives of the same class)? R-vs-E is a same-scene single-edit contrast. R-vs-S is "photo vs generated twin" and carries a COCO-contamination flag rather than being called a provenance effect.

**RQ2 (abstention; the Kalai/HalluLens bridge).** When the prompt adds an unpenalized "cannot tell" option (P1) and when it states Kalai et al.'s confidence target at t = 0.5 (penalty 1) and t = 0.9 (penalty 9) (P2), does the stated penalty move abstention, false-acceptance rate and Effective Reliability? Do rankings change? Does the response differ by image source? How large is the gap between what the model's logits and hidden states could support (selective prediction) and what it verbalizes?

**RQ3 (mitigation under controls).** On the hero model, do VCD and a prior-contrast variant, and on Qwen2-VL-2B the SENTINEL alignment, beat the Mirage output-layer bias control and the Rethinking-CD constant-bias and noise controls at matched yes-ratio on R, E and S? Do they change seeing or only guessing, and what do they cost in recall and coverage?

**Contributions.**

- C1. TriPOPE: a reproducible, pixel-identical, RePOPE-filtered, human-audited real/edited/generated triplet set with a three-remover ablation, blind baselines, and construction and reliability statistics.
- C2. The first penalty-scored (verbatim Kalai confidence target) object-existence evaluation of small VLMs, crossed with image source and compared with logit- and hidden-state-level selective prediction at matched coverage.
- C3. The first size- and token-matched replication of the "synthetic images induce more hallucination" claim on 2025-26 2-4B models with cluster-aware paired tests, plus a measured trace-vs-prior attribution for removed objects.
- C4. Mirage/Rethinking-CD controls at 2-3B scale, on a training-based checkpoint and on edited and generated inputs, with informativeness reported.

**Publishable either way.** A source gap at matched resolution confirms the 2024 finding on modern models with the token explanation excluded; no gap re-attributes it. Equal "yes" rates across removers at the blind-text level attribute removed-object hallucination to the prior; unequal rates attribute it to traces. Zero verbal abstention next to informative logit curves quantifies the representation-action gap at object level; non-zero abstention yields measurable re-ranking. A mitigation point on the base model's bias curve is a bias shift; a point above it with a paired-bootstrap interval is a real gain.

---

## 4. How the advisor's reading list feeds the project

| Paper | How it is used |
|---|---|
| LLM-Check (Sriramanan et al., NeurIPS 2024) | Its three signal families are implemented training-free on the hero model inside one forward pass: logit level (first-token P(yes)/P(no), labelled the Reliable VQA softmax baseline), hidden-state level (Hidden Score: mean log singular values of the prompt hidden-state matrix at a dev-chosen layer, image and text blocks separately, visual sink tokens removed), attention level (per-head log-eigenvalue statistic of the attention map). Used as selection functions: risk-coverage, AUROC, coverage at fixed risk, real-to-synthetic threshold transfer. Step 6. |
| Huang et al. survey (2023) | Frames object hallucination as input-conflicting hallucination driven by co-occurrence priors; motivates POPE's adversarial negatives, the edited arm (object gone, scene prior intact), the text-only blind run and the class-matched false-acceptance reference. Related work. |
| Xu et al., "Hallucination is Inevitable" (2024) | Backdrop: the extension to MLLMs is argued (an image is a finite string, an MLLM a computable function) with the caveat that the theorems concern adversarial ground truth, not a fixed object-presence task. Consequence drawn: calibrated abstention, not elimination, is the target. Discussion. |
| Kalai et al., "Why Language Models Hallucinate" (2025; Nature 2026) | (a) Accuracy-only yes/no scoring rewards guessing, instantiated by POPE's forced choice and Mirage's output-layer bias. (b) The Section 4.2 instruction is quoted verbatim and run at t = 0.5 and 0.9, scored as Effective Reliability with c = t/(1-t) (footnote: the paper prints "t = 0.75 (penalty 2)" although the formula gives 3). (c) The paper's own sentence that the pretraining distribution is "over text or multimodal inputs if included" licenses the bridge. (d) Theorem 1 is cited once as motivation; it is vacuous for binary polling and its subtracted terms are not measurable, so no number is presented as a test. Steps 4-6, Discussion. |
| HalluLens (Meta 2025) | NonExistentRefusal gets an image-grounded analogue (existent class, absent instance): false-acceptance rate on verified-absent yes/no items, FAR on removed objects, and an open-ended "Describe the {object} in this image" arm with a present-object control. Steps 4-5. |
| GitHub hallucination topics | NishilBalar/Awesome-LVLM-Hallucination is the currency check and the release target (pull request in Week 6). |

---

## 5. Phases

| Phase | Weeks | Goal | Deliverables |
|---|---|---|---|
| 1. Feasibility and freeze | Week 1 (Oct 12-18) | Confirm the due date; precision and timing triage of all core models; pin environments; compute the realised item yield before any generation; freeze PROTOCOL.md, the hashed image list and the dev split; pre-register H1-H6 on OSF; start LaMa edits and ControlNet twins on the 1060 | Written due date and scenario; feasibility note (s/item, VRAM, tokens, flip rates) and the GPU decision; realised-yield table; PROTOCOL.md v1.0; OSF registration; citation-search log; second rater booked |
| 2. Arms R and E | Week 2 (Oct 19-25, mid-term break Oct 20-22) | Build R and E with all six labels verified; SD-inpaint and grey variants; blind conditions; negated form; pilot on E and S pairs; anchor run with RePOPE | Dataset v0 on HF; Table 1 construction statistics v0; custom lmms-eval tasks validated; pilot discordance and N fixed in OSF; anchor rows with RePOPE and contamination flags |
| 3. Core evaluation on R and E; S verified | Week 3 (Oct 26-Nov 1) | Run P0/P1/P2 on R and E, both blind conditions, the removed-object probe and the remover ablation; verify arm S labels; draft Sections 1-3 | Per-item JSONL; Table 2 (R and E columns) with tests and intervals; arm S verified (150 images) with covariates; Sections 1-3 and the related-work table |
| 4. S, false premise, signals, mitigation; report v1 | Week 4 (Nov 2-8) | Complete and freeze the three headline exhibits; produce a complete report v1 | Table 2 final; Figure 1 (risk-coverage); Table 3 (mitigation); Table 4 (false premise); report v1 submittable by Nov 8; EACL SRW mentorship request sent by Nov 6; go/no-go on every extension |
| 5. Validity appendix and gated extensions | Week 5 (Nov 9-15) | Reliability block, RePOPE and contamination sensitivity, token-budget 2x2; only the extensions that passed their gates; draft v2 to the advisor | Appendix tables A-D; extension results or explicit "not run"; draft v2 by Nov 13 |
| 6. Revision, release, submissions | Week 6 (Nov 16-22) | Revise after advisor comments; release code and data; submit final report v2 (scenario B); NAACL SRW mentorship request | Final report v2; public GitHub repo and HF dataset; mentorship request; pull request to the awesome list |
| Buffer | Weeks 7-8 (Nov 23-Dec 6) | Absorb slippage; finish the SRW manuscript before finals; JURI submission | Reruns; SRW manuscript by Dec 10; JURI submission |

---

## 6. The ten major steps

Steps 2-8 are the scientific contributions; steps 1, 9 and 10 make them deliverable. The order follows dependencies.

### Step 1. Day-one feasibility: deadline, precision triage, pins, item yield, protocol freeze, pre-registration

- **What.** Confirm the due date. Triage every core model for precision and throughput on Kaggle T4 (200 POPE items each in fp16, then fp32 or emulated bf16). Decide the GPU path. Pin both environments. Compute the realised TriPOPE yield from RePOPE and COCO annotations before any GPU generation. Freeze PROTOCOL.md, the hashed image list and the 40-image dev split. Register H1-H6 on OSF. Run the citation search.
- **Why.** Three findings from the adversarial review live here: Week 13 is Nov 8-12; fp16 on a T4 is documented to crash bf16-trained models (UVLM, arXiv 2603.13893: device-side asserts, half a run lost; remedy native bf16); four lmms-eval wrappers hardcode bf16 and other silent defaults. The naive selection rule yields roughly 110-260 images, not 300, so N must be realised before power is claimed.
- **Inputs.** Kaggle (T4 x2, internet on, HF token); lmms-eval at a pinned commit; models Qwen/Qwen3-VL-2B-Instruct, OpenGVLab/InternVL3_5-2B, google/gemma-4-E2B-it, Qwen/Qwen2-VL-2B-Instruct, adapter psp-dada/Qwen2-VL-2B-Instruct-SENTINEL (PEFT LoRA, Apache-2.0, merged with merge_and_unload); dataset lmms-lab-encoder/POPE (9,000 rows, category column). Pod: RunPod community RTX 3090 ($0.22/h) or RTX 4090 ($0.34/h) with a 50 GB network volume. GTX 1060: two venvs on torch==2.14 from the cu126 index (assert sm_61 in torch.cuda.get_arch_list()); diffusers with stable-diffusion-v1-5, its inpainting checkpoint, lllyasviel/control_v11f1p_sd15_depth, LiheYoung/depth-anything-small-hf, IOPaint LaMa, google/owlv2-base-patch16-ensemble, IDEA-Research/grounding-dino-tiny, open_clip ViT-L/14 and SigLIP. CPU: RePOPE JSONs, COCO instances_val2014, captions_val2014, instances_val2017, RUCAIBox/POPE JSONs.
- **Method.** Day 0-1: email the course for the exact due date and template; activate scenario A or B. Day 1-2 (Kaggle): per model, run lmms-eval on 200 POPE items twice (fp16; then fp32 for 2B models or emulated bf16), record NaN/asserts, flip rate with a Wilson interval, seconds per item, peak VRAM, tokens per image. Decision rule: any core model with more than 1% flips or an assert moves all HF headline runs to the pod in bf16; otherwise Kaggle fp16 with both T4s concurrently. Wrapper arguments: qwen3_vl min_pixels=262144, max_pixels=2097152 (300 tokens at 640x480); internvl3_5 max_num=6, use_flash_attn=False; gemma4 max_soft_tokens=280; qwen2_vl defaults; all greedy, temperature 0, max_new_tokens 16 for yes/no. Day 2 (CPU): join POPE adversarial items to RePOPE by image id and object; per-item inclusion (clean = not wrong, not ambiguous; ambiguous kept but flagged and excluded from tests); removable-object rule (one non-crowd instance, mask area 2-25%, at most one border touch, dilated mask overlapping under 5% of any other positive instance, 10% category cap); pre-registered relaxation ladder (two instances; area 1-35%; add COCO val2014 images outside POPE's 500 using POPE's own sampler with your labels). Target 300 images, minimum 250; hash the list and the dev split before any anchor output is opened. Day 3-4 (1060): one LaMa removal, one 640x480 ControlNet-depth twin, one OWLv2 and one Grounding DINO pass; time each. Day 4: PROTOCOL.md v1.0; OSF registration (H1-H6, primary endpoints, tests, cut order). Day 5: citation search (LLM-Check citing works on Semantic Scholar and Google Scholar plus keyword queries: "eigen", "covariance spectrum", "attention kernel", "logit entropy" with VLM/LVLM/MLLM; "POPE abstain", "object hallucination selective prediction", "I don't know option POPE"); log the neighbours named in Section 13.
- **Output.** Written due date and scenario; feasibility note; two frozen requirements files; realised-yield table; PROTOCOL.md v1.0; OSF link; hashed lists; citation-search log.
- **Compute and cost.** Kaggle 4-6 h ($0); GTX 1060 2-3 h ($0); pod setup 1 h (about $0.30).
- **Risk and fallback.** Gemma 4 E2B (5.1B total) cannot run in fp32 on one T4: pod in bf16 or sharded fp32 across two T4s; google/gemma-3-4b-it if Gemma 4 fails everywhere. Transformers version conflicts: one notebook per model family with its own pin. The course answers late: assume scenario A by Oct 16. InternVL3.5 fails: OpenGVLab/InternVL3-2B.
- **Paper section.** 5.1 and Appendix A.
- **Closest prior work.** UVLM (fp16-on-T4 failure); GHOST-Q (precision as a factor); the harness token divergence is reported as "we found no paper that documents it", with the 2026 benchmark-audit papers (arXiv 2607.10240, 2610.13548) as neighbours.

### Step 2. TriPOPE arms R and E, remover variants, blind conditions, negated form, verification, pilot

- **What.** Arm R from COCO val2014 originals (decoded once, saved as PNG). Arm E by LaMa removal of the chosen object at native resolution. SD-inpainting and grey-mask variants for all removed objects. Text-only and grey-image blind conditions. A negated question template. All six labels verified on every E image. Blinded two-rater audit. Pilot discordance on E and S pairs; fix N.
- **Why.** Arm E is the exact-negative probe (object absent by edit) and the edit-drift control. LaMa fills can insert co-occurring textures and dilated masks can clip other positives, so "exact by construction" becomes "absent by edit, human-verified". The three removers turn the ObjectDrop/Remove360 residual-trace criticism into a measured quantity. The text-only blind run (VDD protocol) is the correct prior estimate; a grey image is visual evidence of emptiness, not a prior.
- **Inputs.** The realised image list; COCO masks; IOPaint LaMa; stable-diffusion-v1-5 inpainting (prompt "empty background, same scene", negative "{object}", 30 steps, seed 1234, fp32 VAE), PowerPaint v1 if it installs cleanly; OWLv2 and Grounding DINO with thresholds calibrated on arm R; CLIP and SigLIP; a named second rater; statsmodels, scikit-learn, bayes_evals.
- **Method.** E: dilate the instance mask 10-15 px (more along visible shadows), LaMa at native size, no resize; the removed object's item flips to "no", the other five keep their labels pending verification. Verify all six objects per E image with a whole-image instruction ("is any instance of X visible anywhere?") plus a residual-trace flag (300 images x 15 s). E-sd and E-grey: same mask, removed-object question only, plus all six questions on a random 100-image subset for drift; run both detectors on every inpainted crop for any COCO class (inserted-object flag) and a CLIP region check. Blind-text: question with no image. Blind-grey: mid-grey image of identical size. PN: "Is it true that there is no {object} in the image?" with flipped labels. P1b paraphrase pre-registered. Audit: second rater, blinded to arm and prompt, random ids: all 300 E absence labels, a uniform 200-item sample across arms, 100 false-premise labels later; report raw agreement, Cohen's kappa, PABAK, adjudicated label accuracy with Wilson intervals per arm. Pilot: 35 E images and the first 35 verified S twins through Qwen3-VL-2B under P0 (about 420 pairs); estimate discordance per arm; fix N with the McNemar power formula (alpha 0.05, power 0.8: 1,742 pairs for 3 points at discordance 0.20; 940 pairs for 5 points at 0.30). PNG for all arms; tokens per image per model logged and asserted equal across arms. Licences: arm R as COCO ids plus a rebuild script; arm E images released with per-image attribution plus masks and scripts; RePOPE joined at build time, not redistributed.
- **Output.** Dataset v0 with metadata; Table 1 construction statistics; the power memo with realised N.
- **Compute and cost.** GTX 1060: LaMa under 1 h, SD-inpainting about 2.5 h, detectors 1-2 h; pod or Kaggle pilot 0.5 h; human: you 6-8 h this week, second rater 2 h; $0-0.50.
- **Risk and fallback.** LaMa leaves shadows or clips neighbours: drop clipped images, analyse with and without flagged items. Diffusion remover inserts objects in over 20% of cases: grey-only ablation. Second rater unavailable in the break week: advisor's graduate student.
- **Paper section.** 3.1-3.3, Appendix B.
- **Closest prior work.** BEAF (ECCV 2024; LaMa removals, SB/ID metrics); ROHE (NAACL 2025); POPEv2/HII-DPO (grey masks); Mutze et al. 2025 (detectors overlook inpainted content); VDD (text-only blind POPE); VHTest and AMBER audit conventions.

### Step 3. Arm S: layout-matched SD 1.5 + ControlNet-depth twins at the photo's exact pixel size

- **What.** For 150 of the selected images (minimum 100), generate a twin conditioned on the real photo's depth map so object positions, scales and counts are preserved, at the photo's exact size, with labels verified identical for all six items; log per-item difficulty covariates in every arm.
- **Why.** A pure text-to-image twin is matched only on size and labels: scenes get written around the questions and detector-selected seeds make positives larger and more salient than in the photo, confounding source with difficulty ("MLLMs Know Where to Look", ICLR 2025, shows object size drives accuracy). Depth conditioning preserves layout by construction; symmetric detector criteria and random seed choice remove the selection asymmetry.
- **Inputs.** Depth maps from depth-anything-small; control_v11f1p_sd15_depth with SD 1.5; prompt = longest COCO caption + ", including {pos1}, {pos2}, {pos3}, photograph"; negative prompt = "{neg1}, {neg2}, {neg3}, cartoon, text, watermark"; 30 steps, guidance 7.5, conditioning scale 0.8, seeds 0-1 (2-3 more if none passes); generated at the real size rounded up to a multiple of 8 then centre-cropped; detectors and CLIP/SigLIP as in step 2.
- **Method.** Apply identical pass/fail criteria to R and S (both detectors find all three positives above the calibrated recall threshold; no negative above the R-calibrated negative threshold); among human-passing seeds choose at random; verify all six labels on every chosen twin; report acceptance by seed count, category and area and the real-arm accuracy on kept vs dropped images (selection effect). Covariates in every arm: detector box area of the queried object, number of detected objects, CLIP and SigLIP distance twin-vs-real, train2017-membership flag. Paired tests use label-preserved items; a conditional logistic model stratified by image with the covariates is reported beside the raw paired test. A pure text-to-image sub-arm (100 images) is a gated extension as the literal replication of "AIGCs Confuse AI Too".
- **Output.** Arm S images and metadata; Table 1 S columns; covariate file.
- **Compute and cost.** GTX 1060: about 50 s per 640x480 ControlNet image x 150 x 2-3 seeds = 4-7 h over two nights plus 2 h retries; detectors 1 h; human 2-3 h; $0.
- **Risk and fallback.** ControlNet reproduces the removed-object silhouette so negatives reappear: lower the conditioning scale to 0.5 or switch to canny. Yield below 100: accept S at 100 and report it as powered for 5-point effects only.
- **Paper section.** 3.2, Appendix B.
- **Closest prior work.** Causal-HalBench (FLUX-ControlNet inpainted counterfactuals, one direction, 7B+); AIGCs Confuse AI Too (SD 1.5 re-renders, no size matching); ODE; HaloQuest. No layout-, size- and token-matched photo/generated twin set exists for object existence.

### Step 4. Core behavioural evaluation: source effect, penalty dose-response, blind references, removed-object probe (headline Table 2)

- **What.** Run every core model on R, E and S under P0 (binary), P1 (unpenalized "cannot tell"), P2-lo (t = 0.5, penalty 1) and P2-hi (t = 0.9, penalty 9); on the two blind conditions under P0/P1; and on the remover variants. Score with cluster-aware statistics.
- **Why.** One table answers RQ1 and RQ2: whether source moves accuracy, false-acceptance and yes-ratio once tokens match; whether a stated penalty moves abstention and false acceptance (no VLM paper has run Kalai's instruction as a dose-response); whether rankings change under Effective Reliability; whether a removed-object "yes" exceeds the class-matched real-arm false-acceptance and the text-only prior.
- **Inputs.** Custom lmms-eval tasks (copies of pope.yaml pointing at the HF dataset; identical generation settings and metrics). P0 post-prompt "Answer the question using a single word or phrase." P1 "Answer with yes, no, or cannot tell." P2: "Answer only if you are >{t} confident, since mistakes are penalized {t/(1-t)} points, while correct answers receive 1 point, and an answer of 'I don't know' receives 0 points." followed by the question and "Answer with yes, no, or I don't know." at t = 0.5 and 0.9. max_new_tokens 16; a regex for yes / no / cannot tell / don't know / not sure / unsure / unknown, with truncations and unparsables counted and 100 hand-inspected. statsmodels (mixed-effects logistic or GEE with image clusters, exact McNemar), scipy bootstrap (BCa, 9,999 image-level resamples), bayes_evals.
- **Method.** R and E first (Week 3), S and blinds next (Week 4). Metrics per run x arm x prompt with n and Wilson intervals: accuracy, precision, recall, F1, yes-ratio, FAR (verified-absent items answered "yes"), FAR_removed (arm E flipped items), BEAF-style SB and ID for R-vs-E, coverage, risk, Effective Reliability Phi_c for c in {1, 3, 9}, abstention-precision lift over the P0 error rate, guessing premium GP_c = c x err_P0. Primary tests, one per hypothesis: H1 source effect: mixed-effects logistic correct ~ arm x model + (1 | image), with per-model paired image-cluster bootstrap intervals. H2: FAR_removed(LaMa) minus same-class FAR on R adversarial negatives, cluster bootstrap interval, with E-sd and E-grey differences as remover-specific trace effects and the text-only "yes" rate as the prior row. H3 penalty effect: abstention and FAR under P2-hi vs P1 per arm, McNemar on abstain indicators plus cluster bootstrap. H4 ranking change: Phi_9 vs P0-F1 orderings with the number of pairwise flips. Each Phi_c is reported under its own elicitation and cross-scored (labelled counterfactual). Holm within each hypothesis family only; everything else exploratory. Stratify by removed-object area tercile, class frequency, trace flag, inserted-object flag, train2017 membership and CLIP distance.
- **Output.** Table 2 (models x arms x {Acc_P0, FAR, Yes%, Abstain%_P1, Abstain%_P2-hi, risk, Phi_1, Phi_9} plus blind columns) with intervals; Figure A (paired differences per arm and per remover); Appendix C.
- **Compute and cost.** Pod (bf16): 5 runs x (R 1,800 + E 1,800 + S 900) x 4 prompts = 90,000 items + blinds 36,000 + variants 9,000 = about 135,000 items at about 0.3 s = 11-14 h ($3-5 on a 4090); Kaggle backup about 30 h.
- **Risk and fallback.** Models ignore or over-use the abstain option: both extremes are findings (FAR and coverage collapse); step 6 gives the prompt-independent mechanism; P1b measures wording sensitivity. A null source effect with tight intervals is a legitimate matched replication.
- **Paper section.** 4 and 5.2.
- **Closest prior work.** Seeing without Looking (unpenalized unknown option, 4B+); MM-AQA; Reliable VQA; Kalai et al.; HalluLens; BEAF; Mirage; VDD/M3ID blind baselines; HQM (negated forms).

### Step 5. Open-ended false-premise arm: the image-grounded NonExistentRefusal analogue

- **What.** Prompt "Describe the {object} in this image." for the removed objects on E (absent) and the same objects on R (present control); classify acceptance vs refusal with a rule-based classifier validated on 200 hand labels; report false-acceptance and true-acceptance rates per model; join false acceptances to the hero model's P(yes).
- **Why.** The only genuine visual analogue of HalluLens' NonExistentRefusal (existent class, absent instance) with a control that stops refusal-happy models from winning; it tests whether yes/no results transfer to open-ended generation.
- **Inputs.** 300 E items and 300 R items; max_new_tokens 128, greedy; denial/absence/hedge phrase lists tuned on the dev split; 200 hand labels (100 double-rated); about 95% human agreement as the target (KoNA's level).
- **Method.** Acceptance if the response attributes properties or location to the object as present; refusal if it states absence, hedges or asks for clarification; kappa against hand labels; FAR_open on E and TAR on R with Wilson intervals; per-model McNemar E vs R; join to step 6. If classifier-human agreement is under 90%, cross-check with a batch LLM judge capped at $2 (GPT-5-nano, $0.11 per 1,000 calls), never as the primary number.
- **Output.** Table 4 (FAR_open vs TAR per model, classifier kappa) and the P(yes) join figure.
- **Compute and cost.** Pod: 5 runs x 600 generations x about 1.5 s = 1.5 h ($0.50).
- **Risk and fallback.** Indirect refusals missed; verbose non-answers: judge cross-check within the cap; report the unparsable share.
- **Paper section.** 5.2 (Table 4), 6.
- **Closest prior work.** HalluLens NonExistentRefusal (text); KoNA (attribute-level false premises on present objects); FPCO-Dialog; VOPE; MM-AQA; HaloQuest false premises.

### Step 6. Selection signals on the hero model: softmax baseline, LLM-Check Hidden Score and attention score; risk-coverage, transfer, verbal-vs-logit gap

- **What.** One HF forward pass per item on Qwen3-VL-2B (P0 prompt, preprocessing identical to lmms-eval): cache first-token yes/no logits, hidden states and eager attention; compute three selection functions; draw risk-coverage curves per arm; report AURC, AUROC, coverage at 1/5/10% risk and ECE; transfer a threshold fixed at 5% risk on R to E (flipped and unchanged items separately) and S; quantify the gap between logit-level coverage at a given risk and verbal coverage at equal risk.
- **Why.** Makes LLM-Check load-bearing (hidden-state and attention eigen-signals, not just logits), gives RQ2 a prompt-independent complement, and is the first real-to-synthetic transfer test of any selection signal on matched items. The softmax curve alone is published (HalluPrism: POPE AUROC 0.823; Anatomy of a Lie), so it is labelled as the baseline.
- **Inputs.** Qwen/Qwen3-VL-2B-Instruct on the pod (attn_implementation="eager", output_hidden_states=True, output_attentions=True); arms R/E/S under P0; dev split for the layer choice and sink rule; sklearn metrics; the step-4 parsed answers for a 99% agreement check on 100 items.
- **Method.** Signal 1: softmax confidence = max(P(yes), P(no)) after summing surface forms and renormalising, plus the un-renormalised yes+no mass. Signal 2: Hidden Score = mean log singular value of the prompt hidden-state matrix at layer L (chosen on the dev split), computed separately on the image-token block and the text-token block, after removing visual sink tokens (hidden-state norm above a dev-chosen multiple of the median; results with and without removal). Signal 3: attention-kernel score = mean over heads of the log-eigenvalue statistic of the attention map at layer L, image-image and text-text blocks. Validate argmax agreement with lmms-eval outputs at 99% or better before reporting. Per arm: AUROC for error prediction, AURC, risk-coverage curves, coverage at 1/5/10% risk, ECE on P(yes); transfer (H5): threshold fixed at 5% risk on R, realised risk and coverage on E-flipped, E-unchanged and S with bootstrap intervals; gap = logit coverage at the verbal risk minus verbal coverage, per arm, with intervals; also the Phi_c achievable by thresholding each signal as the reference for the verbal policy. The attention signal has a kill date (end of Week 4); Hidden Score and softmax are core.
- **Output.** Figure 1 (risk-coverage per arm with the three signals, verbal abstention points from P1/P2 and the transferred threshold) and an appendix table.
- **Compute and cost.** Pod: 4,500 forwards with hidden states and attention, about 1 h ($0.40); CPU SVDs seconds each.
- **Risk and fallback.** Signals near chance is a finding: "no cheap signal supports abstention at 2B on matched items". Softmax only if the others fail.
- **Paper section.** 5.3 (Figure 1), 6.
- **Closest prior work.** LLM-Check; Reliable VQA; HalluPrism; Anatomy of a Lie; WEP (cluster bootstrap, AURC); HalluShift++ and EigenTrack (trained); visual attention sink / EAZY (sink removal); Rep2Act and TRAPSBench (prior demonstrations of the encode-but-do-not-express gap).

### Step 7. Mitigation under Mirage and Rethinking-CD controls: VCD and VDD prior contrast on the hero model, SENTINEL vs base (Table 3)

- **What.** From the step-6 cache plus one noised-image pass, compute VCD and a VDD-style prior contrast (blind-text logits) for yes/no answers on R, E and S, alongside greedy, Mirage's output-layer modification forcing the model's majority direction, Rethinking-CD's constant-bias sweep, a magnitude-matched sign-randomised perturbation, an image-space Gaussian-noise control on a 600-item subset, VCD without the plausibility constraint, and the 25/75 and 75/25 imbalance stress test (arm E is a natural 2/6 imbalance); evaluate SENTINEL vs base Qwen2-VL-2B under every condition of steps 4-5.
- **Why.** The title's "Mitigating" clause at near-zero cost and the first run of these controls at 2-3B, on a training-based checkpoint and on edited/generated inputs. A method whose (yes-ratio, accuracy) point lies on the base model's constant-bias curve is a bias shift; a point above it with a paired-bootstrap interval is a real gain.
- **Inputs.** VCD settings from the official repo (alpha 1, beta 0.1, diffusion noise T = 999); OLM from Mirage (force the majority answer when |p(yes) - p(no)| < tau, tau tuned on the dev split to match the method's yes-ratio); constant bias from Rethinking-CD (b swept; b* reported); SENTINEL merged; the step-6 cache.
- **Method.** Per item: l_o (original), l_n (noised), l_b (blind-text). VCD answer = argmax over yes/no of (1 + alpha) l_o - alpha l_n under the plausibility constraint; VDD the same with l_b; controls as listed; random perturbation = sign-randomised (l_o - l_n) on the yes-minus-no axis with per-item magnitude matched (20 draws). Report accuracy, precision, recall, F1, yes-ratio, FAR per arm with intervals; net gain = method minus its yes-ratio-matched control with a paired-bootstrap interval; McNemar method vs greedy and vs OLM; test whether net gain differs by arm (H6). SENTINEL: its point vs the base model's bias curve at the same yes-ratio; informativeness = positive-item recall, coverage under P1/P2, TAR from step 5; AMBER generative scores for the pair on 200 images as a gated extension.
- **Output.** Table 3 and the accuracy-vs-yes-ratio figure with the bias curve.
- **Compute and cost.** Pod: noised pass 4,500 forwards about 0.5 h; SENTINEL runs are inside steps 4-5; controls on CPU; $0.50.
- **Risk and fallback.** Multiple surface forms of the answer token: sum probabilities. PEFT merge mismatch: load the adapter un-merged. If behind, keep only SENTINEL vs base.
- **Paper section.** 5.4.
- **Closest prior work.** VCD; VDD/M3ID; Mirage (Table 3: greedy 87.1 / VCD 88.6 / OLM 89.6); Rethinking-CD (b* = +0.84, noise substitution, imbalance tests; 7-8B only); SENTINEL; Playing it Safe; SAVER; VOPE.

### Step 8. Validity appendix: harness anchor with RePOPE and contamination flags, reliability block, label-noise sensitivity, token-budget 2x2

- **What.** Run pope_adv (3,000 items) for every core run and full POPE for the two fastest; re-score with RePOPE; flag each POPE image as COCO val2017- or train2017-member and report accuracy and FAR by membership; negated-form acquiescence shift and per-item consistency on arm R for the 2026 models; P1b paraphrase agreement; cross-session retest; split-half reliability, design effect, content validity; a token-budget 2x2 on identical content.
- **Why.** Comparability with published tables; a contamination bound without new photos (about 88% of val2014 images sit in train2017 because COCO 2017 re-split the same images); the reliability numbers no 2025-26 benchmark ships; a measurement of the resolution-induced budget artefact on current dynamic-resolution models without conflating it with composition change.
- **Inputs.** lmms-eval pope task with category filtering; RePOPE JSONs; instances_val2017 id list; hero model with min_pixels = max_pixels settable; 50 R/S pairs (300 items).
- **Method.** (a) Anchor with Wilson and cluster-bootstrap intervals, original vs RePOPE labels, rank changes, compared in direction with RePOPE's own Ovis2/PaliGemma shifts; vendor numbers in a separately labelled column. (b) Contamination: accuracy and FAR by membership with clustered intervals; a gap over 3 points triggers the fresh-photo mini-arm (step 9). (c) Negated form on R for the three 2026 models: acc+, acquiescence shift, parallel-forms correlation (exploratory, with its n of models). (d) P1b vs P1 agreement as the prompt-stability statistic. (e) Retest = anchor-R vs custom-R agreement. (f) 100 random image-level split halves at 50/100/150/300 images; intra-image correlation and design effect 1 + 5 rho; content validity = audit survival. (g) Token-budget 2x2 for the hero: {real photo, its twin} x {native, resized to 512x512, centre-cropped 480x480} plus min_pixels = max_pixels at about 256/391/640/1,024 tokens; new square re-renders only as a labelled composition row. State the claim precisely: equal budgets for the fixed-resolution 2023 models of AIGCs Confuse AI Too; a real confound for ODE and all dynamic-resolution models.
- **Output.** Appendix tables A and C and Figure B.
- **Compute and cost.** Pod: anchor about 2.5 h; negated form and P1b about 1.2 h; 2x2 about 0.2 h; about $1.50; CPU for (e)-(f).
- **Risk and fallback.** Time in Week 5: drop full-POPE runs; negated form on the hero only; keep (b), (e), (f), which are CPU-only. HF mirror pixels differ from COCO originals: hash-check 50 images; if different, report the retest as cross-source.
- **Paper section.** 5.5 (one paragraph), Appendices A and C.
- **Closest prior work.** RePOPE; HQM/HQH; ODE and Common-O (contamination); ResPrune, "When Token Pruning is Worse than Random" (CVPR 2026) and Seeing without Looking for pruning-induced curves; Qwen2-VL Table 7.

### Step 9. Gated extensions (each pre-registered with a go/no-go date; none runs before Table 2, Figure 1 and Table 3 are frozen)

In priority order:

- (a) In-engine F16/Q8_0 vs Q4_K_M GGUF pairing of Qwen3-VL-2B on the 1060 (gate end of Week 4; fixed config; prebuilt llama.cpp CUDA binaries or Ollama first; mmproj-Q8_0 for both precisions; -c 2048; headless; 300-item retest to bound nondeterminism; Q4_K_M is not NF4, stated; cite GHOST-Q and Edge Reliability Gap).
- (b) Qwen/Qwen3.5-2B on the pod with flash-linear-attention and causal-conv1d installed, enable_thinking=False, pixel bounds as Qwen3-VL, gate under 1.5 s/item (end of Week 4); thinking-on only on 300 R items at 512 tokens with a repeat-n-gram stop.
- (c) Pure text-to-image S sub-arm (100 images, exact size, no ControlNet) as the literal AIGCs replication (gate end of Week 3).
- (d) S half of the false-premise arm.
- (e) Distribution-matched fresh-photo mini-arm: 60-80 own campus photos, class-histogram matched, no identifiable people, resized to COCO dimensions, P0/P1 only, unpaired cluster-bootstrap intervals (triggered only by a contamination gap over 3 points in step 8b; gate end of Week 4).
- (f) AMBER generative scores (lmms-eval amber_g, max_new_tokens 512, the official LLM-free scorer) for the SENTINEL pair on 200 images (gate end of Week 4).
- (g) BEAF overlap check: FAR_removed on images shared with BEAF's released set as a fourth remover row (gate end of Week 5).
- (h) AutoHallusion synthetic/real splits (BSD-3) on two models.
- (i) Hidden Score and softmax curves for InternVL3.5-2B and Gemma 4 E2B.
- (j) openbmb/MiniCPM-V-4 via a patched minicpm_v wrapper as a third backbone (4-hour cap in Week 1 only).

Deleted from this term: Arabic prompts (a SemEval-2027 Task 6 / ArabicNLP 2027 system paper after the term, reusing TriPOPE), self-grounding boxes (HalluPrism already uses grounding re-checks), thinking mode as a factor beyond (b).

- **Why.** Person-time, not GPU-time, is the binding constraint; explicit gates keep extensions from eating the writing weeks while preserving the cheapest high-value additions.
- **Compute and cost.** 1060: (a) 8-12 h over two nights; pod: (b) 2-3 h, (c) 0.5 h plus 1060 generation 2 h, (d) 0.3 h, (e) 0.3 h plus 3-5 h annotation, (f) 1 h, (g)-(i) 2-3 h; about $2-4 total.
- **Paper section.** Appendix E; a one-line "not run" statement, citing the nearest prior work, for each item that did not run.

### Step 10. Write-as-you-go report, release and submissions

- **What.** Draft Sections 1-3 with Table 1 by the end of Week 3; freeze the three headline exhibits at the end of Week 4 and produce a complete report v1 that satisfies the RES201 template (scenario A); extend to v2 with the appendix in Weeks 5-6; release code and data; execute the venue timeline.
- **Why.** The registrar calendar makes Nov 8-12 a possible due date and puts EACL's Dec 15 inside finals; a complete v1 at the end of Week 4 is the only schedule that guarantees the deliverable under both scenarios.
- **Inputs.** All JSONL outputs; ACL 2027 style files; the course template and rubric (requested in Week 1); EACL 2027 SRW (mentorship Nov 6, direct submission Dec 15); NAACL 2027 SRW (mentorship Nov 16, submission Jan 11, 2027); JURI; the OSF registration.
- **Method.** Main text: one spine, three behaviours (guess / abstain / hallucinate) x three sources (real / edited / generated): Table 1 construction, Table 2 headline, Figure 1 risk-coverage, Table 3 mitigation, Table 4 false premise (if space); Discussion with the Kalai paragraph, the LLM-Check positioning and benchmark-design implications; Limitations (detector circularity mitigated not removed, generator artefacts, COCO contamination bounded by the membership flag, English prompts, two LLM backbone families with InternVL3.5 sharing Qwen3, 2-4B-effective scale, SD 1.5 generator, hedged negative claims). Every proportion with n and a Wilson interval; every "first" with "to our knowledge". Release: GitHub (task yamls, scripts, statistics, pinned requirements, PROTOCOL.md, OSF link) and an HF dataset described as "reproducible" with per-arm terms; pull request to the awesome list.
- **Output.** Report v1 (Nov 8), report v2 (Nov 22), SRW manuscript (Dec 10), public repo and dataset, mentorship requests, JURI submission.
- **Compute and cost.** Writing time, about 40-50 h across Weeks 3-8.
- **Risk and fallback.** Course template incompatible with ACL short: keep the ACL-short text as the core and wrap it in the course template. Advisor review late: freeze v1 regardless. Over-claiming: "to our knowledge" discipline.
- **Paper section.** Sections 1, 2, 6, 7 and all appendices.

---

## 7. Fixed evaluation protocol

These rules make every number comparable and defensible. They become PROTOCOL.md v1.0 in Week 1.

- **Harness.** lmms-eval at a pinned commit with --log_samples and incremental per-item JSONL (custom process_results hook); batch_size 1 (4 if the Day-1 test shows identical outputs). Never mix lmms-eval and VLMEvalKit numbers.
- **Precision.** Every headline number from HF models in bf16 on a native-bf16 GPU (the pod); Kaggle T4 fp16 only for models with under 1% flips and zero asserts on the 200-item check, never mixed with pod numbers in one table without a precision column. The cached-signal script must agree with lmms-eval parsed answers at 99% on 100 items before any number from it is reported. GGUF and vendor numbers only in separately labelled appendix tables.
- **Images and tokens.** All arms PNG at pixel-identical dimensions per image; tokens per sample logged (image_grid_thw.prod()//4 for Qwen, num_patches x 256 with thumbnail stated for InternVL, 280 soft tokens for Gemma 4, server log for GGUF) and asserted equal across arms; family settings fixed (qwen3_vl min_pixels=262144, max_pixels=2097152; internvl3_5 max_num=6; gemma4 max_soft_tokens=280; qwen2_vl defaults); budgets vary only in the 2x2 and the sweep.
- **Decoding.** Greedy (do_sample=False, num_beams=1, temperature 0); max_new_tokens 16 for all yes/no prompts (truncations counted), 128 for the false-premise arm, 512 for AMBER captions with mean length reported; thinking off; seed fixed.
- **Prompts.** P0 "{question}\nAnswer the question using a single word or phrase."; P1 "{question}\nAnswer with yes, no, or cannot tell."; P1b paraphrase with "I don't know" and reordered options; P2 = the Kalai Section 4.2 sentence at t = 0.5 and t = 0.9 plus "{question}\nAnswer with yes, no, or I don't know."; PN negated template with flipped labels; FP "Describe the {object} in this image."; BLIND-text = question only; BLIND-grey = mid-grey image of identical size.
- **Parsing.** Case-insensitive regex; unparsable and truncated outputs counted, 100 hand-inspected; false-premise responses classified by the dev-tuned rule classifier validated on 200 hand labels; LLM judge only as a capped ($2) cross-check if agreement is under 90%.
- **Labels.** RePOPE for R (ambiguous items flagged, excluded from tests); E absent-by-edit with all six labels human-verified plus trace and inserted-object flags (sensitivity with and without); S human-verified for all six items with symmetric detector criteria calibrated on R and random seed choice; blinded second rater on all E absence labels and a 200-item sample; Cohen's kappa, PABAK, Wilson interval on adjudicated accuracy per arm.
- **Dev split.** 40 images / 240 items outside the test set for every prompt, regex, threshold, tau, layer and sink-rule choice; image list and dev split hashed and OSF-registered before any anchor output is inspected.
- **Yes-bias.** Yes-ratio on every run; negated-form acquiescence shift; blind-text prior row; class-matched real-arm FAR as the trace-free reference; in the mitigation arm greedy, OLM, constant-bias sweep, sign-randomised perturbation, image-noise control, VCD without constraint, imbalance stress test; every gain net of its yes-ratio-matched control.
- **Informativeness.** Positive-item recall, coverage beside risk, TAR beside FAR_open, AMBER Cover if run.
- **Metrics.** Accuracy, precision, recall, F1, yes-ratio, FAR, FAR_removed, BEAF SB/ID; coverage, risk, Phi_c = (#correct - c x #wrong)/N at c in {1, 3, 9} (Kalai's rule at c = t/(1-t); Whitehead's Effective Reliability for 0/1 items), guessing premium, abstention-precision lift, ordered rankings with pairwise flip counts; selection: AUROC, AURC, coverage at 1/5/10% risk, ECE, transferred-threshold risk and coverage, logit-vs-verbal coverage gap.
- **Statistics.** Wilson 95% interval on every proportion with n; one pre-registered primary endpoint and test per hypothesis H1-H6: pooled mixed-effects logistic correct ~ arm x model + (1 | image) and per-model paired image-cluster BCa bootstrap (9,999 resamples) with the design effect 1 + 5 rho reported; exact McNemar as the secondary check; Holm within each hypothesis family only; bayes_evals for cells under 300 items; N fixed from the pilot discordance; split-half reliability curve, negated-form acquiescence, P1b agreement and cross-session retest as the reliability block.
- **Reproducibility.** Per-item JSONL with model id, precision, device, token count, prompt id and hash, seed and wall time; task yamls, pinned requirements, PROTOCOL.md, OSF registration and token logs released.

---

## 8. Models, benchmarks and tools

**Core model runs (five).**

- Qwen/Qwen3-VL-2B-Instruct (Apache-2.0): hero model for selection signals, VCD/VDD and the token 2x2; its GGUF variant for the gated quantization extension.
- OpenGVLab/InternVL3_5-2B (Apache-2.0): tiled architecture, Qwen3 LLM; max_num=6, 1,792 tokens logged; fallback OpenGVLab/InternVL3-2B.
- google/gemma-4-E2B-it (Apache-2.0): 2.3B effective / 5.1B total; the non-Qwen backbone; max_soft_tokens=280; bf16 on the pod; fallback google/gemma-3-4b-it.
- Qwen/Qwen2-VL-2B-Instruct base (Apache-2.0) and the same model with psp-dada/Qwen2-VL-2B-Instruct-SENTINEL merged (LoRA, Apache-2.0, ICCV 2025): the aligned-vs-base pair.

**Extension only.** Qwen/Qwen3.5-2B (pod only; its fast kernels need sm_80+); Qwen/Qwen2.5-VL-3B-Instruct (Qwen Research License; optional anchor row); openbmb/MiniCPM-V-4 (4-hour cap). **Not run.** LLaVA-1.5-7B, REVERSE, PAI/Nullu/MaskCD/AIR ports (cited as future work).

**Benchmarks.** TriPOPE (new): 250-300 COCO val2014 POPE images x 6 adversarial items x arms R / E at identical pixel size, plus S (150 images); conditions P0, P1, P1b, P2-lo, P2-hi, PN, BLIND-text, BLIND-grey; remover variants E-sd and E-grey; per-item metadata, covariates, audit statistics and model outputs released. TriPOPE false-premise set (new): 600 open-ended prompts. POPE adversarial split (3,000; full 9,000 for the two fastest models) re-scored with RePOPE and flagged by COCO train2017/val2017 membership. Token-budget subset: 50 R/S pairs. Extensions: AMBER generative (200 images, SENTINEL pair), AutoHallusion splits, BEAF overlap images, LIME-DATA/POPE hard subset. HallusionBench not run.

**Tools.** stable-diffusion-v1-5 + control_v11f1p_sd15_depth (arm S); depth-anything-small (depth maps); SD 1.5 inpainting or PowerPaint v1 (E-sd); LaMa via IOPaint (arm E); OWLv2 and Grounding DINO tiny (pre-filter and inserted-object check); CLIP ViT-L/14 and SigLIP (distance covariate); lmms-eval; statsmodels, scipy, scikit-learn, bayes_evals.

---

## 9. Compute and money estimate

| Resource | Role | Hours | Cost |
|---|---|---|---|
| GTX 1060 6 GB (yours) | LaMa removals, SD 1.5 inpainting variants, ControlNet-depth twins, detectors, CLIP/SigLIP; optional GGUF quantization extension | 25-40 h unattended (setup 2-3; edits and detectors 4-6; twins 6-9 over two to three nights; pure-T2I extension 2; GGUF 8-12 if gated in; debugging 5-8) | electricity only |
| Kaggle free tier (T4 x2) | Day-1 precision and timing triage, smoke tests, pilot, backup for pod outages | 15-40 of the roughly 180 free hours | $0 |
| RunPod community pod (RTX 3090 $0.22/h or RTX 4090 $0.34/h, prices dated 2026-09-27, re-check in the console) | All bf16 HF inference: core steps 4-8 about 18-22 h on a 4090 plus setup, downloads and reruns | 30-45 GPU-hours | $7-15 plus a 50 GB network volume at $0.05-0.07/GB-month (about $3-4); hard cap $30 |
| APIs | Only a capped batch judge cross-check of the refusal classifier | none in the core | $0-2; cap $5 |
| **Total** | | | **$10-25 expected; hard ceiling $40** |

Why not everything on the GTX 1060: it is about 30x slower than a 4090 on prefill (the cost of image tokens), it cannot run bf16, fp16 runs at 1/64 rate, and 4-bit HF inference needs a pinned torch 2.14 stack. Running the 135,000-item core on it would take roughly one to two weeks of wall-clock with no room for reruns. Why not everything on Kaggle: fp16 on a T4 is documented to crash bf16-trained models, the fp32 fallback does not fit Gemma 4 E2B on one T4, and the weekly quota caps Week 3. A $10-20 pod removes both problems. Colab Pro on an L4 (native bf16) is an acceptable substitute if you prefer a subscription; read its price and burn rate on the day. All per-item throughput numbers are estimates until the Day-1 measurement; a 3x slower pod would raise cloud cost toward the cap without changing the design.

---

## 10. Week-by-week schedule and day-one checklist

### Week 1 (Oct 12-18): deadline, triage, pins, yield, protocol, pre-registration

- Mon: email the course for the exact due date, template and rubric; create RunPod and OSF accounts; book the second rater; start the Kaggle triage notebook.
- Tue: finish triage for all five core runs; decide the GPU path; merge the SENTINEL adapter; run the CPU yield query and freeze the hashed image list and dev split.
- Wed: GTX 1060 stack: two venvs on torch 2.14 cu126; one LaMa removal, one ControlNet-depth twin, one OWLv2 and one Grounding DINO pass; time each.
- Thu: PROTOCOL.md v1.0; OSF registration; custom lmms-eval task skeleton.
- Fri: citation and keyword search log; start LaMa edits overnight; compute depth maps.
- Done when: due date known in writing (or scenario A assumed by Oct 16); feasibility note with the precision table and pod decision; realised yield of 250+ images or the relaxation rung recorded; PROTOCOL.md, hashed lists and OSF registration committed; LaMa edits running.

### Week 2 (Oct 19-25; mid-term break Oct 20-22): arms R and E, variants, blind, verification, pilot, anchor

- Build R and E; verify all six labels per E image with trace flags (about 2 h).
- Generate E-sd and E-grey; run both detectors; CLIP region check.
- Build blind-grey images, PN and P1b templates; validate every custom task with --limit 20 on the dev split; tune the regex and phrase lists on the dev split only.
- Pilot 35 E images and 35 S twins through Qwen3-VL-2B under P0; estimate discordance; fix N in OSF.
- Run pope_adv for all runs (full POPE for the two fastest) on the pod; RePOPE re-scoring; membership flag; hash-check 50 mirror images.
- Second rater: all E absence labels; kappa, PABAK, Wilson intervals; publish dataset v0 and Table 1 v0.
- ControlNet twins for 150 images (2 seeds) overnight on the 1060.
- Done when: dataset v0 on HF; Table 1 v0; pilot discordance and N in OSF; anchor rows; S candidates for 120+ images.

### Week 3 (Oct 26-Nov 1): core evaluation on R and E; S verified; Sections 1-3

- Pod: P0, P1, P2-lo, P2-hi on R and E for all runs; blind-text and blind-grey; E-sd and E-grey removed-object items plus the 100-image drift subset; incremental JSONL.
- Statistics scripts; Table 2 R/E columns and the removed-object probe table.
- Arm S: symmetric criteria, random seed choice, verify all six labels on each chosen twin (about 2 h); covariates; selection-effect table; retries overnight.
- Write Sections 1-3.
- Go/no-go (end of week): pure-T2I sub-arm.
- Done when: Table 2 for R and E with tests and intervals; probe table; arm S verified at 100+ images; Sections 1-3 drafted.

### Week 4 (Nov 2-8): S evaluation, false premise, signals, mitigation; freeze; report v1

- Pod: all four prompts on S; false-premise generations on E and R; label 200 responses (second rater 100); Table 4.
- Hero-model HF script: 4,500 forwards with logits, hidden states and eager attention; 99% agreement check; softmax, Hidden Score, attention score; risk-coverage, AURC, AUROC, coverage at risk, ECE, threshold transfer, verbal-vs-logit gap; Figure 1.
- Mitigation: noised-image pass; VCD, VDD, OLM, constant-bias sweep, sign-randomised and image-noise controls, VCD without constraint, imbalance stress test; SENTINEL vs base; Table 3 and the bias-curve figure.
- Freeze Table 2, Figure 1, Table 3, Table 4; write Sections 4-7; assemble report v1 in the course template; send the EACL SRW mentorship request by Nov 6.
- Go/no-go: quantization extension, Qwen3.5-2B, fresh-photo mini-arm (only if the contamination gap exceeds 3 points), AMBER-g, attention score kept or dropped.
- Done when: report v1 complete and submittable by Nov 8; headline exhibits frozen; mentorship request sent.

### Week 5 (Nov 9-15): validity appendix, gated extensions, draft v2, advisor review

- Submit report v1 if the due date is in week 13; otherwise continue.
- Pod: PN on R for the three 2026 models, P1b on R, token-budget 2x2 and the min_pixels sweep; CPU: split-half curve, design effect, content validity, RePOPE and contamination sensitivity, retest.
- Run only the extensions that passed their gates.
- Appendices A-E; draft v2 to the advisor by Nov 13; limitations and "to our knowledge" pass.
- Done when: appendix tables complete; every extension reported or "not run"; draft v2 with the advisor.

### Week 6 (Nov 16-22): revision, release, final report v2, NAACL SRW mentorship

- Incorporate advisor comments; rerun failed cells; final number-provenance check against PROTOCOL.md and the JSONL.
- Release the GitHub repo and HF dataset; pull request to the awesome list.
- Submit the final report v2 (scenario B); NAACL 2027 SRW mentorship request on Nov 16; confirm EACL SRW dates with the organisers.
- Done when: final report submitted; repo and dataset public; mentorship request sent.

### Weeks 7-8 (Nov 23-Dec 6): buffer

- Absorb slipped core tasks first; convert the report into the 4-page SRW manuscript with the appendix as supplementary material; JURI submission; freeze the manuscript by Dec 10 (EACL Dec 15 only if nothing is pending, otherwise NAACL Jan 11, 2027); archive the pod volume, delete the pod, close the cost ledger under the cap; hand-off note for the Arabic follow-up.

### Day-one checklist

- Email the RES201 instructor: exact due date, report template, rubric, whether an ACL-short body plus appendix satisfies the format; note the registrar dates (week 13 = Nov 8-12; last classes Dec 10; finals Dec 13-24).
- Ask the advisor: (1) can a graduate student act as the blinded second rater for about 4 hours in Weeks 2-4; (2) is a $30 cloud spend acceptable; (3) is the ACL-short plus appendix format acceptable for the course report.
- Accounts: Hugging Face token (accept terms for google/gemma-4-E2B-it if gated), Kaggle (phone-verified for GPU, internet on), RunPod (payment method; community RTX 3090/4090 pod with a 50 GB network volume; read the console price), OSF project, GitHub repo.
- Kaggle environment: git clone lmms-eval and record the commit; pin transformers, accelerate, peft; patch dtype in qwen3_vl.py, internvl3.py/internvl3_5.py, gemma4.py; verify that the POPE category column gives 3,000 adversarial items.
- Precision triage on Kaggle T4 (200 POPE items per model, fp16 then fp32 or emulated bf16) for the five core runs; record NaN/asserts, flip rate with a Wilson interval, seconds per item, peak VRAM, tokens per image (expect 300 / 1,792 with thumbnail / 280 / 391).
- Write the decision rule into PROTOCOL.md: any model with over 1% flips or an assert moves the whole HF panel to the pod in bf16; otherwise Kaggle fp16 with both T4s; test batch_size 4 for identical outputs.
- Pod smoke test (1 hour): same lmms-eval commit, bf16, 50 POPE items per model, seconds per item logged.
- CPU yield query: download RePOPE annotations, COCO instances_val2014, captions_val2014, instances_val2017, RUCAIBox/POPE JSONs; implement per-item inclusion and the removable-object rule; print images and clean items per relaxation rung; hash the image list and the 40-image dev split.
- GTX 1060 (Windows): venv A (diffusers, controlnet, IOPaint) and venv B (transformers detectors, open_clip), both with torch==2.14 from the cu126 index and a constraints file; assert sm_61 in torch.cuda.get_arch_list(); download the SD 1.5, inpainting, ControlNet-depth, depth-anything, OWLv2 and Grounding DINO weights; run one LaMa removal, one ControlNet-depth twin at 640x480 (fp16 UNet, fp32 VAE), one detection pass; time each; no llama.cpp build this week.
- Write PROTOCOL.md v1.0 (Section 7 of this document is the skeleton).
- OSF pre-registration: H1 source effect; H2 trace vs prior; H3 penalty effect; H4 ranking change under Phi_9; H5 threshold transfer; H6 mitigation net gain by arm; N pending the Week-2 pilot; extension gates with dates.
- Citation search log: LLM-Check citing works plus the keyword queries; read Seeing without Looking Tables 3 and 5, HalluPrism's model table and BEAF's dataset page; record every neighbour with one line on what it does and does not cover.
- Calendar blocks: annotation sessions (Weeks 2-4), writing sessions from Oct 26, advisor review slot in the week of Nov 9, NAACL SRW mentorship request on Nov 16.

---

## 11. Report outline and venue plan

**Report outline (ACL short body, 4 pages, plus appendix; wrapped in the course template).**

1. Introduction: object hallucination in small open VLMs; evaluations that reward guessing (Kalai et al.); edited and generated images as controlled probes and the uncontrolled token budget in dynamic-resolution models; three questions and four contributions, each with its "new relative to" statement (BEAF, Seeing without Looking, HalluPrism, Mirage/Rethinking-CD named).
2. Related work: benchmarks and critiques; edited and generated probes and the resolution confound; mitigation and critiques; abstention and selective prediction; internal signals; theory backdrop; a compact table with columns: source arms, per-item pairing, size/token control, labels verified, abstention scored, penalty stated, intervals, sub-4B models.
3. TriPOPE: item selection, RePOPE and realised yield; arm E and arm S; verification and reliability statistics; token-budget control, encoding and licences.
4. Penalty-scored evaluation protocol: P0/P1/P2 with the verbatim Kalai instruction and the dose-response design; Effective Reliability and the Kalai rule; FAR, FAR_removed, the false-premise arm; selection signals; statistics and pre-registration.
5. Experiments: 5.1 models, harness, precision policy, token budgets; 5.2 Table 2: guess, abstain or hallucinate across real, edited and generated images; 5.3 Figure 1: selection signals, transfer and the verbal-vs-logit gap; 5.4 Table 3: mitigation, seeing vs guessing; 5.5 validity in one paragraph.
6. Discussion: benchmark design implications (penalty-stated abstain option, matched sources, intervals); the Kalai paragraph; LLM-Check signals as the cheapest abstention gate; largest-effect ranking (source vs prior vs penalty vs mitigation net gain).
7. Limitations. 8. Conclusion and release statement. Appendices A (setup, precision, tokens, harness divergence), B (dataset card, prompts, audit), C (full tables and tests, power memo, reliability, sensitivity), D (parsing and classifier validation), E (gated extensions that ran).

**Venue plan.** Guaranteed: the RES201 report (v1 by Nov 8, v2 by Nov 22), with a parallel JURI submission. Primary paper target: NAACL 2027 Student Research Workshop (mentorship request Nov 16, 2026; submission Jan 11, 2027, after finals). Opportunistic: EACL 2027 SRW (mentorship request Nov 6 with Table 1 and the dataset card; direct submission Dec 15 is inside finals, so only if the manuscript is final by Dec 10; confirm the organisers' mistyped commitment date). If results are strong: an ICLR 2027 workshop (list Nov 29; papers about Feb 1) or a CVPR 2027 workshop (about Mar 2027). Arabic/dialect follow-up: SemEval-2027 Task 6 or ArabicNLP 2027, reusing TriPOPE.

---

## 12. Risks and the cut list

**Top risks and mitigations.**

- Due date is literally week 13 (Nov 8-12): Day-0 written confirmation; the Week-4 checkpoint is a complete report.
- bf16-trained models on T4 fp16 and bf16-hardcoded wrappers: 200-item precision check, uniform-bf16 pod policy, precision column in every table.
- Person-time (about 2,600-3,300 annotation judgments, custom tasks, scripts, writing): five-run core, E before S, S capped at 150, max_new_tokens 16, gated extensions, write-as-you-go from Week 3, a named second rater.
- Item yield below 250 images: per-item inclusion, pre-registered relaxation ladder, Day-2 CPU query, power recomputed on realised N.
- Arm S difficulty confound and selection asymmetry: ControlNet-depth layout matching, symmetric criteria, random seed choice, covariates, stratified conditional model.
- Residual traces, clipped neighbours and inserted objects in E: under-5% overlap rule, six-label verification, inserted-object check, three-remover ablation with and without flags.
- Cluster-naive tests and multiplicity: one primary endpoint and cluster-aware test per hypothesis, Holm per family.
- Contamination confounding R-vs-S: membership flag, "photo vs generated twin" wording, triggered fresh-photo mini-arm.
- Novelty over-claims: explicit positioning, "to our knowledge", the Week-1 search.
- Pod interruptions: incremental JSONL, resume by image id, network volume, Kaggle backup, $30 cap.
- Abstention extremes, chance-level signals, null source effects: all pre-registered as reportable outcomes.
- Licensing: COCO/Flickr terms for E images, SD outputs under OpenRAIL-M, RePOPE unlicensed (joined at build time), Qwen2.5-VL-3B non-commercial (anchor row only).

**Residual risks (accepted).** A literal week-13 deadline makes v1 thinner on S, the false-premise arm and the appendices (labelled preliminary). A full Kaggle fallback for Gemma 4 E2B roughly triples that model's run time. ControlNet may reproduce negative silhouettes or fail to render all positives. Verbal abstention may be near zero and signals near chance at 2B (pre-registered). The mixed-effects model may not converge on near-zero-error cells (the cluster bootstrap is the fallback primary). The second rater's availability in the break week and your exams in term weeks 10-12 are the likeliest causes of slippage; the buffer absorbs at most one week.

**Cut list, in order.**

1. Every step-9 extension in reverse priority.
2. The attention-kernel score (keep softmax and Hidden Score) and the image-noise control (keep the sign-randomised control).
3. P2-lo on arm S and P1b to arm R only; keep P0, P1 and P2-hi on R and E.
4. Arm S to 100 images and the six-question drift check on variants to 50 images.
5. The anchor to pope_adv for all runs; PN to the hero model.
6. The panel to four runs (drop Gemma 4 E2B if it fails the pod path; state "one LLM backbone family" as the first limitation).

Never cut: arms R and E with verified labels at identical pixel size, blind-text and class-matched FAR references, the removed-object probe, P0/P1/P2-hi with Phi_c scoring, softmax and Hidden Score curves for the hero model, RePOPE re-scoring, the blinded audit with kappa, cluster-aware tests with Wilson intervals, token logging, OSF pre-registration, the dataset card, and report v1 at the end of Week 4.

---

## 13. How this plan was built, and what still needs checking

**Process (12 Oct 2026).** Two multi-agent passes. First, a 12-agent literature and feasibility sweep (benchmarks, mitigation, synthetic-data tooling, the bridge from the reading list, the newest 2025-2026 papers and small models, compute and prices, venues) with a completeness critic and four gap-filling sweeps (the resolution confound, an Arabic angle, statistics, where hallucination originates); about 1,100 web lookups, with primary sources opened for every item marked verified. Second, a design pass: four independent strategy drafts from different cores (benchmark-first, controlled-analysis-first, detection-and-theory-first, minimum-risk), three judges (reviewer, engineer, advisor), a synthesis, four adversarial skeptics (novelty with web search, feasibility on Pascal, rigor, advisor-and-paper), and a final revision that applied more than 50 fixes. The skeptics' findings and the changelog are in `research/02`; the literature with URLs is in `research/01`.

**What the adversarial review changed most.** The deadline (Week 13 = Nov 8-12, verified on the registrar's site); the precision policy (fp16 on T4 crashes bf16-trained models, so headline numbers move to a cheap bf16 pod); the item yield (per-item inclusion and a relaxation ladder instead of assuming 300 images); arm S (ControlNet layout matching instead of free text-to-image); the prior reference (text-only blind run, not a grey image); statistics (cluster-aware primary tests, one per hypothesis); novelty wording (BEAF, Seeing without Looking, HalluPrism, Anatomy of a Lie, Rep2Act and TRAPSBench named as neighbours; several "first" claims narrowed); LLM-Check promoted from a relabelled softmax baseline to real hidden-state and attention signals; the Theorem 1 "check" demoted to a discussion paragraph; the main text collapsed to one spine with four exhibits.

**Still unverified (check before relying on them).** HalluPrism's Gemma 4 and Qwen3-VL checkpoint sizes (affects the "first at this scale" wording for selection signals); Seeing without Looking's full text (Tables 3 and 5, Appendix B); whether the lmms-eval wrappers accept a no-image prompt for InternVL3.5 and Gemma 4 (else the text-only prior row is hero-only via the HF script); the exact RES201 due date and template; whether a graduate student can serve as the second rater; the current RunPod console price; Qwen3.5-2B kernel installation on the pod; all per-item throughput numbers until the Day-1 measurement.
