# 4. Evaluation protocol

## 4.1 Models

Both CORE models ran in bfloat16 on one [[NUM:run.gpu_name]] GPU.

- **Qwen3-VL-2B-Instruct** (`Qwen/Qwen3-VL-2B-Instruct`, Apache-2.0) [@bai2025qwen3vl], the general model: `min_pixels = max_pixels = 1,048,576` gave 1,024 visual tokens per 1024-pixel PNG, asserted from `image_grid_thw`. Its training data are not itemised, so its ChestX-ray14 exposure is unknown.
- **MedGemma 1.5 4B** (`google/medgemma-1.5-4b-it`, Health AI Developer Foundations (HAI-DEF) terms, gated) [@sellergren2026medgemma], the medical control, at its fixed 896-pixel input (256 image tokens, asserted). MIMIC-CXR, Chest ImaGenome (sequential MIMIC-CXR images) and an internal Indian chest X-ray set are named as training data; ChestX-ray14 appears only as an evaluation set in the technical report and is listed without a use marker in the card's data-ownership list; RANZCR CLiP is not mentioned anywhere. Its same-patient sequential training data are relevant to a within-patient benchmark.
- [[TODO: include only if PLUS ran]] **PLUS:** CheXOne (`StanfordAIMI/CheXOne`, CC BY-NC 4.0 per the Hub card) [@stanfordaimi2026chexone; @zhang2026reasoning] and its base, Qwen2.5-VL-3B-Instruct (`Qwen/Qwen2.5-VL-3B-Instruct`, Qwen Research License) [@bai2025qwen25vl], both at `min_pixels = max_pixels = 512*512` (about 324 visual tokens). CheXOne was exposed to ChestX-ray14 pixels; RANZCR CLiP is not among its 30 listed training sources.

## 4.2 Prompts and blind rows

P0, the default binary template of Lotfinia et al. [@lotfinia2026vision] with the device phrase `an endotracheal tube`, asks for a single word, Yes or No (Appendix B gives every string). P1 asks the same question with the options `yes, no, or I don't know`. P2hi prefixes P1 with Kalai et al.'s confidence-target instruction [@kalai2025why; @kalai2026evaluating], with t substituted as a percentage: at $t = 0.9$, a wrong answer costs 9 points, a correct one earns 1 and `I don't know` earns 0. P1 and P2hi thus differ only by the penalty sentence. Test items received P0, P1 and P2hi, and development items P0 alone; P2lo ($t = 0.5$, penalty 1) was PLUS only.

BLIND-text (the prompt alone) and BLIND-grey (a uniform mid-grey 1024-pixel PNG) measured the answer prior under P0, P1 and P2hi. Each is one deterministic greedy generation per model and prompt, reported as its first-token P(yes) and greedy answer without an interval (reference lines in Figure 1).

## 4.3 Decoding and the first-token signal

Decoding was greedy, with at most 16 new tokens, repetition penalty 1.0, a fixed seed, no system prompt and each model's own chat template, with the image placed before the prompt text in a single user turn. From the raw logits at the first generated position (not the processed scores), we computed

$$z = \log \sum_{i \in Y} p_i - \log \sum_{j \in N} p_j ,$$

where $Y$ and $N$ hold the single-token ids of six yes and six no surface forms (Appendix B); coverage is the first-token mass on $Y$ and $N$. A 20-image smoke test on development images under P0 asserted the visual-token counts and required a median coverage of at least 0.90 per model, failing which $z$ would have come from the teacher-forced full-sequence log-likelihoods of "Yes" and "No", each closed by the model's end-of-turn token (outcome in Appendix B).

## 4.4 Parsing

After normalising quotes, markdown emphasis and whitespace, an output beginning with yes or no gave that answer, an abstention phrase anywhere gave abstain, and anything else was unparsable (Appendix B). Outputs reaching 16 tokens without an end-of-sequence token were truncated. Unparsable and truncated outputs were excluded from all denominators and reported as rates per model, prompt and arm, with a sensitivity row counting them as wrong. A run with more than 10% unparsable outputs under P0 was flagged in Table 2.

## 4.5 Metrics

Over parsable items, with n counting yes, no and abstain (an abstention stays in the denominator and is never correct): hit_A = P(yes | A), FAR_B = P(yes | B), FAR_U = P(yes | U) and BA = (P(yes | A) + P(no | B)) / 2. The yes-rate and the abstention rate pool A, B and U. Persistence = P(yes on B | yes on A), over pairs with A answered yes and B parsable; pair-correct = P(yes on A and no on B), over pairs with both parsable. Effective Reliability [@whitehead2022reliable] is ER_t = (correct $-$ c $\times$ wrong) / n over A, B and U, with $c = t/(1-t)$ ($c = 9$ at $t = 0.9$), abstentions scoring 0 and correct meaning yes on A or no on B and U; it equals Kalai et al.'s score at confidence target t. Under P0, we report the AUROC of z for A versus B and for A versus U (Mann-Whitney form [@hanley1982meaning]); because z does not depend on the parse, the AUROC uses every retained item with a finite z.

## 4.6 Hypotheses and primary tests

Each hypothesis had one primary test per model; a Holm family is the per-model primaries of one hypothesis over the two CORE models.

- **H0, floor gate** (P0, after exclusions), evaluated in this order: NO-GO if BA $< 0.60$, or yes-rate $< 0.10$ or $> 0.90$; otherwise GO if BA $\geq 0.70$, hit_A $\geq 0.70$ and $0.25 \leq$ yes-rate $\leq 0.75$; otherwise CONDITIONAL. It is a classification without a test, reported either way.
- **H1, co-occurrence prior**: the P0 yes-rate on U with at least one other labelled line minus U with none; two-sided, with a patient-clustered BCa interval (resampled within strata) and CI-inversion p. The pre-specified substitute (fewer than 30 no-line patients) compares two or more other lines with exactly one.
- **H3, penalty effect**: the exact McNemar test on the abstain indicator, P2hi versus P1, over B and U items parsable under both (the number dropped is reported). Whenever any patient contributes two pairs (the two-pairs-per-patient rung, or an 80-pair rung that needs second pairs), the clustered BCa interval of the paired difference is primary, with CI-inversion p, and McNemar on one random B per patient (with every U item) is a check.
- **H5, detector-gated answers**: the BA gain on A and B of the fused rule ($\beta = 1$) over $\beta = 0$, tested by the one-sided shuffled-expert permutation p (Section 4.7); its clustered BCa interval is descriptive.
- [[TODO: include only if PLUS ran]] **PLUS H2, general versus medical**: one primary endpoint, the paired difference CheXOne minus Qwen2.5-VL-3B in the hallucination-free rate (1 $-$ FAR on B and U, over items parsable under both models), and one secondary, the paired difference in BA on A and B; both two-sided, with patient-clustered BCa intervals and CI-inversion p. PLUS results never change a CORE result.

## 4.7 Arm 1: detector-gated answers

Arm 1 follows CCD's expert-informed stage [@zhang2026ccd], which adds a clipped expert log-odds bias to the token logits. The expert, MedSigLIP-448 (`google/medsiglip-448`, HAI-DEF terms) [@sellergren2025medgemma], was never an evaluated model; s is its zero-shot probability of `a chest x-ray with an endotracheal tube` against `a chest x-ray without an endotracheal tube` (softmax over the pair), and

$$z' = z + \beta \cdot \operatorname{clip}\left(\log \frac{s}{1-s}, -\gamma, \gamma\right), \qquad \gamma = 5,$$

with the answer yes if $z' > 0$ (s is clipped to $[10^{-6}, 1 - 10^{-6}]$ before the log-odds). All rows used the same P0 arrays of A, B and U; items without a finite z or s were dropped and counted:

- greedy text, with its agreement with the $\beta = 0$ rule (an abstention counts as a disagreement);
- baseline $\beta = 0$ (yes if $z > 0$) and primary $\beta = 1$, fixed in advance;
- sensitivity: $\beta \in \{0.5, 1, 2\}$ chosen by BA on the development pool (ties to the value nearest 1);
- detector alone: yes if $s > \tau$, with $\tau$ at the Youden point [@youden1950index] on the development pool;
- shuffled expert: s permuted across the same A and B images 9,999 times, with one-sided $p = (1 + \#\{\text{permuted gain} \geq \text{observed gain}\}) / (1 + 9999)$;
- matched constant bias: $z + b$ over 401 values of $b$ in $[-10, 10]$, read where its yes-rate on A and B is closest to the fused rule's (ties to the smallest $|b|$);
- disagreement-abstain: abstain when the sign of z disagrees with the detector's call ($s > \tau$), with coverage and risk among answered items;
- [[TODO: include only if PLUS ran]] PLUS oracle rows: "oracle label" (BA = 1 by construction) and "clipped oracle" (s set to the label, so z moves by exactly $\pm\gamma$), with the share of items with $|z| > \gamma$, to show how far the rule caps any expert.

Rows report BA, its difference from $\beta = 0$ on the same items, hit_A, FAR_B, FAR_U (U scored by the same rule), yes-rate over A, B and U, and n. CCD reports a random-signal expert test but neither an expert-alone baseline nor a yes-rate-matched bias control, which this arm adds; null or negative results are reported against the controls.

## 4.8 Statistics

Every cell reports n and a 95% interval: Wilson intervals [@wilson1927probable] for rates, and a patient-clustered BCa bootstrap [@efron1987better; @davison1997bootstrap] for BA, ER_t, AUROC and every difference (9,999 resamples of NIH patients, so a pair's two images move together, resampled within the pair and U patient sets when a statistic uses both; cluster-jackknife acceleration; percentile fallback). The bootstrap p-value was the smallest $\alpha$ on a grid from 0.001 to 0.5 at which the $(1-\alpha)$ BCa interval excluded zero, from the same resamples. Paired indicators on identical items used the exact McNemar test [@mcnemar1947note]. Holm's procedure [@holm1979simple] ran within each family at $\alpha = 0.05$, with raw and adjusted p beside the interval. Cells with $n < 30$ received an interval and no test. We report the design effect, $1 + (\bar{m} - 1)\,\mathrm{ICC}$ [@kish1965survey], and fitted no mixed-effects model.

With 120 pairs, a Wilson interval at $p = 0.5$ has a half-width of about 0.09, and two U strata of 60 detect a difference of 0.25 with about 80% power (0.20 with about 60%). The U strata stay at 60 and 60 under every rung, so the H1 power does not change under the 80-pair rung. At 200 pairs (PLUS), the half-width is about 0.07, and strata of 100 detect 0.20 with about 83% power.

## 4.9 Pre-specification

The protocol, with every hypothesis, primary test, exclusion rule and fallback, was frozen in the project's private git repository before any model output existed ([[TODO: freeze commit hash and date]]); later changes were dated in its change log. The study was not publicly preregistered, so the freeze date rests on the repository history rather than on an independent registry.

## 4.10 Ethics and data terms

No IRB determination was requested. The study used only public, de-identified data released for research: no person was contacted, no identifiable information exists in either release, and the only human input was the student's own reading of the public images. Software tests before the protocol was frozen produced model or detector outputs on 11 public NIH images; those images were excluded from item selection.

RANZCR files (labels, JPEGs, identifiers and anything derived from them row by row) stayed in private Kaggle notebooks and on the student's machine; the rented GPU machine held only NIH PNGs and prompt text, and nothing RANZCR-derived reached it or the code repository. MedGemma and MedSigLIP were used under the HAI-DEF terms (accepted on 2 November 2026), with no clinical use and no redistribution of weights. [[TODO: include only if PLUS ran]] Qwen2.5-VL-3B (Qwen Research License) and CheXOne (CC BY-NC 4.0) were used for evaluation only. No online LLM API and no PhysioNet data were used. All outputs are research measurements, not for clinical use.
