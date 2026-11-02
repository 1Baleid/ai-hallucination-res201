# Appendix B. Prompts, parser, record schema and runner settings

## B.1 Prompts

Every prompt came from `cxrdev/prompts.json` (version string `PROTOCOL v0.3 (2026-10-12)`). The template's `{device}` slot was filled with the device phrase; every CORE item used the endotracheal-tube phrase. Each run record stores the exact prompt text and the SHA-256 of its UTF-8 bytes; the hashes below were computed from the strings shown.

**P0** (CORE; binary). SHA-256 `972e9fbdaf3b7ce5c47892b04d55e01518b014285e1d4a678d380e62d896abf0`.

```text
Is an endotracheal tube present in this chest X-ray? Answer with a single word: Yes or No.
```

**P1** (CORE; unpenalised abstention). SHA-256 `de1ee68d42b7d6031c6b303ae3363f6f613a6f2866502469146893f4164b9714`.

```text
Is an endotracheal tube present in this chest X-ray? Answer with yes, no, or I don't know.
```

**P2hi** (CORE; $t = 0.9$, penalty 9). SHA-256 `0561672cfd2f308742ff1c8aa41a0587283d2595f376b364339d7b42ee5429e5`.

```text
Answer only if you are > 90% confident, since mistakes are penalized 9 points, while correct answers receive 1 point, and an answer of "I don't know" receives 0 points. Is an endotracheal tube present in this chest X-ray? Answer with yes, no, or I don't know.
```

**P2lo** (PLUS only; $t = 0.5$, penalty 1). SHA-256 `ac356a34e8a174e93e6496ced862023ab8b6fb9b2a3c4ac510090342e9a5a2b6`.

```text
Answer only if you are > 50% confident, since mistakes are penalized 1 point, while correct answers receive 1 point, and an answer of "I don't know" receives 0 points. Is an endotracheal tube present in this chest X-ray? Answer with yes, no, or I don't know.
```

P0 was checked on 2026-10-12 against `build_prompt()` in `data_loader/build_utils.py` of the Lotfinia et al. repository (github.com/mahshadlotfinia/causal, MIT licence) [@lotfinia2026vision]. Its default variant inserts the display phrase where our template has `{device}`; the repository's "brief" and "clinical" variants were not used. P1 carries the same abstain option as P2hi and P2lo, so the P2-versus-P1 contrast isolates the confidence-target sentence.

The first sentence of P2hi and P2lo is Kalai et al.'s confidence-target instruction (arXiv 2509.04664 v1, Section 4.2) [@kalai2025why] with t substituted as a percentage; the preprint writes the target symbolically. The penalty is $t/(1-t)$: 9 points at $t = 0.9$ and 1 point at $t = 0.5$. The version of record [@kalai2026evaluating] gives its rubric strings in Extended Data Table 2. [[TODO: compare those strings with the preprint sentence through the KFUPM library before quoting them]]

Device phrases in `prompts.json` (only `ett` was used in CORE; the other three belong to the PLUS four-question block):

| Key | Device phrase |
|---|---|
| `ett` | `an endotracheal tube` |
| `ngt` | `a nasogastric or other enteric feeding tube` |
| `cvc` | `a central venous catheter, including a PICC line,` |
| `swan` | `a Swan-Ganz catheter` |

Prompt sets: A, B and U items received P0, P1 and P2hi; development items received P0 only. Blind rows ran P0, P1 and P2hi once per model under BLIND-text (no image) and BLIND-grey (a $1024 \times 1024$ 8-bit greyscale PNG with every pixel at 128, generated on the GPU machine). BLIND-demo (PLUS) prefixed the question with `This is an {AP/PA} chest radiograph of a {age}-year-old {sex}.` filled from the NIH metadata.

The Arm 1 expert used the two MedSigLIP texts below, with a softmax over `logits_per_image` across the pair; s is the probability of the first text.

```text
a chest x-ray with an endotracheal tube
a chest x-ray without an endotracheal tube
```

## B.2 Answer parser (version 2)

Normalisation, in order:

1. map U+2018 and U+2019 to `'`, and U+201C and U+201D to `"`;
2. remove `*`, `_` and backticks (markdown emphasis and code marks);
3. collapse every run of whitespace to one space and strip both ends.

Stage 1 (yes or no) applies `re.match` with the case-insensitive pattern below; the captured word, in lower case, is the answer.

```text
^\W*(yes|no)\b
```

Stage 2 (abstain) applies `re.search` with the case-insensitive abstain lexicon below, anywhere in the normalised output.

```text
(cannot tell|can't tell|don't know|do not know|not sure|unsure|unknown|cannot determine|can't determine|unable to determine|not possible to (tell|determine))
```

Stage 3: any other output is `unparsable`.

An output is truncated when it ended without an end-of-sequence token and generated at least `max_new_tokens` (16) tokens. Unparsable and truncated outputs were excluded from the denominators of hit_A, FAR_B, FAR_U, balanced accuracy, ER_t and the abstain indicator, and reported as their own rates per model, prompt and arm. A sensitivity row counted them as wrong. H3 used only items parsable under both P1 and P2hi and reported the number dropped. A run with more than 10% unparsable outputs under P0 was flagged in Table 2. One hundred outputs were checked by hand against the parser ([[NUM:parse.handcheck_agree]] of 100 agreed); no LLM judge was used.

Examples (input before normalisation, and the parser's output):

| Raw output | Parsed | Rule |
|---|---|---|
| `Yes.` | yes | stage 1 |
| `**No**, there is no tube.` | no | emphasis removed, then stage 1 |
| `I don’t know.` | abstain | curly apostrophe normalised, then stage 2 |
| `Not sure.` | abstain | stage 1 fails at the word boundary; stage 2 |
| `Unknown` | abstain | stage 2 |
| `There is an endotracheal tube.` | unparsable | no rule applies |
| `Nope.` | unparsable | no rule applies |
| `Yes or No` | yes | stage 1; an echoed answer line is a known limit, which the hand-check looks for |

## B.3 Run record

The GPU machine wrote one JSON line per item $\times$ prompt $\times$ condition, with the fields below in this order (`S7_FIELDS` in `pod/common.py`). Arm, pair, patient and label fields were never written there; they were joined locally from `items.csv` at scoring time.

| Field | Type | Content |
|---|---|---|
| `run_id` | str | Run identifier |
| `model_key` | str | Model key in `cxrdev/constants.py` (for example `qwen3vl2b`, `medgemma15`) |
| `model_id` | str | Hugging Face Hub id |
| `revision` | str | Hub commit hash the weights were loaded from (resolved before loading) |
| `transformers_version` | str | transformers version on the GPU machine |
| `torch_version` | str | torch version on the GPU machine |
| `dtype` | str | Weight dtype (bfloat16) |
| `device` | str | torch device string (for example `cuda:0`); the GPU name is in the run's metadata file |
| `item_id` | str | Opaque 10-character item id; `BLIND-TEXT` or `BLIND-GREY` for blind rows |
| `nih_filename` | str | NIH PNG filename; empty for blind rows |
| `image_sha256` | str | SHA-256 of the PNG file bytes (of the generated grey PNG for `BLIND-GREY`; empty for `BLIND-TEXT`) |
| `condition` | str | `image`, `BLIND-text`, `BLIND-grey` or `BLIND-demo` |
| `prompt_id` | str | `P0`, `P1`, `P2hi` or `P2lo` |
| `prompt_sha256` | str | SHA-256 of the UTF-8 prompt text |
| `prompt_text` | str | Prompt text sent to the model |
| `visual_tokens` | int | Visual tokens of the image input (0 for `BLIND-text`) |
| `seed` | int | Seed passed to `torch.manual_seed` before each generation (2026101211) |
| `max_new_tokens` | int | Generation limit (16) |
| `raw_output` | str | Generated tokens up to and including the first end-of-sequence token, decoded without special tokens |
| `n_generated_tokens` | int | Number of generated tokens, up to and including the first end-of-sequence token |
| `ended_with_eos` | bool | Whether generation ended with an end-of-sequence token |
| `parsed` | str | `yes`, `no`, `abstain` or `unparsable` (the parser of B.2; recomputed locally in `parsed_results.csv`) |
| `parser_version` | str | Parser version |
| `logp_yes` | float | Log of the first-token probability mass on the yes forms; under `full_sequence`, the log-likelihood of the completion "Yes" closed by the end-of-turn token |
| `logp_no` | float | Log of the first-token probability mass on the no forms; under `full_sequence`, the log-likelihood of "No" closed by the end-of-turn token |
| `logp_abstain` | float | Log of the first-token mass on abstention first words (exploratory, used in no analysis; always the first-token value; NaN when no form is a single token) |
| `z` | float | `logp_yes - logp_no` (a first-token or a full-sequence difference, as `signal` says) |
| `coverage` | float | `exp(logp_yes) + exp(logp_no)`: the first-token mass on the forms, or under `full_sequence` the probability of the two complete answers |
| `signal` | str | `first_token` or `full_sequence` |
| `yes_ids` | list of int | Token ids of the yes forms that encode to one token |
| `no_ids` | list of int | Token ids of the no forms that encode to one token |
| `wall_s` | float | Wall-clock seconds for the record (image loading, generation and signal) |
| `timestamp_utc` | str | ISO 8601 UTC timestamp ending in `Z` |

`parsed_results.csv` keeps every field except `yes_ids` and `no_ids`, recomputes `parsed` with the current parser, updates `parser_version` and adds `truncated` (bool). The item list sent to the GPU machine held only `item_id`, `nih_filename`, `prompt_set` (`full` for P0, P1 and P2hi; `p0` for P0 only) and `smoke` (0 or 1): no arm, label, RANZCR identifier or patient column.

## B.4 Runner settings

| Setting | Value |
|---|---|
| Loading | `AutoModelForImageTextToText` and `AutoProcessor` at a resolved Hub revision (transformers [[NUM:run.transformers_version]], torch [[NUM:run.torch_version]]) |
| Weights | bfloat16 on one [[NUM:run.gpu_name]] GPU (torch device [[NUM:run.device]]) |
| Hub revisions | Qwen3-VL-2B [[NUM:run.qwen3vl2b.revision]]; MedGemma 1.5 4B [[NUM:run.medgemma15.revision]]; MedSigLIP-448 [[NUM:run.medsiglip.revision]] |
| Chat format | The model's own chat template with the generation prompt; one user turn holding the image first, then the prompt text; no system prompt |
| Decoding | `do_sample=False`, `num_beams=1`, `repetition_penalty=1.0`, `max_new_tokens=16`; the sampling fields of the checkpoint's generation config (temperature, top-k, top-p and similar) cleared; no thinking or reasoning instruction |
| Logits | `output_logits=True`, `return_dict_in_generate=True`; the signal uses the raw first-step logits, never the processed scores |
| Seed | 2026101211 (`derive_seed("model")` = master seed 20261012 $\times$ 100 + 11), set with `torch.manual_seed` before each generation and recorded in every row |
| Image decoding | PNG bytes decoded, converted to 8-bit greyscale and then to three-channel RGB (`convert("L").convert("RGB")`); `image_sha256` is taken over the file bytes |
| Qwen3-VL-2B input | `min_pixels = max_pixels = 1,048,576` (set as the image processor's `size` = {`shortest_edge`, `longest_edge`} and read back) on the $1024 \times 1024$ PNG; 1,024 visual tokens, asserted from `image_grid_thw` on every record |
| MedGemma 1.5 4B input | Fixed $896 \times 896$ processor size with pan-and-scan off (both checked); 256 image tokens, asserted on every record |
| PLUS input (CheXOne, Qwen2.5-VL-3B) | `min_pixels = max_pixels = 512*512`; about 324 visual tokens; CheXOne in instruct mode with the same prompts and `max_new_tokens`. CheXOne's Hub metadata names Qwen2.5-3B-Instruct as its base, while its paper names Qwen2.5-VL-3B (a metadata discrepancy) |
| MedSigLIP-448 | `AutoModel` and `AutoProcessor`, float32 (the runner's default, as queued); each image resized to $448 \times 448$ by bilinear interpolation without antialiasing (the model card's resize) before the processor's rescaling and normalisation; s = softmax over `logits_per_image` across the two texts; never an evaluated model; Youden threshold $\tau$ = [[NUM:detector.tau]] on the development pool |
| Smoke test | 20 development images (10 ETT present, 10 absent) under P0 per model; median coverage of at least 0.90; visual-token assertion; seconds per item |
| Signal fallback | If the smoke coverage failed: teacher-forced log-likelihood of "Yes" and of "No", each closed by the end-of-turn token and appended to the generation prompt (two passes per item), recorded as `signal = full_sequence` |
| Smoke-test outcome | Median coverage: Qwen3-VL-2B [[NUM:smoke.qwen3vl2b.coverage_median]], MedGemma 1.5 4B [[NUM:smoke.medgemma15.coverage_median]]; signal used: Qwen3-VL-2B [[NUM:smoke.qwen3vl2b.signal]], MedGemma 1.5 4B [[NUM:smoke.medgemma15.signal]] |
| Time per item | Median seconds per image record: Qwen3-VL-2B [[NUM:run.qwen3vl2b.s_per_item]] s; MedGemma 1.5 4B [[NUM:run.medgemma15.s_per_item]] s |
| Resume key | (`item_id`, `prompt_id`, `condition`); a file written under another run id, model, revision, signal, library version or prompt hash is refused |

## B.5 Yes and no surface forms

The surface forms come from `cxrdev/constants.py`:

```python
YES_FORMS = ("Yes", "yes", " Yes", " yes", "YES", " YES")
NO_FORMS = ("No", "no", " No", " no", "NO", " NO")
ABSTAIN_FIRST_FORMS = ("I", " I", "Cannot", "cannot", " cannot", "Unknown", "unknown", " unknown")  # exploratory only
```

For each tokenizer, only the forms that encode to exactly one token entered the signal; their ids were written to every record (`yes_ids`, `no_ids`), and the skipped forms were logged. With $\ell$ the log-softmax of the raw first-step logits, $\text{logp\_yes} = \operatorname{logsumexp}_{i \in Y} \ell_i$, $\text{logp\_no} = \operatorname{logsumexp}_{j \in N} \ell_j$, $z = \text{logp\_yes} - \text{logp\_no}$, coverage $= e^{\text{logp\_yes}} + e^{\text{logp\_no}}$, and the normalised probability of yes used for the blind rows is $1 / (1 + e^{-z})$.
