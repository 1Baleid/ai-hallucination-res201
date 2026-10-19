# Day 0 email drafts (2026-10-12, revised after the ethics review)

Three drafts for the student to check, complete the `[...]` fields, and send. Keep each short; the recipients are busy. Send order: instructor and advisor tonight; IRB request at 08:00 on 2026-10-13 with the advisor in copy, before any notebook runs; log the send time in `PROTOCOL.md` section 10.

---

## 1. To the RES 201 instructor / URO contact

Subject: RES 201 final report: due date, template and page limit

Dear [Dr. / Mr. / Ms. NAME],

I am [NAME], [ID], ICS, enrolled in RES 201 under Dr. Muzammil Behzad on the project "Benchmarking and Mitigating Object Hallucination in Multimodal Large Language Models Using Synthetic and Real Visual Data". I am preparing the individual paper-format report and would like to confirm five points so I submit the right document:

1. Is the final report due on 21 October 2026, and through which channel (email, Blackboard, other)?
2. Is there a required template or format (for example an ACL/IEEE style), and what is the page limit?
3. Do appendices and references count toward the page limit?
4. Is a rubric available, so I can check the report against it?
5. The course email mentions Week 13 for the deadline; should I treat 21 October or Week 13 (8-12 November) as the date for the graded report?

Thank you for your help.

Kind regards,
[NAME], [ID], [EMAIL], [PHONE]

---

## 2. To the advisor (Dr. Muzammil Behzad)

Subject: RES 201: Stage 1 scope for your sign-off and five short questions

Dear Dr. Behzad,

Following the project listing for "Evaluating VLMs/LLMs Hallucination in Domain-Specific Tasks", here is the scope I plan to execute and report by 21 October, for your sign-off.

Stage 1 scope (one paragraph): a chest X-ray object-hallucination benchmark for support devices (endotracheal tube first) built from public NIH ChestX-ray14 images matched to the RANZCR CLiP labels, using same-patient device-present / device-absent study pairs as the negative stimulus instead of edited images, plus matched unpaired negatives and blind-text prior rows. I will score two open models (Qwen3-VL-2B and MedGemma-1.5-4B) under a binary prompt, an unpenalised "I don't know" prompt and a penalised abstention prompt (Kalai et al., Nature 2026), and test one grounding-based mitigation (MedSigLIP-gated answers) against detector-alone, shuffled-expert and constant-bias controls. If time allows, I will add CheXOne against its own base model, Qwen2.5-VL-3B, as a paired test of the general-versus-medical question in your 2026 framework paper. Everything is pre-registered and reported with confidence intervals; the longer program (synthetic probes, retrieval, region grounding, a training stage) continues after the course.

Five questions:

1. Ethics: I will send a determination request to IRB@kfupm.edu.sa tomorrow morning for secondary analysis of public de-identified images, with you in copy. May I name you as the supervisor and PI of record, and does the JRCAI umbrella project already hold an IRB number under which I could be added?
2. Is there a lab member with radiology familiarity who could do a 100-item blinded reading (about one hour) later in the term? I have planned without one.
3. May I spend about $15 on cloud GPU time (RunPod) for the model runs?
4. Do you endorse the design above (chest X-ray devices, same-patient pairs, the CheXOne-versus-Qwen2.5-VL-3B pair as a two-sided test)?
5. For a possible training stage after the course: is cluster or cloud training access available, with which GPUs, hours and data rules?

I will send you the draft report on 19-20 October.

Kind regards,
[NAME], [ID], [EMAIL]

---

## 3. To the KFUPM IRB (IRB@kfupm.edu.sa), send 2026-10-13 08:00, cc muzammil.behzad@kfupm.edu.sa

Subject: Request for determination: secondary analysis of public de-identified chest radiographs (RES 201 undergraduate research; supervisor and PI of record Dr. Muzammil Behzad)

Dear IRB Office,

I am [NAME], [ID], an undergraduate in the Information and Computer Science department, enrolled in RES 201 under Dr. Muzammil Behzad (ICS; JRCAI project "Evaluating VLMs/LLMs Hallucination in Domain-Specific Tasks" [umbrella IRB number if known: ...]), who is in copy as supervisor and PI of record. I am writing to request a determination that the activity below is not research on a human subject within the meaning of the Law of Ethics of Research on Living Creatures or, if the office considers it in scope, expedited review under the NCBE Implementing Regulations (v3), Article 10.19 (minimal risk; previously collected, non-identifying records), with exemption from periodic follow-up under Article 10.33 (publicly available, non-identifying data).

Activity: secondary analysis of two public, de-identified chest radiograph datasets released by their custodians for research: NIH ChestX-ray14 (NIH Clinical Center, 2017) and the image-level labels of the RANZCR CLiP dataset (Kaggle, 2021; Tang et al., Scientific Data 2021). The work consists of (a) matching the two public releases of the same radiographs by image content, which is the only linkage in the study, (b) automated inference by open vision-language models on these images, (c) my own non-clinical reading of a few hundred images to check the presence of a medical device against the published labels, and (d) statistical analysis. No patients are contacted, no identifiable information exists in either release or is created, no re-identification is attempted, no clinical decisions are involved, and no new data are collected from human participants in this stage. A later stage may use the same public images to fine-tune models, and may involve a short blinded reading by a lab member; I will inform the office before any such session.

Planned start: 13 October 2026, automated analysis of public de-identified images only, under my supervisor's responsibility. My own reading sessions will pause if the office objects.

Data handling: images and labels are used under their published terms; no label files are redistributed; model inference runs on a rented cloud machine that holds only the public NIH images and prompt text.

Attached: a one-page protocol summary. I registered for the NCBE research-ethics certificate on [DATE] and will send it as soon as it is issued. If the office requires the Research Application Form and CVs instead of a determination, please let me know and I will submit them.

Kind regards,
[NAME], [ID], [EMAIL], [PHONE]
Supervisor and PI of record: Dr. Muzammil Behzad, muzammil.behzad@kfupm.edu.sa

---

Notes for the student
- Fill every `[...]` field; keep the questions at five.
- Attach to the IRB email the one-page summary in `drafts/irb_protocol_summary.md` after completing its `[...]` fields.
- Register on ncbeportal.kacst.gov.sa tonight; complete the course on Days 1-3 so the certificate exists before the Day-4 reading.
- If the advisor replies that the umbrella project has an IRB number, forward that reply to the IRB office as a follow-up rather than withdrawing the request.
- Record the send time of the IRB email in `PROTOCOL.md` section 10 and in `PROGRESS.md`.
