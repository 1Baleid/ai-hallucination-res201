# One-page protocol summary for the IRB determination request (attachment)

Draft 2026-10-12 (revised after an ethics review the same evening) for the student to complete the `[...]` fields and attach to the email in `drafts/emails_day0.md`, section 3.

**Title**: Benchmarking and Mitigating Object Hallucination in Multimodal Large Language Models Using Synthetic and Real Visual Data (RES 201 component; umbrella project "Evaluating VLMs/LLMs Hallucination in Domain-Specific Tasks", SDAIA-KFUPM JRCAI).

**Student**: [NAME], [ID], Information and Computer Science. **Supervisor and PI of record**: Dr. Muzammil Behzad, ICS. **Period**: 13 October 2026 to [END], with a possible later training stage.

**Purpose**: to measure how often small open vision-language models (2-4 billion parameters) report a medical device (an endotracheal tube) that is not present on a chest radiograph, and whether two training-free methods (detector-gated answers; retrieval of similar labelled cases) reduce such errors.

**Data**: two public, de-identified research datasets, used as released by their custodians.
1. NIH ChestX-ray14 (NIH Clinical Center, 2017): 112,120 frontal chest radiographs released publicly with no patient identifiers; used under the custodian's attribution requirements.
2. RANZCR CLiP (Kaggle competition dataset, 2021; Tang et al., Scientific Data 2021): image-level labels for tubes and lines on 30,083 of the same radiographs, created by 43 trained labellers; used under the competition's data rules (academic research permitted; no redistribution; no re-identification); label files are not redistributed.
No identifiable information, no hospital records, no images outside these releases, and no data from KFUPM or any Saudi institution are used.

**Procedures**: (1) matching the two public releases of the same radiographs by image content (dimensions and perceptual hashing), which is the only linkage in the study; (2) automated inference by open-weight models on a rented cloud machine that holds only the public NIH images and prompt text; (3) the student's own non-clinical reading of about 300 images to check whether a device is visible; this reading flags ambiguous items, excludes label-image disagreements after a matching re-check, and is reported as agreement with the published labels; (4) statistical analysis. No patients are contacted; no clinical decisions are made or implied; no new data are collected from human participants in this stage; outputs are research measurements, not medical interpretations.

**Planned start**: 13 October 2026 (automated analysis of public de-identified images only). The student's own reading sessions will pause if the office objects.

**Possible later elements (the office will be notified before they start)**: a short blinded image reading by a lab member or a clinician for label-quality checks; fine-tuning of models on the same public images.

**Risks**: none to patients (public de-identified data; no re-identification attempted; no identifiable data exist in either release). Risks to the student are nil beyond ordinary computer work.

**Data handling**: label files stay on the Kaggle platform and the student's computer; the cloud machine holds only public NIH images and prompt text and is deleted after use; no label data, identifiers or edited images are published; released materials are limited to NIH filenames, code and aggregate results.

**Outputs**: an RES 201 report and possibly a conference paper; all images shown in publications are from the public NIH release.

**Training**: NCBE research-ethics certificate: registered on [DATE]; certificate to follow by [DATE].

**Request**: a determination that this activity is not research on a human subject under the Law of Ethics of Research on Living Creatures or, failing that, expedited review under the NCBE Implementing Regulations (v3), Article 10.19 (minimal risk; previously collected, non-identifying records), with exemption from periodic follow-up under Article 10.33 (publicly available, non-identifying data).

Contact: [NAME], [EMAIL], [PHONE]; supervisor muzammil.behzad@kfupm.edu.sa.
