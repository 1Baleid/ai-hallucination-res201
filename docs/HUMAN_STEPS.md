# Human steps: only the student can do these

Written 2026-11-02. Everything not listed here is agent work. No step involves contacting the advisor (dropped by the student on 2026-11-02). Dates are on hold; do the steps in this order whenever you are ready.

| Step | Unlocks | Your time | Money |
|---|---|---|---|
| 1. Kaggle account, competition rules, API token (DONE 2026-11-02) | label counts, image matching, item selection, image export | about 15 min | none |
| 2. Hugging Face terms and token (DONE 2026-11-02) | MedGemma 1.5 and MedSigLIP downloads | about 5 min | none |
| 3. RunPod account, credit, API key, volume (credit and key DONE 2026-11-02) | the GPU runs | about 15 min | $10 credit, about $5 to $8 used |
| 4. Blind reading of 360 images, then a 100-image retest at least 48 h later | verification, exclusions, kappa | about 3 h, then 0.8 h | none |
| 5. Read and approve the draft report | submission | about 3 h | none |
| Optional: NCBE certificate, IRB request, instructor email | the ethics paragraph, the due-date answer | 2 to 3 h | none |

## 1. Kaggle

1. Sign in at kaggle.com, or create an account.
2. Open https://www.kaggle.com/competitions/ranzcr-clip-catheter-line-classification, go to the Rules tab, and accept the rules (late joining is allowed).
3. Take a screenshot of the Rules tab and keep it outside the repository. Compare its "Data Access and Use" and "Data Security" text with `PROTOCOL.md` section 9, and tell the agent if anything differs.
4. Create an API token under Settings, API, "Create New Token". The Kaggle tool used here (version 2.2.5) accepts either file:
   - the downloaded `kaggle.json`, saved as `C:\Users\messi\.kaggle\kaggle.json`, or
   - a new-style token saved as `C:\Users\messi\.kaggle\access_token`.
5. Phone verification is not needed. The kernels run on CPU with internet switched off.

Never put the token inside the project folder. The `.gitignore` blocks `kaggle.json`, but keeping it in your user folder is safer.

## 2. Hugging Face

1. Sign in at huggingface.co.
2. Open https://huggingface.co/google/medgemma-1.5-4b-it and https://huggingface.co/google/medsiglip-448 and accept the Health AI Developer Foundations terms on each. Access is granted immediately.
3. Create a token with Read access under Settings, Access Tokens.
4. In a terminal opened in the project folder, run the line below and paste the token when asked. It is stored in your user profile, not in the repository.

```
.venv\Scripts\hf.exe auth login
```

## 3. RunPod

1. Create an account at runpod.io and add $10 of credit.
2. Under Settings, API Keys, create a key and paste it into `C:\Users\messi\.runpod\config.toml` on the line `apikey = "..."`. That file already exists with an empty key.
3. Under Storage, create a 50 GB network volume in Secure Cloud, in a data center that lists the RTX A5000.
4. When the agent is ready to run, it generates an SSH key on this machine and shows you the public key. Paste it under Settings, SSH Public Keys.
5. The agent starts the pod, runs the queue, downloads the results and stops the pod. Check the billing page afterwards to confirm nothing is still running.

## 4. Blind reading

1. When the agent says the images are ready, run the tool it names (`tools/read_tool.py`) with the practice list first, about 10 images.
2. Then read the 360 study images in one or two sittings of about 3 hours in total. For each image, answer whether an endotracheal tube is present, and set a flag when unsure, when it is a tracheostomy tube, or when the tube is out of the field. `docs/ETT_reference_sheet.md` explains what to look for.
3. Do not open any model output before the reading is saved. The tool saves and fingerprints the file when you finish.
4. At least 48 hours later, run the retest list of 100 images in the same tool.

## 5. The report

Read the full draft, correct anything wrong, and approve it. The agent formats it for the course template once you say which template applies.

## Optional ethics and course steps

- NCBE research-ethics certificate at ncbeportal.kacst.gov.sa.
- IRB determination request: not requested (your decision, 2026-11-02). The report states that no determination was requested because the study uses only public, de-identified data.
- Instructor email: `drafts/emails_day0.md` section 1, to confirm the due date and the template.
- OSF preregistration: skipped (your decision, 2026-11-02). The protocol is frozen by a dated git commit instead, and the report says the study was not publicly preregistered.
