Days 1–2 — Setup & Data

[x] Fork the repo, checkout topic-recognition, create your folder recognition/MambaIRv2_sXXXXXXX/

[x] Find the starter notebook mentioned in the spec (check Blackboard/course materials)
- https://colab.research.google.com/drive/1AyC5c4tVwstI0irzbgbVxQecxg1D_ux7?usp=sharing 

[x] Set up your conda environment locally with PyTorch

[] Understand the dataset — the spec says “supplied test images” so check what’s actually provided

[] Write your feasibility review (needed for tutor check-off AND README)

[] Start dataset.py

Days 3–4 — Baseline

[] Get the pretrained MambaIRv2 weights running (frozen, no fine-tuning)
[] Run inference on the test set — this becomes your baseline
[] Record PSNR, SSIM, LPIPS, Colorfulness scores on the baseline
[] Commit this working baseline

Days 5–7 — Fine-tuning

[] Implement fine-tuning in train.py
[] Train locally on your RTX 5060
[] Compare metrics before/after fine-tuning
[] If you need more compute, push to Rangpur via sbatch

Days 8–9 — Analysis & Predict

[] Write predict.py with side-by-side visualisations (greyscale input / baseline / fine-tuned / ground truth)
[] Pick 3–5 failure cases and write the autopsy (color bleeding, desaturation, chromatic hallucinations)
[] Draft your engineering recommendation

Days 10–11 — README & Submit

[] Complete README with all required sections
[] Final commit sweep
[] Open Pull Request, Turnitin PDF submission