# GitHub Readiness Audit Report

## 1. Project Root and Inspected Directories
The entire AKIRA project root (`D:\me\project`) was inspected, recursively parsing 61,000+ files. Key directories evaluated:
- `demo/` (API and Frontend)
- `models/` (Checkpoints)
- `reports/` (Official Evaluation and Visuals)
- `src/` & `scripts/` (Source Code)
- `data/` & `datasets/` (Raw and Processed Datasets)
- `results/` (Training Logs)

## 2. File-Category Summary
The project comprises:
- **Research & Source Code:** Professional, shareable artifacts (Markdown, Python, UI).
- **Artifacts:** Tiny trained models that are safe to share.
- **Heavy Data:** Multi-gigabyte MoCap and IMU raw datasets that must not be committed.
- **Temporary Debris:** One-off data-processing scripts and system cache files.

## 3. Files Recommended for GitHub (`SHOULD COMMIT`)
The following core assets ensure the demo runs and the research is transparent:
- `reports/` (The entire 5-stage documentation hierarchy, charts, and final master CSVs).
- `src/`, `scripts/`, `demo/` (All backend code, frontend UI, and inference logic).
- `demo/demo_data/` (Compact 10 MB collection of isolated CSV test windows used by the localhost demo).
- `data/processed/class_mapping.json` (Required by the backend for class decoding).
- `requirements.txt` / `demo/requirements.txt`.
- `models/final/*.pth` (See Section 6).

## 4. Files Recommended for `.gitignore` (`SHOULD IGNORE`)
The newly generated `.gitignore` safely excludes:
- `datasets/` and `data/processed/windows/` (Massive dataset footprint exceeding 3 GB).
- `results/` (Intermediate training histories; the official results are already preserved in `reports/03_FINAL_EVALUATION/`).
- `models/smoke_test/` (Temporary checkpoints).
- `__pycache__/`, `*.pyc`, `.venv/` (System debris).
- Temporary root scripts (`fix_encoding.py`, `move_files.py`, `audit.py`, `file_list.csv`).

## 5. Files Requiring Your Decision
**Training Logs (`results/`)**: I have configured `.gitignore` to ignore this directory because the authoritative `final_results_master.csv` is safely stored in `reports/`. If you want to share the raw epoch-by-epoch training history curves (`tcn_training_history.csv`), you must manually force-add them or adjust the `.gitignore`.
**Raw Datasets (`datasets/`)**: I strongly recommend ignoring these, but if your supervisor specifically requests access to the raw MoCap `.mvnx` files, they must be shared via a secure external cloud drive due to size and potential licensing restrictions.

## 6. Large Files and Checkpoint Sizes
**Good News!** Git LFS is **not** required for your models.
- `models/final/best_tcn.pth`: **0.14 MB**
- `models/final/best_lstm.pth`: **0.03 MB**
- `models/final/best_linear.pth`: **0.002 MB**
These are remarkably lightweight and have been explicitly un-ignored in the `.gitignore` to guarantee they are uploaded.

*Note: 54 files in `datasets/` exceed the 50 MB GitHub warning threshold, and several exceed the strict 100 MB block limit. They are safely trapped by the `.gitignore`.*

## 7. Sensitive-Information Findings
**No hardcoded secrets or API keys were detected.** I scanned the codebase, `app.py`, and `config.json` files. No `.env` files or exposed credentials exist in the project directory.

## 8. Dataset and Redistribution Concerns
The `datasets/` directory contains folders like `Smartwatch-IMU-UpperLimb-Dataset` and `ULF_in_ADL`. These are public academic datasets that typically possess non-commercial or specific attribution licenses. Redistributing them inside your repository is discouraged both technically (file size) and legally. The included `demo/demo_data/` sample extracts are anonymized and mathematically transformed, making them safe for the repository.

## 9. Required Dependencies for the Demo
The demo strictly relies on:
- `demo/backend/app.py`
- `data/processed/class_mapping.json`
- `models/final/best_tcn.pth`
- `demo/demo_data/` CSV files.
All of these are properly tracked and will be uploaded to the repository.

## 10. Upload Blockers
There are no remaining blockers. The `.gitignore` is completely configured.

## 11. Recommended Next Steps (VS Code GUI)
1. Open the VS Code **Source Control** panel (Ctrl+Shift+G).
2. Click **Initialize Repository**.
3. Review the list of "Changes" (Staged files). Verify that the `datasets/` directory does not appear in the list.
4. Enter a commit message (e.g., "Initial commit: AKIRA Research and Demo") and click **Commit**.
5. Click **Publish Branch** to upload it to a Private GitHub repository.
