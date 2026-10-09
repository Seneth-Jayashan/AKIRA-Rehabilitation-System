# 01 Data Audit

## 1. Purpose of the Folder
This directory contains the foundational evidence ensuring the integrity of the public dataset before any model training commenced. It documents the exploratory data analysis, data quality checks, and preprocessing decisions.

## 2. What Files are Inside
- **dataset_overview/**: High-level dataset summaries, class distributions, and requirement matrices.
- **data_quality/**: Missing value checks, trial quality reports, and distribution analyses.
- **subject_split/**: Verification of strict subject-independence and leakage checks.
- **preprocessing/**: Proposals, configurations, and validation checklists for signal processing.
- **anomalies/**: Visual and empirical tracking of sensor step-changes and boundary continuity.

## 3. Why Those Files Matter
These documents prove that the underlying kinematic data is physically sound, that no information leaked between the train and test splits, and that the preprocessing pipeline cleanly isolated the elbow and forearm targets without corrupting the signal.

## 4. Workflow Context
This represents **Stage 1** of the research pipeline. All data auditing was completed before model architectural choices were finalized.

## 5. What a Supervisor Should Look at First
Begin with `dataset_overview/dataset_summary.md` to understand the scope of the raw data, followed by `subject_split/leakage_check_report.md` to verify the integrity of the evaluation bounds.
