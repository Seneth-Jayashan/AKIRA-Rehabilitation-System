# Data Directory Guide (`data/`)

## 1. Purpose of Data Folders

- **`datasets/raw/` (Existing Local Raw Storage — Read-Only):**
  Contains the **13 original raw public datasets** (`~68.23 GB` total across `55,792` files) already present in the workspace prior to project initialization. In compliance with project governance, these files are preserved strictly in-place without modification, copying, or moving.
- **`data/raw/`:**
  Designated canonical raw data directory in the project layout. Rather than duplicating 68.23 GB of files on disk, scripts will either read directly from `datasets/raw/` via configurable path resolution or use directory junctions/symlinks inside `data/raw/` once approved.
- **`data/interim/`:**
  Stores intermediate, cleaned, schema-harmonized, and uniformly resampled per-trial time-series files generated during Stage 3 (prior to sliding-window segmentation).
- **`data/processed/`:**
  Stores the final subject-independent `train`, `val`, and `test` windowed tensors, scalers (fitted strictly on `train` subjects), and split manifests used identically across all four models (TCN, 1D CNN, LSTM, GRU) in Stage 4.

---

## 2. Inventory of Original Raw Datasets (`d:\me\project\datasets\raw\`)

| # | Folder Name in `datasets/raw/` | Size | File Count | Primary Formats | Modalities | Original Source / Reference |
| :- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `EMG elbow dataset` | 304.30 MB | 130 | `.txt` | 2-ch sEMG (raw + filtered) + Shimmer3 derived elbow angle | Toro-Ossaba & Tejada (Zenodo; Shimmer3 EMG/angle dataset) |
| 2 | `JU-IMU` | 383.21 MB | 574 | `.csv` | 5 × 9-axis IMUs (R/L wrist, R/L upper arm, trunk) | Jeonju University IMU Dataset (`youngminoh7/JU-IMU`) |
| 3 | `mm-fit` | 3.53 GB | 302 | `.npy`, `.csv` | 2 smartwatches, earbud, smartphones, 2D/3D pose | Strömbäck et al., *IMWUT / UbiComp* 2020 (MM-Fit) |
| 4 | `MoCap and IMU Dataset for Upper Limb Joint Angle Estimation` | 1.45 GB | 378 | `.csv` | 3 IMUs (chest, R/L wrist) + 47-bone OptiTrack MoCap | Niño-Tejada et al., *Electronics* 2025 (Zenodo `10.5281/zenodo.15778949`) |
| 5 | `Optotrak vs DOT` | 5.43 GB | 7,711 | `.csv`, `.c3d`, `.xls`, `.mat` (extensionless) | 5 × Xsens DOT IMUs (Trunk, Scapula, R Upper Arm, R Forearm, R Hand) + NDI Optotrak Certus MoCap | University Medical Centre Groningen (UMCG), Jan 2023 |
| 6 | `PHYTMO` | 11.68 GB | 7,076 | `.csv`, `.m` | 4 × 9-axis NGIMU (L/R Upper Arm + L/R Forearm in `upper`) + OptiTrack rigid bodies | García-de-Villa et al., *Scientific Data* 9, 266 (2022) (Zenodo) |
| 7 | `REHAB` | 1.07 GB | 36,110 | `.npy` | Dual upper-limb 9-axis IMUs (MPU9250) + flex-sensor data glove | Zhou et al., *Scientific Data* / *J. NeuroEng. Rehabil.* 2025 (REHAB-120) |
| 8 | `Smartwatch IMU Dataset for Upper-Limb Movement Ana` | 576.41 MB | 11 | `.csv`, `.md`, `.json` | Apple Watch Series 6 (wrist IMU + CoreMotion) | Benachour et al., *Eng. Appl. Artif. Intell.* 156:111194 (2025) (CC BY 4.0) |
| 9 | `StrokeRehab` | 5.80 GB | 1,426 | `.csv` | 9 × Noraxon MyoMotion IMUs (incl. L/R Upper Arm & Forearm) | Upper-extremity stroke rehab benchmark (healthy controls branch present locally) |
| 10 | `UCI Biceps Curl` | 1.59 MB | 1 | `.csv` | 4 × 9-DOF IMUs (`belt`, `arm`, `forearm`, `dumbbell`) | Velloso et al., *AH '13* (PUC-Rio Weight Lifting Exercises 4,024-row excerpt) |
| 11 | `ULF_in_ADL` | 24.66 GB | 4,493 | `.mvnx`, `.xlsx` | 17 × Xsens MVN IMUs (full body incl. L/R Upper Arm & Forearm) | SoftPro / Univ. Hospital Zurich Upper Limb Function in ADL dataset |
| 12 | `ULTra-MoCap` | 12.29 GB | 1,102 | `.csv`, `.trc`, `.mot`, `.osim`, `.h5` | 6 × Delsys Trigno 6-DOF IMUs + 3 sEMGs + 60-marker Vicon MoCap / OpenSim IK | ULTra-MoCap upper-limb biomechanics dataset |
| 13 | `VIDIMU` | 1.38 GB | 2,005 | `.csv`, `.sto`, `.mot`, `.raw` | 5 × IMUs (Torso, L/R Humerus, L/R Radius in `A05-A13`) + OpenSim IK + 3D Video Pose | Martinez-Hernandez et al. (VIDIMU multimodal human motion dataset) |

See [`docs/DATASET_AUDIT.md`](../docs/DATASET_AUDIT.md) for the complete technical audit of schemas, units, sampling rates, missing values, and task suitability.
