# AKIRA Wrist Component: Dataset Audit Report

## 1. 3DCDataset
* **Original source**: 3DCDataset-main
* **File structure**: TXT files containing raw numbers. Organized by Participant/test or train/Modality.
* **Number of participants**: 22
* **Number of sessions**: NOT CONFIRMED (Usually 1 session per participant recorded in cycles).
* **Number of trials**: Multiple gestures repeated 4 times per session.
* **Sensor types**: 10 bipolar EMG channels + 1 Myo Armband IMU.
* **Sensor locations**: Forearm.
* **Sampling frequency**: 1000 Hz (EMG), IMU frequency NOT CONFIRMED (implicit in rows).
* **Accelerometer availability**: NOT AVAILABLE (Outputs 4-column Quaternions).
* **Gyroscope availability**: NOT AVAILABLE (Outputs 4-column Quaternions).
* **Timestamp availability**: NOT AVAILABLE (Implicit sampling rate).
* **Exercise availability**: wrist flexion, wrist extension, radial deviation, ulnar deviation, pronation, supination, plus grasps.
* **Labels**: Gesture ID (0 to 10).
* **Missing data**: None apparent in standard files.
* **Data quality**: Consistent format, but IMU data is pre-processed into quaternions rather than raw 6-axis measurements.
* **Strengths**: Contains the exact 6 wrist movements required for AKIRA.
* **Limitations**: ONE IMU ONLY. No raw accelerometer or gyroscope data. 
* **AKIRA relevance**: CATEGORY E (or D). USEFUL FOR EXERCISE REFERENCE BUT NOT A TWO-IMU AKIRA DATASET. Cannot be standardized into raw IMU format.

## 2. Hand and Arm Rehabilitation 2D Dataset
* **Original source**: d7f2p6h4x3-1
* **File structure**: Directory of 960 JPG image files.
* **Number of participants**: NOT CONFIRMED.
* **Number of sessions**: NOT CONFIRMED.
* **Number of trials**: NOT CONFIRMED.
* **Sensor types**: Camera (RGB).
* **Sensor locations**: External.
* **Sampling frequency**: NOT AVAILABLE.
* **Accelerometer availability**: NOT AVAILABLE.
* **Gyroscope availability**: NOT AVAILABLE.
* **Timestamp availability**: NOT AVAILABLE.
* **Exercise availability**: Visual poses.
* **Labels**: Inferred from image contents.
* **Missing data**: N/A
* **Data quality**: 2D images.
* **Strengths**: Visual reference of hand positions.
* **Limitations**: No time-series sensor data. No IMU.
* **AKIRA relevance**: CATEGORY E. NOT USEFUL FOR THE AKIRA WRIST COMPONENT.

## 3. OpenMuscle Dataset
* **Original source**: OpenMuscle-Dataset-main
* **File structure**: TXT (JSON-like dicts) and CSV files.
* **Number of participants**: NOT CONFIRMED.
* **Number of sessions**: Multiple captures.
* **Number of trials**: NOT CONFIRMED.
* **Sensor types**: Hall effect sensors, pressure grid (flexgrid).
* **Sensor locations**: Forearm/Hand.
* **Sampling frequency**: Variable/NOT CONFIRMED.
* **Accelerometer availability**: NOT AVAILABLE.
* **Gyroscope availability**: NOT AVAILABLE.
* **Timestamp availability**: YES (ticks / rec_time).
* **Exercise availability**: Finger/hand biomechanics movements.
* **Labels**: Embedded in file names and metadata.
* **Missing data**: None observed in samples.
* **Data quality**: High resolution specialized hardware data.
* **Strengths**: High density muscle deformation data.
* **Limitations**: NO IMU DATA.
* **AKIRA relevance**: CATEGORY E. NOT USEFUL FOR THE AKIRA WRIST COMPONENT.

## 4. 29666735 Wrist Kinematics Dataset
* **Original source**: 29666735.zip
* **File structure**: BIDS format (EDF, TSV, JSON).
* **Number of participants**: At least 11.
* **Number of sessions**: 1 task session per participant.
* **Number of trials**: Multiple runs per participant.
* **Sensor types**: EEG and Biomech Wrist Exoskeleton (motor encoders).
* **Sensor locations**: Scalp (EEG), Wrist (Exoskeleton).
* **Sampling frequency**: NOT CONFIRMED (typically 250Hz - 1000Hz).
* **Accelerometer availability**: NOT AVAILABLE.
* **Gyroscope availability**: NOT AVAILABLE.
* **Timestamp availability**: YES (via onset/duration and sampling intervals).
* **Exercise availability**: Continuous 2D wrist pointing task (combined Flexion-Extension and Ulnar-Radial Deviation).
* **Labels**: Task events and continuous wrist angles (radians).
* **Missing data**: Minimal, clean laboratory data.
* **Data quality**: High-grade kinematic tracking.
* **Strengths**: Contains true wrist joint angles (kinematics) in standard radians for the specific axes AKIRA tracks.
* **Limitations**: NO IMU DATA. Exoskeleton kinematics only.
* **AKIRA relevance**: CATEGORY C. KINEMATIC / WRIST MOVEMENT REFERENCE DATA. Excellent for verifying ROM and wrist angle calculation algorithms, but cannot be used for raw IMU sensor fusion development.

## 5. WatchHand Dataset
* **Original source**: WatchHand-main
* **File structure**: CSV, MP4, NPY, TXT files organized by subject and study.
* **Number of participants**: Multiple (e.g. sub10).
* **Number of sessions**: Multiple blocks per participant.
* **Number of trials**: Multiple trials per block.
* **Sensor types**: Smartwatch IMU, Acoustic sensing.
* **Sensor locations**: Wrist/Hand.
* **Sampling frequency**: NOT CONFIRMED (High frequency for acoustics, standard for IMU).
* **Accelerometer availability**: YES (X, Y, Z).
* **Gyroscope availability**: YES (X, Y, Z).
* **Timestamp availability**: YES (Unix timestamp in milliseconds).
* **Exercise availability**: 3D Hand poses (primarily finger joints).
* **Labels**: Continuous 3D hand joint tracking coordinates.
* **Missing data**: Sparse sampling intervals in some CSV strings.
* **Data quality**: Contains real 6-axis raw IMU data bundled in large strings.
* **Strengths**: The only dataset in this batch with actual raw Accelerometer + Gyroscope data.
* **Limitations**: ONE IMU ONLY. Located on the wrist/hand, missing the forearm sensor. Focused on hand poses rather than wrist ROM exercises.
* **AKIRA relevance**: CATEGORY B. RAW 6-AXIS IMU DATA BUT DIFFERENT SENSOR CONFIGURATION. Useful for single-IMU algorithm development but not a two-IMU AKIRA dataset.

---

## FINAL RECOMMENDATION

**A. Which dataset is BEST for AKIRA wrist IMU development?**
None of the datasets provide the required two-IMU configuration. The **WatchHand Dataset** is the closest for single-IMU algorithm testing because it has raw accelerometer and gyroscope data.

**B. Which dataset contains the most relevant wrist exercises?**
**3DCDataset** contains the exact isolated wrist exercises (WRIST_FLEXION, WRIST_EXTENSION, RADIAL_DEVIATION, ULNAR_DEVIATION, PRONATION, SUPINATION).

**C. Which dataset has the best accelerometer + gyroscope data?**
**WatchHand Dataset**. It is the only dataset containing raw 3-axis accelerometer and 3-axis gyroscope data.

**D. Which dataset has a forearm sensor?**
**3DCDataset** (Myo armband) and **OpenMuscle** (Flexgrid). However, neither provides raw 6-axis IMU data.

**E. Which dataset has a hand/wrist sensor?**
**WatchHand Dataset** (Smartwatch) and **29666735 Dataset** (Wrist exoskeleton).

**F. Does any public dataset actually match the AKIRA two-IMU configuration?**
**NO.** There are no public datasets in this download batch containing a synchronized Forearm + Hand two-IMU setup with raw 6-axis data.

**G. Which datasets can be standardized into the AKIRA raw format?**
**WatchHand Dataset** could partially be standardized (it has one IMU, not two). However, none can be perfectly standardized into the 2-IMU AKIRA format.

**H. Which datasets cannot be standardized and why?**
- **3DCDataset**: Only provides quaternions (4 columns), lacks raw ax, ay, az, gx, gy, gz.
- **29666735 Dataset**: Contains joint angles (kinematics), not IMU measurements.
- **OpenMuscle**: Uses hall effect/pressure sensors, not IMU.
- **Hand and Arm Rehabilitation 2D**: Only contains image files.

**I. Which dataset is best used as a kinematic reference?**
**29666735 Wrist Kinematics Dataset**. It provides high-quality continuous wrist joint angles in radians for flexion/extension and radial/ulnar deviation.

**J. Which dataset is best used as rehabilitation/exercise reference?**
**3DCDataset** is best for defining the execution and timing of the 6 core wrist exercises.

**K. Which required AKIRA exercises are missing?**
WatchHand (the only IMU dataset) is missing the specific isolated wrist rehabilitation exercises, focusing instead on continuous hand/finger poses.

**L. What information will still need to be collected using the actual AKIRA two-IMU hardware?**
A completely new dataset must be collected using the AKIRA dual-IMU hardware to capture true relative forearm-to-hand raw 6-axis IMU data across all 6 core wrist exercises.

**M. What should be the next step after this audit?**
Do not merge these datasets. Keep 29666735 as a kinematic gold-standard reference for evaluating the sensor fusion math. Write a data collection protocol to record actual dual-IMU data using the AKIRA hardware.
