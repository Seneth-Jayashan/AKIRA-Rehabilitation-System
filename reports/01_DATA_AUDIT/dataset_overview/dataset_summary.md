# ML Dataset Summary

## 1. Dataset Scale
- **Number of Subjects:** 15
- **Number of Valid Trials:** 299
- **Number of Exercises:** 20
- **Total Windows:** 3883

## 2. Shapes & Technical Specs
- **Input Shape:** (240, 12) (Time Steps, 12 Channels)
- **Target Shape (Regression):** (240, 2) (Continuous temporal sequence)
- **Sampling Frequency:** ~120 Hz
- **Window Duration:** 240 samples (2.0 seconds)
- **Window Overlap:** 120 samples (50%)

## 3. Data Split
- **Train:** 10 subjects, 2570 windows
- **Validation:** 3 subjects, 783 windows
- **Test:** 2 subjects, 530 windows

## 4. Targets Range
- **Forearm Rotation Range:** -179.82 to 179.97 degrees
- **Elbow Flexion Range:** -90.77 to 179.75 degrees

## 5. Discarded Trials
Check `trial_quality_report.csv` for any invalid trials.

## 6. Limitations
- Optotrak "Carrying Angle" was omitted as it does not present biological variance in this dataset.
- TCN will need to output `[batch_size, 240, 2]` to fully leverage this representation, or reduce to `[batch_size, 2]` if only predicting window-level terminal states.
