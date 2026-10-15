# AKIRA Recommended Standard Raw Schema

Any dataset containing genuine raw 6-axis IMU data that is integrated into the AKIRA environment should be converted to this standardized schema.

## Schema Columns

* `dataset`
* `participant_id`
* `session_id`
* `trial_id`
* `timestamp`
* `exercise_original`
* `exercise_akira`
* `side`
* `sensor_id`
* `sensor_location`
* `ax`
* `ay`
* `az`
* `gx`
* `gy`
* `gz`
* `label`
* `sampling_rate`
* `source_file`

## Rules

* **ax, ay, az**: Accelerometer X, Y, Z axes.
* **gx, gy, gz**: Gyroscope X, Y, Z axes.
* **Magnetometer**: EXCLUDED. Do not include magnetometer data in the standard schema.
* **Missing values**: If a dataset does not contain one of these fields (e.g. `gy`), mark the column as `NOT AVAILABLE` or leave it null. **DO NOT** create fake values or interpolate missing sensors.
* **Single IMU Data**: If a dataset only has one sensor (e.g. `sensor_location` = HAND), do NOT duplicate it to simulate a FOREARM sensor. 

Currently, no audited public dataset fully populates this schema for a dual-IMU configuration. This schema will act as the target spec for the custom dual-IMU data collected using the AKIRA hardware.
