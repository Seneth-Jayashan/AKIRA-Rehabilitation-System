/*
 * ======================================================================================
 * AKIRA KNEE REHABILITATION SYSTEM — DUAL-IMU ESP32 FIRMWARE (100 Hz)
 * ======================================================================================
 * Student Investigator: Wickramarathna S A (IT23222540)
 * Component: Knee Rehabilitation Following Tibial Plateau ORIF
 *
 * HARDWARE WIRING (Two MPU6050 6-DOF IMUs on a single ESP32 I2C Bus):
 *   - ESP32 3.3V   ---> IMU 1 VCC (Thigh) & IMU 2 VCC (Shin)
 *   - ESP32 GND    ---> IMU 1 GND (Thigh) & IMU 2 GND (Shin)
 *   - ESP32 GPIO21 ---> IMU 1 SDA (Thigh) & IMU 2 SDA (Shin)
 *   - ESP32 GPIO22 ---> IMU 1 SCL (Thigh) & IMU 2 SCL (Shin)
 *   - IMU 1 AD0    ---> Connected to GND  (I2C Address = 0x68 -> THIGH SENSOR)
 *   - IMU 2 AD0    ---> Connected to 3.3V (I2C Address = 0x69 -> SHIN SENSOR)
 *
 * SENSOR PLACEMENT ON PATIENT LEG:
 *   - IMU 1 (Thigh): 10 cm ABOVE center of kneecap (anterior femur centerline)
 *   - IMU 2 (Shin) : 10 cm BELOW center of kneecap (anterior tibia flat bone)
 *   - Axis Alignment: X-arrow UP toward hip, Y-arrow SIDEWAYS across knee joint hinge.
 *
 * OUTPUT FORMAT (100 Hz = every 10 ms):
 *   thigh_ax,thigh_ay,thigh_az,thigh_gx,thigh_gy,thigh_gz,shin_ax,shin_ay,shin_az,shin_gx,shin_gy,shin_gz
 *   - Accelerometer units: g (1g = 9.81 m/s^2, sensitivity +/- 4g -> 8192 LSB/g)
 *   - Gyroscope units    : deg/s (sensitivity +/- 500 deg/s -> 65.5 LSB/(deg/s))
 * ======================================================================================
 */

#include <Wire.h>
#include "BluetoothSerial.h"

BluetoothSerial SerialBT;

// I2C Addresses for Dual IMUs
const uint8_t IMU_THIGH_ADDR = 0x68; // AD0 connected to GND
const uint8_t IMU_SHIN_ADDR  = 0x69; // AD0 connected to 3.3V

// MPU6050 Register Map
const uint8_t REG_PWR_MGMT_1   = 0x6B;
const uint8_t REG_ACCEL_CONFIG = 0x1C;
const uint8_t REG_GYRO_CONFIG  = 0x1B;
const uint8_t REG_ACCEL_XOUT_H = 0x3B;

// Scale Factors (+/- 4g and +/- 500 deg/s)
const float ACCEL_SCALE = 8192.0f; // LSB per g
const float GYRO_SCALE  = 65.5f;   // LSB per deg/s

// 100 Hz Sampling Interval (10,000 microseconds = 10 ms)
const unsigned long SAMPLE_INTERVAL_US = 10000;
unsigned long lastSampleMicros = 0;

// Gyroscope Calibration Offsets (computed during 2-second stationary startup)
float thighGyroBias[3] = {0.0f, 0.0f, 0.0f};
float shinGyroBias[3]  = {0.0f, 0.0f, 0.0f};

void configureIMU(uint8_t addr) {
  // Wake up MPU6050
  Wire.beginTransmission(addr);
  Wire.write(REG_PWR_MGMT_1);
  Wire.write(0x00);
  Wire.endTransmission(true);

  // Set Accelerometer range to +/- 4g (0x08)
  Wire.beginTransmission(addr);
  Wire.write(REG_ACCEL_CONFIG);
  Wire.write(0x08);
  Wire.endTransmission(true);

  // Set Gyroscope range to +/- 500 deg/s (0x08)
  Wire.beginTransmission(addr);
  Wire.write(REG_GYRO_CONFIG);
  Wire.write(0x08);
  Wire.endTransmission(true);
}

void readIMURaw(uint8_t addr, float &ax, float &ay, float &az, float &gx, float &gy, float &gz) {
  Wire.beginTransmission(addr);
  Wire.write(REG_ACCEL_XOUT_H);
  Wire.endTransmission(false);
  Wire.requestFrom(addr, (uint8_t)14, (uint8_t)true);

  int16_t raw_ax = (Wire.read() << 8) | Wire.read();
  int16_t raw_ay = (Wire.read() << 8) | Wire.read();
  int16_t raw_az = (Wire.read() << 8) | Wire.read();
  Wire.read(); Wire.read(); // Skip temperature bytes
  int16_t raw_gx = (Wire.read() << 8) | Wire.read();
  int16_t raw_gy = (Wire.read() << 8) | Wire.read();
  int16_t raw_gz = (Wire.read() << 8) | Wire.read();

  ax = raw_ax / ACCEL_SCALE;
  ay = raw_ay / ACCEL_SCALE;
  az = raw_az / ACCEL_SCALE;
  gx = raw_gx / GYRO_SCALE;
  gy = raw_gy / GYRO_SCALE;
  gz = raw_gz / GYRO_SCALE;
}

void calibrateGyroscopes() {
  const int N_CAL = 200; // 2.0 seconds at 100 Hz
  for (int i = 0; i < N_CAL; i++) {
    float tax, tay, taz, tgx, tgy, tgz;
    float sax, say, saz, sgx, sgy, sgz;
    readIMURaw(IMU_THIGH_ADDR, tax, tay, taz, tgx, tgy, tgz);
    readIMURaw(IMU_SHIN_ADDR,  sax, say, saz, sgx, sgy, sgz);

    thighGyroBias[0] += tgx; thighGyroBias[1] += tgy; thighGyroBias[2] += tgz;
    shinGyroBias[0]  += sgx; shinGyroBias[1]  += sgy; shinGyroBias[2]  += sgz;
    delay(10);
  }
  for (int k = 0; k < 3; k++) {
    thighGyroBias[k] /= N_CAL;
    shinGyroBias[k]  /= N_CAL;
  }
}

void setup() {
  Serial.begin(115200);
  SerialBT.begin("AKIRA_Knee_ESP32"); // Bluetooth device name
  Wire.begin(21, 22);                 // SDA = GPIO 21, SCL = GPIO 22
  Wire.setClock(400000);              // 400 kHz Fast I2C

  configureIMU(IMU_THIGH_ADDR);
  configureIMU(IMU_SHIN_ADDR);
  delay(200);

  // Perform 2-second stationary gyroscope bias calibration
  calibrateGyroscopes();
  lastSampleMicros = micros();
}

void loop() {
  unsigned long now = micros();
  if (now - lastSampleMicros >= SAMPLE_INTERVAL_US) {
    lastSampleMicros += SAMPLE_INTERVAL_US;

    float tax, tay, taz, tgx, tgy, tgz;
    float sax, say, saz, sgx, sgy, sgz;

    readIMURaw(IMU_THIGH_ADDR, tax, tay, taz, tgx, tgy, tgz);
    readIMURaw(IMU_SHIN_ADDR,  sax, say, saz, sgx, sgy, sgz);

    // Subtract static gyroscope bias
    tgx -= thighGyroBias[0]; tgy -= thighGyroBias[1]; tgz -= thighGyroBias[2];
    sgx -= shinGyroBias[0];  sgy -= shinGyroBias[1];  sgz -= shinGyroBias[2];

    // Format 12-channel CSV line
    char line[180];
    snprintf(
      line, sizeof(line),
      "%.3f,%.3f,%.3f,%.2f,%.2f,%.2f,%.3f,%.3f,%.3f,%.2f,%.2f,%.2f\n",
      tax, tay, taz, tgx, tgy, tgz,
      sax, say, saz, sgx, sgy, sgz
    );

    // Stream simultaneously over USB Serial and Bluetooth
    Serial.print(line);
    if (SerialBT.hasClient()) {
      SerialBT.print(line);
    }
  }
}
