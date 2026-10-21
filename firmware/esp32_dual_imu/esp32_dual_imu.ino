/*
 * ======================================================================================
 * AKIRA ELBOW REHABILITATION SYSTEM — DUAL-IMU ESP32 BLE + USB SERIAL FIRMWARE (100 Hz)
 * ======================================================================================
 * Component: Elbow Rehabilitation Following Olecranon Fracture ORIF
 *
 * SUPPORTS BOTH:
 *   1. BLE (Bluetooth Low Energy) — Device Name: "ESP32-REHAB"
 *      - Service UUID       : 6e400001-b5a3-f393-e0a9-e50e24dcca9e
 *      - TX Characteristic  : 6e400002-b5a3-f393-e0a9-e50e24dcca9e (Notify @ 100 Hz)
 *   2. USB Serial (115200 baud)
 *
 * HARDWARE WIRING (Two MPU6050 6-DOF IMUs on a single ESP32 I2C Bus):
 *   - ESP32 3.3V   ---> IMU 1 VCC (Upper Arm) & IMU 2 VCC (Forearm)
 *   - ESP32 GND    ---> IMU 1 GND (Upper Arm) & IMU 2 GND (Forearm)
 *   - ESP32 GPIO21 ---> IMU 1 SDA (Upper Arm) & IMU 2 SDA (Forearm)
 *   - ESP32 GPIO22 ---> IMU 1 SCL (Upper Arm) & IMU 2 SCL (Forearm)
 *   - IMU 1 AD0    ---> Connected to GND  (I2C Address = 0x68 -> UPPER ARM / RUA)
 *   - IMU 2 AD0    ---> Connected to 3.3V (I2C Address = 0x69 -> FOREARM / RLA)
 *
 * MATHEMATICAL SCALING & CALIBRATION EQUATIONS:
 * ======================================================================================
 * 1. Accelerometer Physical Scaling (+/- 4g Full-Scale Range):
 *      Sensitivity = 8192 LSB/g  (from MPU-6050 Register 28 ACCEL_CONFIG = 0x08)
 *      Formula:
 *        a [g] = raw_int16 / 8192.0
 *        a [m/s^2] = (raw_int16 / 8192.0) * 9.80665
 *
 * 2. Gyroscope Physical Scaling (+/- 500 deg/s Full-Scale Range):
 *      Sensitivity = 65.5 LSB/(deg/s) (from MPU-6050 Register 27 GYRO_CONFIG = 0x08)
 *      Formula:
 *        omega [deg/s] = raw_int16 / 65.5
 *        omega [rad/s] = (raw_int16 / 65.5) * (pi / 180.0)
 *
 * 3. Hardware Sampling Rate Divider (100 Hz = 10 ms interval):
 *      f_sample = f_internal / (1 + SMPLRT_DIV)
 *               = 1000 Hz / (1 + 9) = 100.0 Hz  ==>  Delta_t = 10,000 microseconds
 *
 * 4. Static Gyroscope Zero-Rate Bias Estimation:
 *      During initial 2.0 s stationary calibration (N_cal = 200 samples @ 100 Hz):
 *        b_gyro[axis] = (1 / N_cal) * sum_{i=1}^{N_cal} omega_raw[axis, i]
 *      Real-time compensation:
 *        omega_calibrated[axis, t] = omega_measured[axis, t] - b_gyro[axis]
 *
 * OUTPUT FORMAT (100 Hz = every 10 ms):
 *   rua_ax,rua_ay,rua_az,rua_gx,rua_gy,rua_gz,rla_ax,rla_ay,rla_az,rla_gx,rla_gy,rla_gz
 * ======================================================================================
 */

#include <Wire.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>

// BLE UUIDs (Standard Nordic UART Service matching AKIRA ecosystem)
#define SERVICE_UUID           "6e400001-b5a3-f393-e0a9-e50e24dcca9e"
#define CHARACTERISTIC_UUID_TX "6e400002-b5a3-f393-e0a9-e50e24dcca9e"

BLEServer *pServer = NULL;
BLECharacteristic *pTxCharacteristic = NULL;
bool bleDeviceConnected = false;

class MyServerCallbacks: public BLEServerCallbacks {
  void onConnect(BLEServer* pServer) {
    bleDeviceConnected = true;
  }
  void onDisconnect(BLEServer* pServer) {
    bleDeviceConnected = false;
    pServer->getAdvertising()->start();
  }
};

// I2C Addresses for Dual IMUs (Elbow Rehabilitation)
const uint8_t IMU_UPPER_ARM_ADDR = 0x68; // AD0 connected to GND  (RUA)
const uint8_t IMU_FOREARM_ADDR   = 0x69; // AD0 connected to 3.3V (RLA)

// MPU6050 Register Map
const uint8_t REG_PWR_MGMT_1   = 0x6B;
const uint8_t REG_SMPLRT_DIV   = 0x19;
const uint8_t REG_CONFIG       = 0x1A; // Digital Low-Pass Filter (DLPF)
const uint8_t REG_GYRO_CONFIG  = 0x1B;
const uint8_t REG_ACCEL_CONFIG = 0x1C;
const uint8_t REG_ACCEL_XOUT_H = 0x3B;

// ======================================================================================
// MATHEMATICAL CONVERSION CONSTANTS
// ======================================================================================
// Accelerometer: +/- 4g full scale ==> 65536 / 8g = 8192 LSB/g (Eq. 1)
const float ACCEL_SCALE = 8192.0f; 

// Gyroscope: +/- 500 deg/s full scale ==> 65536 / 1000 deg/s = 65.536 LSB/(deg/s) (Eq. 2)
const float GYRO_SCALE  = 65.5f;   

// 100 Hz Sampling Interval: Delta_t = 1 / 100.0 Hz = 0.010 s = 10,000 microseconds (Eq. 3)
const unsigned long SAMPLE_INTERVAL_US = 10000;
unsigned long lastSampleMicros = 0;

// Gyroscope Static Calibration Offsets (Zero-Rate Bias vectors b_gyro in R^3, Eq. 4)
float upperArmGyroBias[3] = {0.0f, 0.0f, 0.0f};
float forearmGyroBias[3]  = {0.0f, 0.0f, 0.0f};

void configureIMU(uint8_t addr) {
  // Wake up MPU6050 and select PLL with X-axis gyroscope reference
  Wire.beginTransmission(addr);
  Wire.write(REG_PWR_MGMT_1);
  Wire.write(0x01);
  Wire.endTransmission(true);

  // Enable Hardware Digital Low-Pass Filter (DLPF_CFG = 4 -> 20 Hz bandwidth, 1 kHz internal rate)
  // Prevents aliasing and eliminates high-frequency I2C register read glitches
  Wire.beginTransmission(addr);
  Wire.write(REG_CONFIG);
  Wire.write(0x04);
  Wire.endTransmission(true);

  // Set sample rate divider (Eq. 3): 1000 Hz / (1 + 9) = 100 Hz
  Wire.beginTransmission(addr);
  Wire.write(REG_SMPLRT_DIV);
  Wire.write(0x09);
  Wire.endTransmission(true);

  // Configure Accelerometer range: +/- 4g (8192 LSB/g, Eq. 1)
  Wire.beginTransmission(addr);
  Wire.write(REG_ACCEL_CONFIG);
  Wire.write(0x08);
  Wire.endTransmission(true);

  // Configure Gyroscope range: +/- 500 deg/s (65.5 LSB/deg/s, Eq. 2)
  Wire.beginTransmission(addr);
  Wire.write(REG_GYRO_CONFIG);
  Wire.write(0x08);
  Wire.endTransmission(true);
}

bool readIMURaw(uint8_t addr, float &ax, float &ay, float &az, float &gx, float &gy, float &gz) {
  Wire.beginTransmission(addr);
  Wire.write(REG_ACCEL_XOUT_H);
  if (Wire.endTransmission(false) != 0) {
    ax = ay = az = gx = gy = gz = 0.0f;
    return false;
  }
  uint8_t count = Wire.requestFrom(addr, (uint8_t)14, (uint8_t)true);
  if (count < 14) {
    ax = ay = az = gx = gy = gz = 0.0f;
    return false;
  }

  int16_t raw_ax = (Wire.read() << 8) | Wire.read();
  int16_t raw_ay = (Wire.read() << 8) | Wire.read();
  int16_t raw_az = (Wire.read() << 8) | Wire.read();
  Wire.read(); Wire.read(); // Skip temperature
  int16_t raw_gx = (Wire.read() << 8) | Wire.read();
  int16_t raw_gy = (Wire.read() << 8) | Wire.read();
  int16_t raw_gz = (Wire.read() << 8) | Wire.read();

  // Reject uninitialized or disconnected bus states
  if ((raw_ax == -1 && raw_ay == -1 && raw_az == -1) || (raw_ax == 0 && raw_ay == 0 && raw_az == 0)) {
    ax = ay = az = gx = gy = gz = 0.0f;
    return false;
  }

  // Convert raw 16-bit counts to physical units (Eq. 1 & 2)
  ax = raw_ax / ACCEL_SCALE; // in g
  ay = raw_ay / ACCEL_SCALE;
  az = raw_az / ACCEL_SCALE;
  gx = raw_gx / GYRO_SCALE;  // in deg/s
  gy = raw_gy / GYRO_SCALE;
  gz = raw_gz / GYRO_SCALE;
  return true;
}

void calibrateGyroscopes() {
  // MATHEMATICAL FORMULA: Static Gyroscope Mean Zero-Rate Bias (Eq. 4)
  //   b_gyro = (1 / N_cal) * sum_{i=1}^{N_cal} omega_i,  where N_cal = 200 samples (2.0s)
  const int N_CAL = 200; // 2.0 seconds at 100 Hz
  int uaValid = 0, faValid = 0;
  for (int i = 0; i < N_CAL; i++) {
    float uax, uay, uaz, ugx, ugy, ugz;
    float fax, fay, faz, fgx, fgy, fgz;
    if (readIMURaw(IMU_UPPER_ARM_ADDR, uax, uay, uaz, ugx, ugy, ugz)) {
      upperArmGyroBias[0] += ugx; upperArmGyroBias[1] += ugy; upperArmGyroBias[2] += ugz;
      uaValid++;
    }
    if (readIMURaw(IMU_FOREARM_ADDR, fax, fay, faz, fgx, fgy, fgz)) {
      forearmGyroBias[0] += fgx; forearmGyroBias[1] += fgy; forearmGyroBias[2] += fgz;
      faValid++;
    }
    delay(10);
  }
  for (int k = 0; k < 3; k++) {
    upperArmGyroBias[k] = (uaValid > 0) ? (upperArmGyroBias[k] / uaValid) : 0.0f;
    forearmGyroBias[k]  = (faValid  > 0) ? (forearmGyroBias[k]  / faValid)  : 0.0f;
  }
}

void setupBLE() {
  BLEDevice::init("ESP32-REHAB");
  BLEDevice::setMTU(185);
  pServer = BLEDevice::createServer();
  pServer->setCallbacks(new MyServerCallbacks());

  BLEService *pService = pServer->createService(SERVICE_UUID);
  pTxCharacteristic = pService->createCharacteristic(
    CHARACTERISTIC_UUID_TX,
    BLECharacteristic::PROPERTY_NOTIFY | BLECharacteristic::PROPERTY_READ
  );
  pTxCharacteristic->addDescriptor(new BLE2902());

  pService->start();
  BLEAdvertising *pAdvertising = BLEDevice::getAdvertising();
  pAdvertising->addServiceUUID(SERVICE_UUID);
  pAdvertising->setScanResponse(true);
  pAdvertising->start();
}

void setup() {
  Serial.begin(115200);
  Wire.begin(21, 22);    // SDA = GPIO 21, SCL = GPIO 22
  Wire.setClock(100000); // 100 kHz Standard-Mode I2C for high noise immunity over wearable straps
  Wire.setTimeOut(5);    // 5 ms bus timeout recovery

  configureIMU(IMU_UPPER_ARM_ADDR);
  configureIMU(IMU_FOREARM_ADDR);
  delay(200);

  setupBLE();
  calibrateGyroscopes();
  lastSampleMicros = micros();
}

// 52-Byte Packed Binary Packet Struct (<I12f)
struct __attribute__((packed)) DualIMUPacket {
  uint32_t timestamp_ms;
  float ch[12]; // [0..5] = Upper Arm (0x68), [6..11] = Forearm (0x69)
};

void loop() {
  unsigned long now = micros();
  // Fixed-interval 10 ms (100 Hz) execution loop (Eq. 3)
  if (now - lastSampleMicros >= SAMPLE_INTERVAL_US) {
    lastSampleMicros += SAMPLE_INTERVAL_US;

    float uax, uay, uaz, ugx, ugy, ugz;
    float fax, fay, faz, fgx, fgy, fgz;

    bool uaOk = readIMURaw(IMU_UPPER_ARM_ADDR, uax, uay, uaz, ugx, ugy, ugz);
    bool faOk = readIMURaw(IMU_FOREARM_ADDR,   fax, fay, faz, fgx, fgy, fgz);

    // Apply real-time zero-rate gyro bias subtraction (Eq. 4)
    if (uaOk) {
      ugx -= upperArmGyroBias[0]; ugy -= upperArmGyroBias[1]; ugz -= upperArmGyroBias[2];
    }
    if (faOk) {
      fgx -= forearmGyroBias[0];  fgy -= forearmGyroBias[1];  fgz -= forearmGyroBias[2];
    }

    // 1. Send 12-channel CSV line over USB Serial
    char line[160];
    snprintf(
      line, sizeof(line),
      "%.3f,%.3f,%.3f,%.2f,%.2f,%.2f,%.3f,%.3f,%.3f,%.2f,%.2f,%.2f\n",
      uax, uay, uaz, ugx, ugy, ugz,
      fax, fay, faz, fgx, fgy, fgz
    );
    Serial.print(line);

    // 2. Send 52-byte Dual-IMU binary packet (<I12f) over BLE
    if (bleDeviceConnected && pTxCharacteristic != NULL) {
      DualIMUPacket pkt;
      pkt.timestamp_ms = millis();
      pkt.ch[0] = uax; pkt.ch[1] = uay; pkt.ch[2] = uaz;
      pkt.ch[3] = ugx; pkt.ch[4] = ugy; pkt.ch[5] = ugz;
      pkt.ch[6] = fax; pkt.ch[7] = fay; pkt.ch[8] = faz;
      pkt.ch[9] = fgx; pkt.ch[10] = fgy; pkt.ch[11] = fgz;
      pTxCharacteristic->setValue((uint8_t*)&pkt, sizeof(DualIMUPacket));
      pTxCharacteristic->notify();
    }
  }
}
