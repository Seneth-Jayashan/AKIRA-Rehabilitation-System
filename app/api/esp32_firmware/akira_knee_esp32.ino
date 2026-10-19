/*
 * ======================================================================================
 * AKIRA KNEE REHABILITATION SYSTEM — DUAL-IMU ESP32 BLE + USB SERIAL FIRMWARE (100 Hz)
 * ======================================================================================
 * Student Investigator: Wickramarathna S A (IT23222540)
 * Component: Knee Rehabilitation Following Tibial Plateau ORIF
 *
 * SUPPORTS BOTH:
 *   1. BLE (Bluetooth Low Energy) — Device Name: "AKIRA_Knee_BLE"
 *      - Service UUID       : 6e400001-b5a3-f393-e0a9-e50e24dcca9e
 *      - TX Characteristic  : 6e400003-b5a3-f393-e0a9-e50e24dcca9e (Notify @ 100 Hz)
 *   2. USB Serial (115200 baud)
 *
 * HARDWARE WIRING (Two MPU6050 6-DOF IMUs on a single ESP32 I2C Bus):
 *   - ESP32 3.3V   ---> IMU 1 VCC (Thigh) & IMU 2 VCC (Shin)
 *   - ESP32 GND    ---> IMU 1 GND (Thigh) & IMU 2 GND (Shin)
 *   - ESP32 GPIO21 ---> IMU 1 SDA (Thigh) & IMU 2 SDA (Shin)
 *   - ESP32 GPIO22 ---> IMU 1 SCL (Thigh) & IMU 2 SCL (Shin)
 *   - IMU 1 AD0    ---> Connected to GND  (I2C Address = 0x68 -> THIGH SENSOR)
 *   - IMU 2 AD0    ---> Connected to 3.3V (I2C Address = 0x69 -> SHIN SENSOR)
 *
 * OUTPUT FORMAT (100 Hz = every 10 ms):
 *   thigh_ax,thigh_ay,thigh_az,thigh_gx,thigh_gy,thigh_gz,shin_ax,shin_ay,shin_az,shin_gx,shin_gy,shin_gz
 * ======================================================================================
 */

#include <Wire.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>

// BLE UUIDs (Standard Nordic UART Service)
#define SERVICE_UUID           "6e400001-b5a3-f393-e0a9-e50e24dcca9e"
#define CHARACTERISTIC_UUID_TX "6e400003-b5a3-f393-e0a9-e50e24dcca9e"

BLEServer *pServer = NULL;
BLECharacteristic *pTxCharacteristic = NULL;
bool bleDeviceConnected = false;

class MyServerCallbacks: public BLEServerCallbacks {
  void onConnect(BLEServer* pServer) {
    bleDeviceConnected = true;
  }
  void onDisconnect(BLEServer* pServer) {
    bleDeviceConnected = false;
    pServer->getAdvertising()->start(); // Restart advertising if disconnected
  }
};

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

// Gyroscope Calibration Offsets
float thighGyroBias[3] = {0.0f, 0.0f, 0.0f};
float shinGyroBias[3]  = {0.0f, 0.0f, 0.0f};

void configureIMU(uint8_t addr) {
  Wire.beginTransmission(addr);
  Wire.write(REG_PWR_MGMT_1);
  Wire.write(0x00);
  Wire.endTransmission(true);

  Wire.beginTransmission(addr);
  Wire.write(REG_ACCEL_CONFIG);
  Wire.write(0x08); // +/- 4g
  Wire.endTransmission(true);

  Wire.beginTransmission(addr);
  Wire.write(REG_GYRO_CONFIG);
  Wire.write(0x08); // +/- 500 deg/s
  Wire.endTransmission(true);
}

bool readIMURaw(uint8_t addr, float &ax, float &ay, float &az, float &gx, float &gy, float &gz) {
  Wire.beginTransmission(addr);
  Wire.write(REG_ACCEL_XOUT_H);
  if (Wire.endTransmission(false) != 0) {
    // Sensor at this I2C address (0x68 or 0x69) is NOT connected! Output strict 0.0
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

  // Detect all-0xFF or all-0x00 invalid bus reads
  if ((raw_ax == -1 && raw_ay == -1 && raw_az == -1) || (raw_ax == 0 && raw_ay == 0 && raw_az == 0)) {
    ax = ay = az = gx = gy = gz = 0.0f;
    return false;
  }

  ax = raw_ax / ACCEL_SCALE;
  ay = raw_ay / ACCEL_SCALE;
  az = raw_az / ACCEL_SCALE;
  gx = raw_gx / GYRO_SCALE;
  gy = raw_gy / GYRO_SCALE;
  gz = raw_gz / GYRO_SCALE;
  return true;
}

void calibrateGyroscopes() {
  const int N_CAL = 200; // 2.0 seconds at 100 Hz
  int thighValid = 0, shinValid = 0;
  for (int i = 0; i < N_CAL; i++) {
    float tax, tay, taz, tgx, tgy, tgz;
    float sax, say, saz, sgx, sgy, sgz;
    if (readIMURaw(IMU_THIGH_ADDR, tax, tay, taz, tgx, tgy, tgz)) {
      thighGyroBias[0] += tgx; thighGyroBias[1] += tgy; thighGyroBias[2] += tgz;
      thighValid++;
    }
    if (readIMURaw(IMU_SHIN_ADDR, sax, say, saz, sgx, sgy, sgz)) {
      shinGyroBias[0] += sgx; shinGyroBias[1] += sgy; shinGyroBias[2] += sgz;
      shinValid++;
    }
    delay(10);
  }
  for (int k = 0; k < 3; k++) {
    thighGyroBias[k] = (thighValid > 0) ? (thighGyroBias[k] / thighValid) : 0.0f;
    shinGyroBias[k]  = (shinValid  > 0) ? (shinGyroBias[k]  / shinValid)  : 0.0f;
  }
}

void setupBLE() {
  BLEDevice::init("ESP32-REHAB");
  BLEDevice::setMTU(185); // Allow full 52-byte 12-channel packet in one BLE notification
  pServer = BLEDevice::createServer();
  pServer->setCallbacks(new MyServerCallbacks());

  BLEService *pService = pServer->createService(SERVICE_UUID);
  pTxCharacteristic = pService->createCharacteristic(
    "6e400002-b5a3-f393-e0a9-e50e24dcca9e",
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
  Wire.setClock(400000); // 400 kHz Fast I2C

  configureIMU(IMU_THIGH_ADDR);
  configureIMU(IMU_SHIN_ADDR);
  delay(200);

  setupBLE();
  calibrateGyroscopes();
  lastSampleMicros = micros();
}

struct __attribute__((packed)) DualIMUPacket {
  uint32_t timestamp_ms;
  float ch[12]; // [0..5] = Thigh (0x68), [6..11] = Shin (0x69)
};

void loop() {
  unsigned long now = micros();
  if (now - lastSampleMicros >= SAMPLE_INTERVAL_US) {
    lastSampleMicros += SAMPLE_INTERVAL_US;

    float tax, tay, taz, tgx, tgy, tgz;
    float sax, say, saz, sgx, sgy, sgz;

    bool thighOk = readIMURaw(IMU_THIGH_ADDR, tax, tay, taz, tgx, tgy, tgz);
    bool shinOk  = readIMURaw(IMU_SHIN_ADDR,  sax, say, saz, sgx, sgy, sgz);

    if (thighOk) {
      tgx -= thighGyroBias[0]; tgy -= thighGyroBias[1]; tgz -= thighGyroBias[2];
    }
    if (shinOk) {
      sgx -= shinGyroBias[0];  sgy -= shinGyroBias[1];  sgz -= shinGyroBias[2];
    }

    // 1. Send 12-channel CSV line over USB Serial
    char line[160];
    snprintf(
      line, sizeof(line),
      "%.3f,%.3f,%.3f,%.2f,%.2f,%.2f,%.3f,%.3f,%.3f,%.2f,%.2f,%.2f\n",
      tax, tay, taz, tgx, tgy, tgz,
      sax, say, saz, sgx, sgy, sgz
    );
    Serial.print(line);

    // 2. Send 52-byte Dual-IMU binary packet (<I12f) over BLE
    if (bleDeviceConnected && pTxCharacteristic != NULL) {
      DualIMUPacket pkt;
      pkt.timestamp_ms = millis();
      pkt.ch[0] = tax; pkt.ch[1] = tay; pkt.ch[2] = taz;
      pkt.ch[3] = tgx; pkt.ch[4] = tgy; pkt.ch[5] = tgz;
      pkt.ch[6] = sax; pkt.ch[7] = say; pkt.ch[8] = saz;
      pkt.ch[9] = sgx; pkt.ch[10] = sgy; pkt.ch[11] = sgz;
      pTxCharacteristic->setValue((uint8_t*)&pkt, sizeof(DualIMUPacket));
      pTxCharacteristic->notify();
    }
  }
}
