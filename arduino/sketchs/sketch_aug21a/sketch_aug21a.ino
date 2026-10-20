#include <Wire.h>
#include <U8g2lib.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>

// =====================================
// BLE UUIDs
// =====================================
#define SERVICE_UUID           "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
#define DATA_CHAR_UUID         "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"
#define STATUS_CHAR_UUID       "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"

// =====================================================================
// DIRECT I2C SENSOR ADDRESSES — NO MULTIPLEXER
// Both sensors share SDA=GPIO21 and SCL=GPIO22.
// Sensor 1 AD0 LOW  -> address 0x68
// Sensor 2 AD0 HIGH -> address 0x69
// =====================================================================
#define MPU_SENSOR_1_ADDR 0x68
#define MPU_SENSOR_2_ADDR 0x69

// =====================================
// MPU6050 REGISTERS
// =====================================
#define WHO_AM_I_REG   0x75
#define CONFIG_REG     0x1A
#define GYRO_CONFIG    0x1B
#define ACCEL_CONFIG   0x1C
#define ACCEL_XOUT_H   0x3B
#define GYRO_XOUT_H    0x43
#define USER_CTRL_REG  0x6A
#define PWR_MGMT_1     0x6B
#define PWR_MGMT_2     0x6C

// =====================================
// OLED DISPLAY
// =====================================
U8G2_SH1106_128X64_NONAME_F_HW_I2C oled(U8G2_R0, U8X8_PIN_NONE);

// =====================================
// SENSOR DATA TYPES & SYNCHRONIZED PACKET
// =====================================
struct ImuReading {
    float ax, ay, az;
    float gx, gy, gz;
};

// 12-channel synchronized IMU packet (52 bytes total)
// Sensor 1 (6 channels) + Sensor 2 (6 channels) sampled synchronously
struct __attribute__((packed)) DualImuPacket {
    uint32_t timestamp;  // 4 bytes: ESP32 millis() timestamp
    // Sensor 1 / Proximal IMU (0x68) - 24 bytes
    float s1_ax;
    float s1_ay;
    float s1_az;
    float s1_gx;
    float s1_gy;
    float s1_gz;
    // Sensor 2 / Distal IMU (0x69) - 24 bytes
    float s2_ax;
    float s2_ay;
    float s2_az;
    float s2_gx;
    float s2_gy;
    float s2_gz;
};

ImuReading sensor1Data = {0};
ImuReading sensor2Data = {0};
bool sensor1Available = false;
bool sensor2Available = false;

// =====================================
// BLE VARIABLES
// =====================================
BLEServer* pServer = NULL;
BLECharacteristic* pDataCharacteristic = NULL;
bool deviceConnected = false;
bool oldDeviceConnected = false;

class MyServerCallbacks: public BLEServerCallbacks {
    void onConnect(BLEServer* pServer) {
      deviceConnected = true;
    }

    void onDisconnect(BLEServer* pServer) {
      deviceConnected = false;
    }
};

// =====================================
// WRITE IMU REGISTER (DIRECT I2C)
// =====================================
bool writeRegister(uint8_t address, uint8_t reg, uint8_t value) {
  Wire.beginTransmission(address);
  Wire.write(reg);
  Wire.write(value);
  return (Wire.endTransmission(true) == 0);
}

// =====================================
// READ MULTIPLE BYTES (With STOP & Turnaround Delay)
// =====================================
bool readBytes(uint8_t address, uint8_t reg, uint8_t *buffer, uint8_t length) {
  Wire.beginTransmission(address);
  Wire.write(reg);
  // Releasing the bus with STOP condition (true) ensures a STOP condition before the read
  if (Wire.endTransmission(true) != 0) {
    return false;
  }
  delayMicroseconds(50);
  uint8_t received = Wire.requestFrom((uint8_t)address, length);
  if (received != length) {
    return false;
  }
  for (uint8_t i = 0; i < length; i++) {
    buffer[i] = Wire.read();
  }
  return true;
}

// =====================================
// COMBINE HIGH + LOW BYTES
// =====================================
inline int16_t combineBytes(uint8_t high, uint8_t low) {
  return (int16_t)((high << 8) | low);
}

// =====================================
// READ MPU6050 INTO ImuReading STRUCT
// Explicitly reads ACCEL (0x3B) and GYRO (0x43) separately
// to prevent auto-increment stalls across the temperature registers.
// =====================================
bool readMPU6050(uint8_t address, ImuReading &out) {
  uint8_t accelData[6];
  uint8_t gyroData[6];

  // 1. Read 6 bytes of Accelerometer data (0x3B..0x40)
  if (!readBytes(address, ACCEL_XOUT_H, accelData, 6)) {
    return false;
  }

  // 2. Read 6 bytes of Gyroscope data (0x43..0x48)
  if (!readBytes(address, GYRO_XOUT_H, gyroData, 6)) {
    return false;
  }

  int16_t rawAx = combineBytes(accelData[0], accelData[1]);
  int16_t rawAy = combineBytes(accelData[2], accelData[3]);
  int16_t rawAz = combineBytes(accelData[4], accelData[5]);

  int16_t rawGx = combineBytes(gyroData[0], gyroData[1]);
  int16_t rawGy = combineBytes(gyroData[2], gyroData[3]);
  int16_t rawGz = combineBytes(gyroData[4], gyroData[5]);

  // Accelerometer: ±2g full scale -> 16384 LSB/g
  out.ax = rawAx / 16384.0f;
  out.ay = rawAy / 16384.0f;
  out.az = rawAz / 16384.0f;

  // Gyroscope: ±250 °/s full scale -> 131 LSB/(°/s)
  out.gx = rawGx / 131.0f;
  out.gy = rawGy / 131.0f;
  out.gz = rawGz / 131.0f;

  return true;
}

// =====================================
// INITIALIZE AN IMU AT A DIRECT I2C ADDRESS
// =====================================
bool initSensor(uint8_t address, const char* sensorName) {
  // Confirm the sensor acknowledges before attempting configuration.
  Wire.beginTransmission(address);
  if (Wire.endTransmission(true) != 0) {
    Serial.printf("ERROR: %s not found at 0x%02X\n", sensorName, address);
    return false;
  }

  delay(10);

  // Reset device.
  if (!writeRegister(address, PWR_MGMT_1, 0x80)) {
    Serial.printf("ERROR: %s reset command failed\n", sensorName);
    return false;
  }
  delay(100);

  // Wake up and select gyro X PLL clock.
  if (!writeRegister(address, PWR_MGMT_1, 0x01)) {
    Serial.printf("ERROR: %s wake/clock init failed\n", sensorName);
    return false;
  }
  delay(15);

  writeRegister(address, PWR_MGMT_2, 0x00);
  writeRegister(address, USER_CTRL_REG, 0x00);
  writeRegister(address, ACCEL_CONFIG, 0x00); // +/-2g
  writeRegister(address, GYRO_CONFIG, 0x00);  // +/-250 deg/s
  writeRegister(address, CONFIG_REG, 0x03);   // DLPF ~42Hz
  delay(10);

  uint8_t whoami = 0, pwr1 = 0, pwr2 = 0, gyroCfg = 0;
  readBytes(address, WHO_AM_I_REG, &whoami, 1);
  readBytes(address, PWR_MGMT_1, &pwr1, 1);
  readBytes(address, PWR_MGMT_2, &pwr2, 1);
  readBytes(address, GYRO_CONFIG, &gyroCfg, 1);

  Serial.printf("%s [0x%02X]: WHO_AM_I=0x%02X, PWR1=0x%02X, PWR2=0x%02X, GYRO_CFG=0x%02X\n",
                sensorName, address, whoami, pwr1, pwr2, gyroCfg);

  if (whoami != 0x68) {
    Serial.printf("WARNING: %s returned unexpected WHO_AM_I value 0x%02X\n",
                  sensorName, whoami);
  }

  if (pwr2 != 0x00) {
    Serial.printf("WARNING: %s PWR_MGMT_2 is 0x%02X\n", sensorName, pwr2);
  }

  Serial.printf("%s ready at I2C address 0x%02X\n", sensorName, address);
  return true;
}

// =====================================
// OLED DISPLAY FUNCTIONS
// =====================================
void displaySensorData() {
  oled.clearBuffer();
  oled.setFont(u8g2_font_6x10_tf);

  if (deviceConnected) {
    oled.drawStr(0, 9, "BLE: Connected");
  } else {
    oled.drawStr(0, 9, "BLE: Disconnected");
  }
  oled.drawHLine(0, 11, 128);

  // Sensor 1 (Proximal: Tibia / Thigh / Wrist / UpperArm)
  char line[32];
  oled.drawStr(0, 22, "S1 Prox (0x68)");
  snprintf(line, sizeof(line), "A:%4.1f %4.1f %4.1f", sensor1Data.ax, sensor1Data.ay, sensor1Data.az);
  oled.drawStr(0, 32, line);

  // Sensor 2 (Distal: Foot / Shank / Hand / Forearm)
  oled.drawStr(0, 44, "S2 Dist (0x69)");
  snprintf(line, sizeof(line), "A:%4.1f %4.1f %4.1f", sensor2Data.ax, sensor2Data.ay, sensor2Data.az);
  oled.drawStr(0, 54, line);

  snprintf(line, sizeof(line), "AKIRA Hub @ 100Hz");
  oled.drawStr(0, 64, line);

  oled.sendBuffer();
}

void displayStreamingScreen() {
  oled.clearBuffer();
  oled.setFont(u8g2_font_6x10_tf);
  oled.drawStr(0, 15, "AKIRA 4-Member Hub");
  oled.drawHLine(0, 18, 128);
  oled.drawStr(0, 30, "BLE: Connected");
  oled.drawStr(0, 44, "S1(68) + S2(69)");
  oled.drawStr(0, 58, "Streaming @ 100 Hz");
  oled.sendBuffer();
}

// =====================================
// SETUP
// =====================================
void setup() {
  Serial.begin(115200);
  delay(1000);

  // I2C shared by both directly connected IMUs and OLED
  Wire.begin(21, 22);
  Wire.setClock(400000);

  // Initialize OLED
  oled.begin();
  oled.clearBuffer();
  oled.setFont(u8g2_font_6x10_tf);
  oled.drawStr(0, 15, "AKIRA Rehab System");
  oled.drawStr(0, 30, "4-Member Data Hub");
  oled.drawStr(0, 48, "Initializing...");
  oled.sendBuffer();

  Serial.println();
  Serial.println("=================================================");
  Serial.println("AKIRA REHABILITATION: DIRECT DUAL-IMU PLATFORM");
  Serial.println("M1: Ankle (Tibia/Foot)  | M2: Knee (Thigh/Shank)");
  Serial.println("M3: Wrist (Wrist/Hand)  | M4: Elbow (UpperArm/Forearm)");
  Serial.println("=================================================");

  // Initialize BLE
  BLEDevice::init("ESP32-REHAB");
  pServer = BLEDevice::createServer();
  pServer->setCallbacks(new MyServerCallbacks());

  BLEService *pService = pServer->createService(SERVICE_UUID);

  pDataCharacteristic = pService->createCharacteristic(
                      DATA_CHAR_UUID,
                      BLECharacteristic::PROPERTY_NOTIFY
                    );
  pDataCharacteristic->addDescriptor(new BLE2902());

  BLECharacteristic *pStatusCharacteristic = pService->createCharacteristic(
                      STATUS_CHAR_UUID,
                      BLECharacteristic::PROPERTY_READ
                    );
  pStatusCharacteristic->setValue("AKIRA Direct Dual-IMU 12-Channel Hub Ready");

  pService->start();

  BLEAdvertising *pAdvertising = BLEDevice::getAdvertising();
  pAdvertising->addServiceUUID(SERVICE_UUID);
  pAdvertising->setScanResponse(true);
  pAdvertising->setMinPreferred(0x06);
  pAdvertising->setMinPreferred(0x12);
  BLEDevice::startAdvertising();

  Serial.println("BLE Ready to Pair!");

  // Initialize both sensors directly by their distinct I2C addresses.
  sensor1Available = initSensor(MPU_SENSOR_1_ADDR, "Sensor 1 (Proximal)");
  sensor2Available = initSensor(MPU_SENSOR_2_ADDR, "Sensor 2 (Distal)");

  // Show status on OLED
  oled.clearBuffer();
  oled.drawStr(0, 15, "Sensors (0x68/0x69):");
  char statusLine[32];
  snprintf(statusLine, sizeof(statusLine), "S1 Prox: %s", sensor1Available ? "OK" : "ERROR");
  oled.drawStr(0, 32, statusLine);
  snprintf(statusLine, sizeof(statusLine), "S2 Dist: %s", sensor2Available ? "OK" : "ERROR");
  oled.drawStr(0, 48, statusLine);
  oled.drawStr(0, 62, "Waiting for BLE...");
  oled.sendBuffer();

  delay(1500);
  Serial.println("AKIRA Direct Dual IMU Platform Ready!");
}

// =====================================
// MAIN LOOP
// =====================================
void loop() {
  static unsigned long lastSampleTime = 0;

  if (deviceConnected) {
    if (!oldDeviceConnected) {
      oldDeviceConnected = true;
      displayStreamingScreen();
      Serial.println("App connected! Starting synchronized 100Hz 12-channel stream...");
    }

    unsigned long now = millis();
    // 10ms interval = exactly 100Hz sampling rate
    if (now - lastSampleTime >= 10) {
      lastSampleTime = now;

      DualImuPacket packet;
      memset(&packet, 0, sizeof(DualImuPacket));
      packet.timestamp = now;

      // 1. Sample Sensor 1 directly at I2C address 0x68.
      if (sensor1Available) {
        if (readMPU6050(MPU_SENSOR_1_ADDR, sensor1Data)) {
          packet.s1_ax = sensor1Data.ax;
          packet.s1_ay = sensor1Data.ay;
          packet.s1_az = sensor1Data.az;
          packet.s1_gx = sensor1Data.gx;
          packet.s1_gy = sensor1Data.gy;
          packet.s1_gz = sensor1Data.gz;
        }
      }

      // 2. Sample Sensor 2 directly at I2C address 0x69.
      if (sensor2Available) {
        if (readMPU6050(MPU_SENSOR_2_ADDR, sensor2Data)) {
          packet.s2_ax = sensor2Data.ax;
          packet.s2_ay = sensor2Data.ay;
          packet.s2_az = sensor2Data.az;
          packet.s2_gx = sensor2Data.gx;
          packet.s2_gy = sensor2Data.gy;
          packet.s2_gz = sensor2Data.gz;
        }
      }

      // Notify the Android data collector with 52-byte synchronized packet
      pDataCharacteristic->setValue((uint8_t*)&packet, sizeof(DualImuPacket));
      pDataCharacteristic->notify();
    }

  } else {
    // Handle disconnection
    if (oldDeviceConnected) {
      delay(500);
      pServer->startAdvertising();
      Serial.println("Device disconnected. Restarting BLE advertising...");
      oldDeviceConnected = false;
    }

    unsigned long now = millis();
    static unsigned long lastDisplayUpdate = 0;

    if (now - lastDisplayUpdate > 500) {
      if (sensor1Available) {
        readMPU6050(MPU_SENSOR_1_ADDR, sensor1Data);
      }
      if (sensor2Available) {
        readMPU6050(MPU_SENSOR_2_ADDR, sensor2Data);
      }
      displaySensorData();
      lastDisplayUpdate = now;
    }
  }
}
