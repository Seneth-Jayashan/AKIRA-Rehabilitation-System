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
// I2C ADDRESSES & PCA9548A MULTIPLEXER CHANNELS
// Supports all 4 Project Members / Body Components:
// - Member 1 (Ankle): Sensor 1 = Tibia (CH0),     Sensor 2 = Foot (CH1)
// - Member 2 (Knee):  Sensor 1 = Thigh (CH0),     Sensor 2 = Shank (CH1)
// - Member 3 (Wrist): Sensor 1 = Wrist (CH0),     Sensor 2 = Hand (CH1)
// - Member 4 (Elbow): Sensor 1 = Upper Arm (CH0), Sensor 2 = Forearm (CH1)
// =====================================================================
#define PCA9548A_ADDR 0x70
#define MPU6500_ADDR  0x68

#define CH_SENSOR_1   0  // Proximal Segment (Tibia / Thigh / Wrist / Upper Arm)
#define CH_SENSOR_2   1  // Distal Segment   (Foot / Shank / Hand / Forearm)

// =====================================
// MPU6500 REGISTERS
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
    // Sensor 1 / Proximal IMU (CH0) - 24 bytes
    float s1_ax;
    float s1_ay;
    float s1_az;
    float s1_gx;
    float s1_gy;
    float s1_gz;
    // Sensor 2 / Distal IMU (CH1) - 24 bytes
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
// SELECT PCA9548A CHANNEL (Fast, No Delay)
// =====================================
bool selectChannel(uint8_t channel) {
  Wire.beginTransmission(PCA9548A_ADDR);
  Wire.write(1 << channel);
  return (Wire.endTransmission(true) == 0);
}

// =====================================
// WRITE MPU6500 REGISTER
// =====================================
bool writeRegister(uint8_t reg, uint8_t value) {
  Wire.beginTransmission(MPU6500_ADDR);
  Wire.write(reg);
  Wire.write(value);
  return (Wire.endTransmission(true) == 0);
}

// =====================================
// READ MULTIPLE BYTES (With STOP & Turnaround Delay)
// =====================================
bool readBytes(uint8_t reg, uint8_t *buffer, uint8_t length) {
  Wire.beginTransmission(MPU6500_ADDR);
  Wire.write(reg);
  // Releasing the bus with STOP condition (true) ensures PCA9548A multiplexer does not glitch
  if (Wire.endTransmission(true) != 0) {
    return false;
  }
  delayMicroseconds(50);
  uint8_t received = Wire.requestFrom((uint8_t)MPU6500_ADDR, length);
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
// READ MPU6500 INTO ImuReading STRUCT
// Explicitly reads ACCEL (0x3B) and GYRO (0x43) separately
// to prevent auto-increment stalls across the temperature registers.
// =====================================
bool readMPU6500(ImuReading &out) {
  uint8_t accelData[6];
  uint8_t gyroData[6];

  // 1. Read 6 bytes of Accelerometer data (0x3B..0x40)
  if (!readBytes(ACCEL_XOUT_H, accelData, 6)) {
    return false;
  }

  // 2. Read 6 bytes of Gyroscope data (0x43..0x48)
  if (!readBytes(GYRO_XOUT_H, gyroData, 6)) {
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
// INITIALIZE AN IMU ON SPECIFIC CHANNEL
// Ensures device reset, Gyro PLL clock selection, and active gyro axes
// =====================================
bool initSensorOnChannel(uint8_t channel, const char* sensorName) {
  if (!selectChannel(channel)) {
    Serial.printf("ERROR: PCA9548A failed to select channel %d (%s)\n", channel, sensorName);
    return false;
  }
  delay(15);

  // 1. Device Reset to clear any corrupted state or standby locks
  writeRegister(PWR_MGMT_1, 0x80);
  delay(100);

  // 2. Wake up and set clock source to PLL with Gyro X reference (0x01)
  // CLKSEL 1 powers up the gyroscope drive circuitry and provides a stable clock reference
  if (!writeRegister(PWR_MGMT_1, 0x01)) {
    Serial.printf("ERROR: %s wake/clock init failed\n", sensorName);
    return false;
  }
  delay(15);

  // 3. Clear all standby bits in PWR_MGMT_2 (0x6C)
  // STBY_XG (bit 2), STBY_YG (bit 1), STBY_ZG (bit 0) MUST be 0 for Gyro to run!
  writeRegister(PWR_MGMT_2, 0x00);
  delay(10);

  // 4. Disable FIFO and I2C Master mode
  writeRegister(USER_CTRL_REG, 0x00);
  delay(5);

  // 5. Configure Accelerometer ±2g (0x00)
  writeRegister(ACCEL_CONFIG, 0x00);
  delay(5);

  // 6. Configure Gyroscope ±250 deg/s (0x00)
  writeRegister(GYRO_CONFIG, 0x00);
  delay(5);

  // 7. Configure Digital Low Pass Filter (DLPF ~42Hz bandwidth)
  writeRegister(CONFIG_REG, 0x03);
  delay(5);

  // Diagnostic readback
  uint8_t whoami = 0, pwr1 = 0, pwr2 = 0, gyroCfg = 0;
  readBytes(WHO_AM_I_REG, &whoami, 1);
  readBytes(PWR_MGMT_1, &pwr1, 1);
  readBytes(PWR_MGMT_2, &pwr2, 1);
  readBytes(GYRO_CONFIG, &gyroCfg, 1);

  Serial.printf("%s [CH%d]: WHO_AM_I=0x%02X, PWR1=0x%02X, PWR2=0x%02X, GYRO_CFG=0x%02X\n",
                sensorName, channel, whoami, pwr1, pwr2, gyroCfg);

  if (pwr2 != 0x00) {
    Serial.printf("WARNING: %s PWR_MGMT_2 is 0x%02X (standby bits non-zero!)\n", sensorName, pwr2);
  }

  Serial.printf("%s ready on PCA9548A channel %d!\n", sensorName, channel);
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
  oled.drawStr(0, 22, "S1 Prox (CH0)");
  snprintf(line, sizeof(line), "A:%4.1f %4.1f %4.1f", sensor1Data.ax, sensor1Data.ay, sensor1Data.az);
  oled.drawStr(0, 32, line);

  // Sensor 2 (Distal: Foot / Shank / Hand / Forearm)
  oled.drawStr(0, 44, "S2 Dist (CH1)");
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
  oled.drawStr(0, 44, "S1(CH0) + S2(CH1)");
  oled.drawStr(0, 58, "Streaming @ 100 Hz");
  oled.sendBuffer();
}

// =====================================
// SETUP
// =====================================
void setup() {
  Serial.begin(115200);
  delay(1000);

  // 400kHz Fast Mode I2C
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
  Serial.println("AKIRA REHABILITATION: 4-MEMBER DUAL-IMU PLATFORM");
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
  pStatusCharacteristic->setValue("AKIRA Dual-IMU 12-Channel Hub Ready");

  pService->start();

  BLEAdvertising *pAdvertising = BLEDevice::getAdvertising();
  pAdvertising->addServiceUUID(SERVICE_UUID);
  pAdvertising->setScanResponse(true);
  pAdvertising->setMinPreferred(0x06);
  pAdvertising->setMinPreferred(0x12);
  BLEDevice::startAdvertising();

  Serial.println("BLE Ready to Pair!");

  // Initialize Sensor 1 (CH0 - Proximal) & Sensor 2 (CH1 - Distal)
  sensor1Available = initSensorOnChannel(CH_SENSOR_1, "Sensor 1 (Proximal CH0)");
  sensor2Available = initSensorOnChannel(CH_SENSOR_2, "Sensor 2 (Distal CH1)");

  // Show status on OLED
  oled.clearBuffer();
  oled.drawStr(0, 15, "Sensors (CH0/CH1):");
  char statusLine[32];
  snprintf(statusLine, sizeof(statusLine), "S1 Prox: %s", sensor1Available ? "OK" : "ERROR");
  oled.drawStr(0, 32, statusLine);
  snprintf(statusLine, sizeof(statusLine), "S2 Dist: %s", sensor2Available ? "OK" : "ERROR");
  oled.drawStr(0, 48, statusLine);
  oled.drawStr(0, 62, "Waiting for BLE...");
  oled.sendBuffer();

  delay(1500);
  Serial.println("AKIRA Dual IMU Platform Ready!");
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

      // 1. Synchronously sample Sensor 1 (PCA9548A CH0)
      if (sensor1Available && selectChannel(CH_SENSOR_1)) {
        if (readMPU6500(sensor1Data)) {
          packet.s1_ax = sensor1Data.ax;
          packet.s1_ay = sensor1Data.ay;
          packet.s1_az = sensor1Data.az;
          packet.s1_gx = sensor1Data.gx;
          packet.s1_gy = sensor1Data.gy;
          packet.s1_gz = sensor1Data.gz;
        }
      }

      // 2. Synchronously sample Sensor 2 (PCA9548A CH1)
      if (sensor2Available && selectChannel(CH_SENSOR_2)) {
        if (readMPU6500(sensor2Data)) {
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
      if (sensor1Available && selectChannel(CH_SENSOR_1)) {
        readMPU6500(sensor1Data);
      }
      if (sensor2Available && selectChannel(CH_SENSOR_2)) {
        readMPU6500(sensor2Data);
      }
      displaySensorData();
      lastDisplayUpdate = now;
    }
  }
}