#include <Wire.h>
#include <U8g2lib.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>

#define SERVICE_UUID           "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
#define DATA_CHAR_UUID         "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"
#define STATUS_CHAR_UUID       "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"

// =====================================
// I2C ADDRESSES
// =====================================

#define PCA9548A_ADDR 0x70
#define MPU6500_ADDR  0x68

// =====================================
// MPU6500 REGISTERS
// =====================================

#define WHO_AM_I_REG   0x75
#define PWR_MGMT_1     0x6B
#define ACCEL_XOUT_H   0x3B
#define GYRO_XOUT_H    0x43

// =====================================
// OLED
// =====================================

U8G2_SH1106_128X64_NONAME_F_HW_I2C oled(U8G2_R0, U8X8_PIN_NONE);

// =====================================
// SENSOR DATA
// =====================================

float ax;
float ay;
float az;

float gx;
float gy;
float gz;

struct __attribute__((packed)) ImuPacket {
    uint32_t timestamp;
    float ax;
    float ay;
    float az;
    float gx;
    float gy;
    float gz;
};

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
// SELECT PCA9548A CHANNEL
// =====================================

bool selectChannel(uint8_t channel) {
  Wire.beginTransmission(PCA9548A_ADDR);
  Wire.write(1 << channel);
  uint8_t error = Wire.endTransmission(true);
  if (error != 0) {
    return false;
  }
  delay(5);
  return true;
}

// =====================================
// WRITE MPU6500 REGISTER
// =====================================

bool writeRegister(uint8_t reg, uint8_t value) {
  Wire.beginTransmission(MPU6500_ADDR);
  Wire.write(reg);
  Wire.write(value);
  return Wire.endTransmission(true) == 0;
}

// =====================================
// READ MULTIPLE BYTES
// =====================================

bool readBytes(uint8_t reg, uint8_t *buffer, uint8_t length) {
  Wire.beginTransmission(MPU6500_ADDR);
  Wire.write(reg);
  if (Wire.endTransmission(true) != 0) {
    return false;
  }
  delayMicroseconds(100);
  uint8_t received = Wire.requestFrom(MPU6500_ADDR, length);
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

int16_t combineBytes(uint8_t high, uint8_t low) {
  return (int16_t)((high << 8) | low);
}

// =====================================
// READ MPU6500
// =====================================

bool readMPU6500() {
  uint8_t accelData[6];
  uint8_t gyroData[6];

  if (!readBytes(ACCEL_XOUT_H, accelData, 6)) {
    return false;
  }
  if (!readBytes(GYRO_XOUT_H, gyroData, 6)) {
    return false;
  }

  int16_t rawAx = combineBytes(accelData[0], accelData[1]);
  int16_t rawAy = combineBytes(accelData[2], accelData[3]);
  int16_t rawAz = combineBytes(accelData[4], accelData[5]);

  int16_t rawGx = combineBytes(gyroData[0], gyroData[1]);
  int16_t rawGy = combineBytes(gyroData[2], gyroData[3]);
  int16_t rawGz = combineBytes(gyroData[4], gyroData[5]);

  // Accelerometer ±2g
  ax = rawAx / 16384.0;
  ay = rawAy / 16384.0;
  az = rawAz / 16384.0;

  // Gyroscope ±250 °/s
  gx = rawGx / 131.0;
  gy = rawGy / 131.0;
  gz = rawGz / 131.0;

  return true;
}

// =====================================
// OLED DISPLAY
// =====================================

void displaySensorData() {
  oled.clearBuffer();

  // Title
  oled.setFont(u8g2_font_6x10_tf);
  
  if (deviceConnected) {
    oled.drawStr(0, 9, "BLE: Connected");
  } else {
    oled.drawStr(0, 9, "BLE: Disconnected");
  }

  oled.drawHLine(0, 12, 128);

  // Accelerometer
  oled.drawStr(0, 23, "ACC (g)");
  char line[32];
  snprintf(line, sizeof(line), "X:%5.2f Y:%5.2f", ax, ay);
  oled.drawStr(0, 34, line);
  snprintf(line, sizeof(line), "Z:%5.2f", az);
  oled.drawStr(0, 44, line);

  // Gyroscope
  oled.drawStr(0, 55, "GYRO (deg/s)");
  snprintf(line, sizeof(line), "X:%4.0f Y:%4.0f Z:%4.0f", gx, gy, gz);
  oled.drawStr(0, 64, line);

  oled.sendBuffer();
}

void displayStreamingScreen() {
  oled.clearBuffer();
  oled.setFont(u8g2_font_6x10_tf);
  oled.drawStr(0, 20, "BLE: Connected");
  oled.drawStr(0, 40, "Streaming data...");
  oled.drawStr(0, 55, "See app for values");
  oled.sendBuffer();
}

// =====================================
// SETUP
// =====================================

void setup() {
  Serial.begin(115200);
  delay(1000);

  // ESP32 I2C
  Wire.begin(21, 22);
  Wire.setClock(100000);

  // OLED
  oled.begin();
  oled.clearBuffer();
  oled.setFont(u8g2_font_6x10_tf);
  oled.drawStr(0, 20, "Initializing...");
  oled.drawStr(0, 40, "MPU6500 CH0");
  oled.sendBuffer();

  Serial.println();
  Serial.println("======================================");
  Serial.println("ESP32 REHAB SENSOR SYSTEM");
  Serial.println("======================================");

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
  pStatusCharacteristic->setValue("OK");

  pService->start();

  BLEAdvertising *pAdvertising = BLEDevice::getAdvertising();
  pAdvertising->addServiceUUID(SERVICE_UUID);
  pAdvertising->setScanResponse(true);
  pAdvertising->setMinPreferred(0x06);
  pAdvertising->setMinPreferred(0x12);
  BLEDevice::startAdvertising();

  Serial.println("BLE Ready!");

  // Select CH0
  if (!selectChannel(0)) {
    Serial.println("ERROR: PCA9548A CH0 failed");
    oled.clearBuffer();
    oled.drawStr(0, 25, "PCA9548A ERROR");
    oled.sendBuffer();
    return;
  }

  // Wake MPU6500
  if (!writeRegister(PWR_MGMT_1, 0x00)) {
    Serial.println("ERROR: MPU6500 wake failed");
    oled.clearBuffer();
    oled.drawStr(0, 25, "MPU6500 ERROR");
    oled.sendBuffer();
    return;
  }

  delay(100);
  Serial.println("System Ready!");
}

// =====================================
// LOOP
// =====================================

void loop() {
  static unsigned long lastSampleTime = 0;
  
  if (deviceConnected) {
    // connecting state change
    if (!oldDeviceConnected) {
      oldDeviceConnected = true;
      // Show static streaming screen so we don't block the loop updating the OLED
      displayStreamingScreen();
    }

    unsigned long now = millis();
    // 10ms interval = 100Hz sampling rate
    if (now - lastSampleTime >= 10) {
      lastSampleTime = now;

      if (!selectChannel(0)) {
        Serial.println("CH0 selection failed");
        return; // Skip this cycle
      }

      if (readMPU6500()) {
        // Create and send packet
        ImuPacket packet;
        packet.timestamp = now;
        packet.ax = ax;
        packet.ay = ay;
        packet.az = az;
        packet.gx = gx;
        packet.gy = gy;
        packet.gz = gz;

        pDataCharacteristic->setValue((uint8_t*)&packet, sizeof(ImuPacket));
        pDataCharacteristic->notify();
        
      } else {
        Serial.println("ERROR: MPU6500 read failed");
      }
    }
    
  } else {
    // disconnecting state change
    if (oldDeviceConnected) {
      delay(500); // give the bluetooth stack the chance to get things ready
      pServer->startAdvertising(); // restart advertising
      Serial.println("start advertising");
      oldDeviceConnected = false;
    }
    
    // If not connected, update the display occasionally
    unsigned long now = millis();
    static unsigned long lastDisplayUpdate = 0;
    
    // Show live sensor data on OLED at 2Hz when disconnected
    if (now - lastDisplayUpdate > 500) {
      if (selectChannel(0)) {
        readMPU6500();
      }
      displaySensorData();
      lastDisplayUpdate = now;
    }
  }
}