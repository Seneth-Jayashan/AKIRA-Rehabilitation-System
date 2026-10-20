/*
 * ESP32 Dual-IMU Rehabilitation Wearable Firmware
 * 
 * Target: ESP32-WROOM-32 / ESP32-S3
 * Sensors: Dual 6-Axis IMUs (MPU-6050 / ICM-42688 / BMI270)
 *   - IMU 1 (Upper Arm, RUA): I2C Address 0x68 (AD0 -> GND)
 *   - IMU 2 (Forearm, RLA):   I2C Address 0x69 (AD0 -> 3.3V)
 * Output Transports:
 *   - USB-Serial @ 115200 baud (Web Serial API compatible)
 *   - Bluetooth BLE / SPP @ 100 Hz
 * 
 * Packet: 64-Byte Binary Packed Struct (< = Little Endian)
 * Magic Header: 0xAA55AA55
 */

#include <Wire.h>

#define SERIAL_BAUD 115200
#define SAMPLE_INTERVAL_MS 10 // 100 Hz

#define IMU1_ADDR 0x68 // Upper Arm (RUA)
#define IMU2_ADDR 0x69 // Forearm (RLA)

#define MAGIC_HEADER 0xAA55AA55

// 64-Byte Binary Packet Struct
#pragma pack(push, 1)
struct SensorPacket {
  uint32_t magic;         // 0xAA55AA55 (4 bytes)
  uint32_t timestamp_ms;  // Milliseconds since boot (4 bytes)
  uint32_t seq_num;       // Monotonic sequence number (4 bytes)
  
  // Upper Arm IMU (RUA) - 24 bytes
  float rua_ax; // m/s^2
  float rua_ay;
  float rua_az;
  float rua_gx; // rad/s
  float rua_gy;
  float rua_gz;

  // Forearm IMU (RLA) - 24 bytes
  float rla_ax; // m/s^2
  float rla_ay;
  float rla_az;
  float rla_gx; // rad/s
  float rla_gy;
  float rla_gz;

  uint16_t status_flags;  // Bit 0: IMU1 err, Bit 1: IMU2 err (2 bytes)
  uint16_t checksum;      // 16-bit additive checksum (2 bytes)
};
#pragma pack(pop)

SensorPacket packet;
uint32_t sequence_counter = 0;
unsigned long last_sample_time = 0;
bool imu1_ok = false;
bool imu2_ok = false;

// Conversion Constants
const float ACC_SCALE_G_TO_MPS2 = 9.80665f / 16384.0f; // +-2g default
const float GYR_SCALE_TO_RADS   = (3.14159265f / 180.0f) / 131.0f; // +-250 deg/s default

bool checkI2C(uint8_t addr) {
  Wire.beginTransmission(addr);
  return (Wire.endTransmission() == 0);
}

void initIMU(uint8_t addr) {
  Wire.beginTransmission(addr);
  Wire.write(0x6B); // PWR_MGMT_1
  Wire.write(0x00); // Wake up MPU6050
  Wire.endTransmission(true);
}

void readIMU(uint8_t addr, float &ax, float &ay, float &az, float &gx, float &gy, float &gz) {
  Wire.beginTransmission(addr);
  Wire.write(0x3B); // ACCEL_XOUT_H
  Wire.endTransmission(false);
  Wire.requestFrom((int)addr, 14, true);

  if (Wire.available() >= 14) {
    int16_t raw_ax = (Wire.read() << 8) | Wire.read();
    int16_t raw_ay = (Wire.read() << 8) | Wire.read();
    int16_t raw_az = (Wire.read() << 8) | Wire.read();
    int16_t raw_temp = (Wire.read() << 8) | Wire.read(); // unused
    int16_t raw_gx = (Wire.read() << 8) | Wire.read();
    int16_t raw_gy = (Wire.read() << 8) | Wire.read();
    int16_t raw_gz = (Wire.read() << 8) | Wire.read();

    ax = (float)raw_ax * ACC_SCALE_G_TO_MPS2;
    ay = (float)raw_ay * ACC_SCALE_G_TO_MPS2;
    az = (float)raw_az * ACC_SCALE_G_TO_MPS2;

    gx = (float)raw_gx * GYR_SCALE_TO_RADS;
    gy = (float)raw_gy * GYR_SCALE_TO_RADS;
    gz = (float)raw_gz * GYR_SCALE_TO_RADS;
  }
}

uint16_t computeChecksum(const uint8_t *data, size_t len) {
  uint16_t sum = 0;
  for (size_t i = 0; i < len; i++) {
    sum += data[i];
  }
  return sum;
}

void setup() {
  Serial.begin(SERIAL_BAUD);
  while (!Serial && millis() < 1000);

  Wire.begin(21, 22); // SDA=21, SCL=22
  Wire.setClock(400000); // 400 kHz fast I2C

  imu1_ok = checkI2C(IMU1_ADDR);
  if (imu1_ok) initIMU(IMU1_ADDR);

  imu2_ok = checkI2C(IMU2_ADDR);
  if (imu2_ok) initIMU(IMU2_ADDR);

  packet.magic = MAGIC_HEADER;
}

void loop() {
  // Handle handshake requests from host
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    if (cmd == "PING" || cmd == "HANDSHAKE") {
      // Respond with verified device identity and dual-IMU status
      Serial.print("ACK:ESP32_DUAL_IMU;IMU1=");
      Serial.print(imu1_ok ? "OK" : "ERR");
      Serial.print(";IMU2=");
      Serial.print(imu2_ok ? "OK" : "ERR");
      Serial.println(";RATE=100HZ");
      return;
    }
  }

  unsigned long current_time = millis();
  if (current_time - last_sample_time >= SAMPLE_INTERVAL_MS) {
    last_sample_time = current_time;

    packet.timestamp_ms = current_time;
    packet.seq_num = sequence_counter++;
    packet.status_flags = 0;

    if (imu1_ok) {
      readIMU(IMU1_ADDR, packet.rua_ax, packet.rua_ay, packet.rua_az, packet.rua_gx, packet.rua_gy, packet.rua_gz);
    } else {
      packet.status_flags |= 0x01; // IMU1 fault
      packet.rua_ax = packet.rua_ay = packet.rua_az = 0.0f;
      packet.rua_gx = packet.rua_gy = packet.rua_gz = 0.0f;
    }

    if (imu2_ok) {
      readIMU(IMU2_ADDR, packet.rla_ax, packet.rla_ay, packet.rla_az, packet.rla_gx, packet.rla_gy, packet.rla_gz);
    } else {
      packet.status_flags |= 0x02; // IMU2 fault
      packet.rla_ax = packet.rla_ay = packet.rla_az = 0.0f;
      packet.rla_gx = packet.rla_gy = packet.rla_gz = 0.0f;
    }

    // Checksum over the first 62 bytes (header, payload, status_flags)
    packet.checksum = computeChecksum((const uint8_t *)&packet, 62);

    // Stream binary packet over serial
    Serial.write((const uint8_t *)&packet, sizeof(SensorPacket));
  }
}
