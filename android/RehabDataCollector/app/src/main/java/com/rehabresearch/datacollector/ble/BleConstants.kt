package com.rehabresearch.datacollector.ble

import java.util.UUID

object BleConstants {

    // ===========================
    // Device
    // ===========================

    const val DEVICE_NAME_PREFIX = "ESP32-REHAB"

    const val SCAN_TIMEOUT_MILLIS = 15000L
    const val CONNECTION_TIMEOUT_MILLIS = 10000L

    // ===========================
    // BLE UUIDs
    // MUST MATCH ESP32
    // ===========================

    val IMU_SERVICE_UUID: UUID =
        UUID.fromString("6E400001-B5A3-F393-E0A9-E50E24DCCA9E")

    val IMU_DATA_CHARACTERISTIC_UUID: UUID =
        UUID.fromString("6E400002-B5A3-F393-E0A9-E50E24DCCA9E")

    val DEVICE_STATUS_CHARACTERISTIC_UUID: UUID =
        UUID.fromString("6E400003-B5A3-F393-E0A9-E50E24DCCA9E")

    val CLIENT_CHARACTERISTIC_CONFIG_UUID: UUID =
        UUID.fromString("00002902-0000-1000-8000-00805F9B34FB")

    // ===========================
    // Packet Format
    // ===========================

    const val EXPECTED_PACKET_SIZE_BYTES = 28

    // ===========================
    // Connection
    // ===========================

    const val MTU_SIZE = 247

    const val RSSI_REFRESH_INTERVAL = 2000L

    // ===========================
    // Debug
    // ===========================

    const val LOG_TAG = "BLE"
}