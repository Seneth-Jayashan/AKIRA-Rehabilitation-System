package com.rehabresearch.datacollector.ble

/**
 * Represents a synchronized 12-channel IMU sample packet.
 * Captures 2 paired sensors for the 4 rehabilitation components:
 * - Member 1 (Ankle): Sensor 1 = Tibia (CH0),     Sensor 2 = Foot (CH1)
 * - Member 2 (Knee):  Sensor 1 = Thigh (CH0),     Sensor 2 = Shank (CH1)
 * - Member 3 (Wrist): Sensor 1 = Wrist (CH0),     Sensor 2 = Hand (CH1)
 * - Member 4 (Elbow): Sensor 1 = Upper Arm (CH0), Sensor 2 = Forearm (CH1)
 */
data class ImuPacket(
    val timestampMillis: Long,

    // Sensor 1 / Proximal Segment (PCA9548A CH0)
    val s1Ax: Float,
    val s1Ay: Float,
    val s1Az: Float,
    val s1Gx: Float,
    val s1Gy: Float,
    val s1Gz: Float,

    // Sensor 2 / Distal Segment (PCA9548A CH1)
    val s2Ax: Float = 0f,
    val s2Ay: Float = 0f,
    val s2Az: Float = 0f,
    val s2Gx: Float = 0f,
    val s2Gy: Float = 0f,
    val s2Gz: Float = 0f,

    // Aliases for Member 1 (Ankle: Tibia & Foot) for backward compatibility
    val tibiaAx: Float = s1Ax,
    val tibiaAy: Float = s1Ay,
    val tibiaAz: Float = s1Az,
    val tibiaGx: Float = s1Gx,
    val tibiaGy: Float = s1Gy,
    val tibiaGz: Float = s1Gz,

    val footAx: Float = s2Ax,
    val footAy: Float = s2Ay,
    val footAz: Float = s2Az,
    val footGx: Float = s2Gx,
    val footGy: Float = s2Gy,
    val footGz: Float = s2Gz,

    // General single-sensor aliases
    val ax: Float = s1Ax,
    val ay: Float = s1Ay,
    val az: Float = s1Az,
    val gx: Float = s1Gx,
    val gy: Float = s1Gy,
    val gz: Float = s1Gz,

    val isDualSensor: Boolean = true,

    // Optional fields
    val mx: Float? = null,
    val my: Float? = null,
    val mz: Float? = null,
    val quatW: Float? = null,
    val quatX: Float? = null,
    val quatY: Float? = null,
    val quatZ: Float? = null,
    val temperatureC: Float? = null
)

sealed class BleConnectionState {
    object Disconnected : BleConnectionState()
    object Scanning : BleConnectionState()
    data class DeviceFound(val name: String, val address: String, val rssi: Int) : BleConnectionState()
    object Connecting : BleConnectionState()
    object DiscoveringServices : BleConnectionState()
    object EnablingNotifications : BleConnectionState()
    data class Connected(val name: String, val address: String) : BleConnectionState()
    data class Error(val message: String) : BleConnectionState()
}

data class BleLinkStats(
    val batteryPercent: Int? = null,
    val rssi: Int = 0,
    val packetsReceived: Long = 0,
    val packetsDropped: Long = 0,
    val currentSampleRateHz: Float = 0f
)