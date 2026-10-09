package com.rehabresearch.datacollector.ble

data class ImuPacket(
    val timestampMillis: Long,

    // Accelerometer
    val ax: Float,
    val ay: Float,
    val az: Float,

    // Gyroscope
    val gx: Float,
    val gy: Float,
    val gz: Float,

    // Magnetometer (optional)
    val mx: Float? = null,
    val my: Float? = null,
    val mz: Float? = null,

    // Quaternion (optional)
    val quatW: Float? = null,
    val quatX: Float? = null,
    val quatY: Float? = null,
    val quatZ: Float? = null,

    // Temperature (optional)
    val temperatureC: Float? = null
)

sealed class BleConnectionState {

    object Disconnected : BleConnectionState()

    object Scanning : BleConnectionState()

    data class DeviceFound(
        val name: String,
        val address: String,
        val rssi: Int
    ) : BleConnectionState()

    object Connecting : BleConnectionState()

    object DiscoveringServices : BleConnectionState()

    object EnablingNotifications : BleConnectionState()

    data class Connected(
        val name: String,
        val address: String
    ) : BleConnectionState()

    data class Error(
        val message: String
    ) : BleConnectionState()
}

data class BleLinkStats(
    val batteryPercent: Int? = null,
    val rssi: Int = 0,
    val packetsReceived: Long = 0,
    val packetsDropped: Long = 0,
    val currentSampleRateHz: Float = 0f
)