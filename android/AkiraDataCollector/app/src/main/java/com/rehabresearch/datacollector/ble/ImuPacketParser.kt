package com.rehabresearch.datacollector.ble

import android.util.Log
import java.nio.ByteBuffer
import java.nio.ByteOrder

object ImuPacketParser {

    fun parse(bytes: ByteArray): ImuPacket? {
        return try {
            val buffer = ByteBuffer
                .wrap(bytes)
                .order(ByteOrder.LITTLE_ENDIAN)

            when (bytes.size) {
                // 12-Channel Dual-IMU synchronized packet: 4 + 24 + 24 = 52 bytes
                BleConstants.DUAL_IMU_PACKET_SIZE_BYTES -> {
                    val timestamp = buffer.int.toLong() and 0xffffffffL

                    // Sensor 1 / Proximal Segment (PCA9548A CH0)
                    val s1Ax = buffer.float
                    val s1Ay = buffer.float
                    val s1Az = buffer.float
                    val s1Gx = buffer.float
                    val s1Gy = buffer.float
                    val s1Gz = buffer.float

                    // Sensor 2 / Distal Segment (PCA9548A CH1)
                    val s2Ax = buffer.float
                    val s2Ay = buffer.float
                    val s2Az = buffer.float
                    val s2Gx = buffer.float
                    val s2Gy = buffer.float
                    val s2Gz = buffer.float

                    ImuPacket(
                        timestampMillis = timestamp,
                        s1Ax = s1Ax, s1Ay = s1Ay, s1Az = s1Az,
                        s1Gx = s1Gx, s1Gy = s1Gy, s1Gz = s1Gz,
                        s2Ax = s2Ax, s2Ay = s2Ay, s2Az = s2Az,
                        s2Gx = s2Gx, s2Gy = s2Gy, s2Gz = s2Gz,
                        isDualSensor = true
                    )
                }

                // Legacy Single IMU packet: 4 + 24 = 28 bytes
                BleConstants.LEGACY_PACKET_SIZE_BYTES -> {
                    val timestamp = buffer.int.toLong() and 0xffffffffL

                    val ax = buffer.float
                    val ay = buffer.float
                    val az = buffer.float
                    val gx = buffer.float
                    val gy = buffer.float
                    val gz = buffer.float

                    ImuPacket(
                        timestampMillis = timestamp,
                        s1Ax = ax, s1Ay = ay, s1Az = az,
                        s1Gx = gx, s1Gy = gy, s1Gz = gz,
                        s2Ax = 0f, s2Ay = 0f, s2Az = 0f,
                        s2Gx = 0f, s2Gy = 0f, s2Gz = 0f,
                        isDualSensor = false
                    )
                }

                else -> {
                    Log.e(
                        BleConstants.LOG_TAG,
                        "Invalid packet size = ${bytes.size} (expected ${BleConstants.DUAL_IMU_PACKET_SIZE_BYTES} or ${BleConstants.LEGACY_PACKET_SIZE_BYTES})"
                    )
                    null
                }
            }
        } catch (e: Exception) {
            Log.e(
                BleConstants.LOG_TAG,
                "Parser error",
                e
            )
            null
        }
    }
}