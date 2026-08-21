package com.rehabresearch.datacollector.ble

import android.util.Log
import java.nio.ByteBuffer
import java.nio.ByteOrder

object ImuPacketParser {

    fun parse(bytes: ByteArray): ImuPacket? {

        if (bytes.size != BleConstants.EXPECTED_PACKET_SIZE_BYTES) {

            Log.e(
                BleConstants.LOG_TAG,
                "Invalid packet size = ${bytes.size}"
            )

            return null
        }

        return try {

            val buffer = ByteBuffer
                .wrap(bytes)
                .order(ByteOrder.LITTLE_ENDIAN)

            val timestamp = buffer.int.toLong() and 0xffffffffL

            val ax = buffer.float
            val ay = buffer.float
            val az = buffer.float

            val gx = buffer.float
            val gy = buffer.float
            val gz = buffer.float

            ImuPacket(
                timestampMillis = timestamp,
                ax = ax,
                ay = ay,
                az = az,
                gx = gx,
                gy = gy,
                gz = gz
            )

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