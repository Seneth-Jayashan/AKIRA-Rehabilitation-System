package com.rehabresearch.datacollector.data.local.entity

import androidx.room.Entity
import androidx.room.ForeignKey
import androidx.room.Index
import androidx.room.PrimaryKey

/**
 * One row per IMU packet received over BLE. At 100 Hz a 60-second session
 * produces 6000 rows — Room/SQLite handles this fine, but always insert in
 * batches (see SensorDataDao.insertBatch) rather than one-by-one.
 */
@Entity(
    tableName = "sensor_readings",
    foreignKeys = [
        ForeignKey(
            entity = SessionEntity::class,
            parentColumns = ["sessionId"],
            childColumns = ["sessionId"],
            onDelete = ForeignKey.CASCADE
        )
    ],
    indices = [Index("sessionId")]
)
data class SensorReadingEntity(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val sessionId: String,
    val timestampMillis: Long, // device-relative timestamp from ESP32, not wall clock
    val ax: Float,
    val ay: Float,
    val az: Float,
    val gx: Float,
    val gy: Float,
    val gz: Float,
    val quatW: Float? = null,
    val quatX: Float? = null,
    val quatY: Float? = null,
    val quatZ: Float? = null,
    val temperatureC: Float? = null
)
