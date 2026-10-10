package com.rehabresearch.datacollector.data.local.dao

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import com.rehabresearch.datacollector.data.local.entity.SensorReadingEntity

@Dao
interface SensorDataDao {

    // Sensor writes happen at 50-100Hz — ALWAYS batch insert, never insert one row at a time
    // or you will fall behind the incoming BLE stream and drop packets.
    @Insert(onConflict = OnConflictStrategy.IGNORE)
    suspend fun insertBatch(readings: List<SensorReadingEntity>)

    @Query("SELECT * FROM sensor_readings WHERE sessionId = :sessionId ORDER BY timestampMillis ASC")
    suspend fun getForSession(sessionId: String): List<SensorReadingEntity>

    @Query("SELECT COUNT(*) FROM sensor_readings WHERE sessionId = :sessionId")
    suspend fun countForSession(sessionId: String): Long

    @Query("DELETE FROM sensor_readings WHERE sessionId = :sessionId")
    suspend fun deleteForSession(sessionId: String)
}
