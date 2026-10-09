package com.rehabresearch.datacollector.data.local.dao

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import androidx.room.Update
import com.rehabresearch.datacollector.data.local.entity.SessionEntity
import com.rehabresearch.datacollector.data.local.entity.SessionStatus
import kotlinx.coroutines.flow.Flow

@Dao
interface SessionDao {

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insert(session: SessionEntity)

    @Update
    suspend fun update(session: SessionEntity)

    @Query("SELECT * FROM sessions WHERE sessionId = :sessionId")
    suspend fun getById(sessionId: String): SessionEntity?

    @Query("SELECT * FROM sessions WHERE sessionId = :sessionId")
    fun observeById(sessionId: String): Flow<SessionEntity?>

    @Query("SELECT * FROM sessions ORDER BY startedAtEpochMillis DESC")
    fun observeAll(): Flow<List<SessionEntity>>

    @Query("SELECT * FROM sessions WHERE patientId = :patientId ORDER BY startedAtEpochMillis DESC")
    fun observeForPatient(patientId: String): Flow<List<SessionEntity>>

    @Query("SELECT COUNT(*) FROM sessions WHERE startedAtEpochMillis >= :startOfDayMillis")
    suspend fun countSessionsToday(startOfDayMillis: Long): Int

    @Query("SELECT COUNT(*) FROM sessions")
    suspend fun countAll(): Int

    @Query("SELECT * FROM sessions WHERE status = :status ORDER BY startedAtEpochMillis ASC")
    suspend fun getByStatus(status: SessionStatus): List<SessionEntity>

    @Query("DELETE FROM sessions WHERE sessionId = :sessionId")
    suspend fun delete(sessionId: String)
}
