package com.rehabresearch.datacollector.data.local.dao

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import androidx.room.Update
import com.rehabresearch.datacollector.data.local.entity.PatientEntity
import kotlinx.coroutines.flow.Flow

@Dao
interface PatientDao {

    @Insert(onConflict = OnConflictStrategy.ABORT)
    suspend fun insert(patient: PatientEntity)

    @Update
    suspend fun update(patient: PatientEntity)

    @Query("SELECT * FROM patients ORDER BY createdAtEpochMillis DESC")
    fun observeAll(): Flow<List<PatientEntity>>

    @Query("SELECT * FROM patients WHERE patientId = :patientId")
    suspend fun getById(patientId: String): PatientEntity?

    @Query("SELECT * FROM patients WHERE patientId = :patientId")
    fun observeById(patientId: String): Flow<PatientEntity?>

    @Query("SELECT COUNT(*) FROM patients")
    suspend fun count(): Int

    @Query("""
        SELECT COUNT(DISTINCT p.patientId) FROM patients p
        INNER JOIN sessions s ON s.patientId = p.patientId
        WHERE s.startedAtEpochMillis >= :startOfDayMillis
    """)
    suspend fun countPatientsSeenToday(startOfDayMillis: Long): Int

    @Query("DELETE FROM patients WHERE patientId = :patientId")
    suspend fun delete(patientId: String)
}
