package com.rehabresearch.datacollector.data.repository

import com.rehabresearch.datacollector.data.local.dao.SensorDataDao
import com.rehabresearch.datacollector.data.local.dao.SessionDao
import com.rehabresearch.datacollector.data.local.entity.SensorReadingEntity
import com.rehabresearch.datacollector.data.local.entity.SessionEntity
import com.rehabresearch.datacollector.data.local.entity.SessionStatus
import kotlinx.coroutines.flow.Flow
import java.time.LocalDate
import java.time.ZoneOffset
import java.util.UUID
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class SessionRepository @Inject constructor(
    private val sessionDao: SessionDao,
    private val sensorDataDao: SensorDataDao
) {
    fun observeAll(): Flow<List<SessionEntity>> = sessionDao.observeAll()

    fun observeForPatient(patientId: String): Flow<List<SessionEntity>> =
        sessionDao.observeForPatient(patientId)

    fun observeById(sessionId: String): Flow<SessionEntity?> = sessionDao.observeById(sessionId)

    suspend fun getById(sessionId: String): SessionEntity? = sessionDao.getById(sessionId)

    fun newSessionId(): String = UUID.randomUUID().toString()

    suspend fun startSession(session: SessionEntity) = sessionDao.insert(session)

    suspend fun updateSession(session: SessionEntity) = sessionDao.update(session)

    /**
     * Buffered write path for the live BLE stream. The RecordingViewModel should
     * accumulate packets in memory and flush in batches (e.g. every 50-100 samples
     * or every 500ms, whichever comes first) rather than calling this per-packet.
     */
    suspend fun appendSensorBatch(readings: List<SensorReadingEntity>) {
        if (readings.isNotEmpty()) sensorDataDao.insertBatch(readings)
    }

    suspend fun getSensorReadings(sessionId: String): List<SensorReadingEntity> =
        sensorDataDao.getForSession(sessionId)

    suspend fun getSampleCount(sessionId: String): Long = sensorDataDao.countForSession(sessionId)

    suspend fun countToday(): Int {
        val startOfDay = LocalDate.now().atStartOfDay(ZoneOffset.systemDefault()).toInstant().toEpochMilli()
        return sessionDao.countSessionsToday(startOfDay)
    }

    /** Sessions that finished recording but haven't had their CSV exported yet. */
    suspend fun getUnexportedSessions(): List<SessionEntity> =
        sessionDao.getByStatus(SessionStatus.COMPLETED)
}
