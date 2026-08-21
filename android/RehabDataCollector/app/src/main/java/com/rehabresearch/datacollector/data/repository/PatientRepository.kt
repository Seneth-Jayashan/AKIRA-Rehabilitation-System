package com.rehabresearch.datacollector.data.repository

import com.rehabresearch.datacollector.data.local.dao.PatientDao
import com.rehabresearch.datacollector.data.local.entity.PatientEntity
import kotlinx.coroutines.flow.Flow
import java.time.LocalDate
import java.time.ZoneOffset
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class PatientRepository @Inject constructor(
    private val patientDao: PatientDao
) {
    fun observeAll(): Flow<List<PatientEntity>> = patientDao.observeAll()

    fun observeById(patientId: String): Flow<PatientEntity?> = patientDao.observeById(patientId)

    suspend fun getById(patientId: String): PatientEntity? = patientDao.getById(patientId)

    suspend fun addPatient(patient: PatientEntity) = patientDao.insert(patient)

    suspend fun updatePatient(patient: PatientEntity) = patientDao.update(patient)

    suspend fun deletePatient(patientId: String) = patientDao.delete(patientId)

    /** Generates the next sequential patient ID, e.g. P001, P002, ... */
    suspend fun generateNextPatientId(): String {
        val count = patientDao.count()
        return "P%03d".format(count + 1)
    }

    suspend fun countSeenToday(): Int {
        val startOfDay = LocalDate.now().atStartOfDay(ZoneOffset.systemDefault()).toInstant().toEpochMilli()
        return patientDao.countPatientsSeenToday(startOfDay)
    }
}
