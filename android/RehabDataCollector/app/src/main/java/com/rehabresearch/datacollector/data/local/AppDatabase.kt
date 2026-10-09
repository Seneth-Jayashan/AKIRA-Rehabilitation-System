package com.rehabresearch.datacollector.data.local

import androidx.room.Database
import androidx.room.RoomDatabase
import androidx.room.TypeConverters
import com.rehabresearch.datacollector.data.local.dao.PatientDao
import com.rehabresearch.datacollector.data.local.dao.SensorDataDao
import com.rehabresearch.datacollector.data.local.dao.SessionDao
import com.rehabresearch.datacollector.data.local.entity.PatientEntity
import com.rehabresearch.datacollector.data.local.entity.SensorReadingEntity
import com.rehabresearch.datacollector.data.local.entity.SessionEntity

@Database(
    entities = [PatientEntity::class, SessionEntity::class, SensorReadingEntity::class],
    version = 1,
    exportSchema = true // schema history matters for a research dataset app — never turn this off
)
@TypeConverters(Converters::class)
abstract class AppDatabase : RoomDatabase() {
    abstract fun patientDao(): PatientDao
    abstract fun sessionDao(): SessionDao
    abstract fun sensorDataDao(): SensorDataDao

    companion object {
        const val DB_NAME = "rehab_data_collector.db"
    }
}
