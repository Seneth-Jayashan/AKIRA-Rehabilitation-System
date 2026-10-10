package com.rehabresearch.datacollector.di

import android.content.Context
import androidx.room.Room
import com.rehabresearch.datacollector.data.local.AppDatabase
import com.rehabresearch.datacollector.data.local.dao.PatientDao
import com.rehabresearch.datacollector.data.local.dao.SensorDataDao
import com.rehabresearch.datacollector.data.local.dao.SessionDao
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.android.qualifiers.ApplicationContext
import dagger.hilt.components.SingletonComponent
import javax.inject.Singleton

@Module
@InstallIn(SingletonComponent::class)
object DatabaseModule {

    @Provides
    @Singleton
    fun provideDatabase(@ApplicationContext context: Context): AppDatabase =
        Room.databaseBuilder(context, AppDatabase::class.java, AppDatabase.DB_NAME)
            // Research data should never be silently wiped by a schema bump during dev.
            // Replace this with a real Migration once the schema stabilizes.
            .fallbackToDestructiveMigration()
            .build()

    @Provides
    fun providePatientDao(db: AppDatabase): PatientDao = db.patientDao()

    @Provides
    fun provideSessionDao(db: AppDatabase): SessionDao = db.sessionDao()

    @Provides
    fun provideSensorDataDao(db: AppDatabase): SensorDataDao = db.sensorDataDao()
}
