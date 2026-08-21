package com.rehabresearch.datacollector.data.local.entity

import androidx.room.Entity
import androidx.room.PrimaryKey

enum class Gender { MALE, FEMALE, OTHER }

enum class SurgeryType { HIP_REPLACEMENT, KNEE_REPLACEMENT, OTHER }

enum class BodySide { LEFT, RIGHT, BILATERAL }

@Entity(tableName = "patients")
data class PatientEntity(
    @PrimaryKey val patientId: String, // e.g. "P001" — human-readable, therapist-assigned
    val name: String,
    val age: Int,
    val gender: Gender,
    val heightCm: Float,
    val weightKg: Float,
    val surgeryType: SurgeryType,
    val surgerySide: BodySide,
    val surgeryDateEpochDay: Long,
    val physiotherapistName: String,
    val createdAtEpochMillis: Long = System.currentTimeMillis()
) {
    /** BMI = kg / m^2, computed on the fly rather than stored, since height/weight can be edited. */
    val bmi: Float
        get() {
            val heightM = heightCm / 100f
            return if (heightM > 0f) weightKg / (heightM * heightM) else 0f
        }
}
