package com.rehabresearch.datacollector.data.local.entity

import androidx.room.Entity
import androidx.room.ForeignKey
import androidx.room.Index
import androidx.room.PrimaryKey
import com.rehabresearch.datacollector.data.local.entity.BodySide

enum class ExerciseType {
    STRAIGHT_LEG_RAISE,
    HEEL_SLIDE,
    HIP_ABDUCTION,
    HIP_FLEXION,
    MINI_SQUAT,
    SIT_TO_STAND,
    WALKING,
    STEP_UP,
    STANDING_BALANCE
}

enum class Difficulty { EASY, MEDIUM, HARD }

/**
 * RECORDING: BLE stream is actively writing sensor_readings.
 * COMPLETED: recording stopped, but the therapist hasn't saved post-session
 *            labels / exported the CSV yet.
 * EXPORTED: labels saved and CSV + metadata sidecar written to disk — this
 *           is the terminal state for a local-only research tool (no cloud).
 */
enum class SessionStatus { RECORDING, COMPLETED, EXPORTED }

@Entity(
    tableName = "sessions",
    foreignKeys = [
        ForeignKey(
            entity = PatientEntity::class,
            parentColumns = ["patientId"],
            childColumns = ["patientId"],
            onDelete = ForeignKey.CASCADE
        )
    ],
    indices = [Index("patientId")]
)
data class SessionEntity(
    @PrimaryKey val sessionId: String, // UUID
    val patientId: String,
    val exercise: ExerciseType,
    val side: BodySide,
    val difficulty: Difficulty,
    val recoveryWeek: Int,
    val targetReps: Int,
    val actualReps: Int = 0,
    val therapistName: String,
    val startedAtEpochMillis: Long,
    val endedAtEpochMillis: Long? = null,
    val durationMillis: Long = 0,
    val avgSampleFrequencyHz: Float = 0f,
    val packetCount: Long = 0,
    val droppedPacketCount: Long = 0,
    val maxAngleDegrees: Float? = null,
    val painLevel: Int? = null, // 0-10
    val assistiveDevice: String? = null,
    val correctMovement: Boolean? = null,
    val compensationObserved: Boolean = false,
    val therapistNotes: String = "",
    val videoFilePath: String? = null,
    val csvFilePath: String? = null,
    val status: SessionStatus = SessionStatus.RECORDING
)
