package com.rehabresearch.datacollector.data.local.entity

import androidx.room.Entity
import androidx.room.ForeignKey
import androidx.room.Index
import androidx.room.PrimaryKey
import com.rehabresearch.datacollector.data.local.entity.BodySide

enum class ExerciseType {
    // Legacy exercises (kept for backwards compatibility with existing local database rows)
    STRAIGHT_LEG_RAISE,
    HEEL_SLIDE,
    HIP_ABDUCTION,
    HIP_FLEXION,
    MINI_SQUAT,
    SIT_TO_STAND,
    WALKING,
    STEP_UP,
    STANDING_BALANCE,

    // Wrist Component
    WRIST_FLEXION_AND_EXTENSION,
    RADIAL_AND_ULNAR_DEVIATION,
    WRIST_PRONATION_AND_SUPINATION,
    GRIPS_RELATED_MOVEMENTS,

    // Knee Component
    QUADRICEPS_SETS,
    SEATED_KNEE_EXTENSIONS,

    // Ankle Component
    ANKLE_PUMPS,
    ANKLE_CIRCLES,
    ANKLE_ALPHABET,
    DORSIFLEXION,
    PLANTARFLEXION,
    SEATED_HEEL_RAISES,
    STANDING_HEEL_RAISES,
    SINGLE_LEG_BALANCE,

    // Elbow Component
    ELBOW_FLEXION,
    ELBOW_EXTENSION,
    ASSISTED_ELBOW_RANGE_OF_MOTION,
    FOREARM_PRONATION,
    FOREARM_SUPINATION,
    BICEPS_CURL,
    WALL_PUSH_UP
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
