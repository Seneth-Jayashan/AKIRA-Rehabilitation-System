package com.rehabresearch.datacollector.data.local.entity

import androidx.room.Entity
import androidx.room.ForeignKey
import androidx.room.Index
import androidx.room.PrimaryKey

/**
 * 4 Rehabilitation Components & Sensor Placements for the 4 Team Members:
 * - Member 1 (My Component - Ankle): Sensor 1 = Tibia (Lower Leg), Sensor 2 = Foot (Dorsum)
 * - Member 2 (Knee Component):       Sensor 1 = Thigh (Femur), Sensor 2 = Shank (Tibia)
 * - Member 3 (Wrist Component):      Sensor 1 = Wrist (Forearm), Sensor 2 = Hand (Dorsum)
 * - Member 4 (Elbow Component):      Sensor 1 = Upper Arm (Humerus), Sensor 2 = Forearm (Radius/Ulna)
 */
enum class RehabComponent(
    val displayName: String,
    val memberLabel: String,
    val sensor1Default: String, // Proximal IMU (PCA9548A CH0)
    val sensor2Default: String  // Distal IMU (PCA9548A CH1)
) {
    ANKLE(
        displayName = "Ankle Component",
        memberLabel = "Member 1",
        sensor1Default = "Tibia (Lower Leg)",
        sensor2Default = "Foot (Dorsum)"
    ),
    KNEE(
        displayName = "Knee Component",
        memberLabel = "Member 2",
        sensor1Default = "Thigh (Femur)",
        sensor2Default = "Shank (Tibia)"
    ),
    WRIST(
        displayName = "Wrist Component",
        memberLabel = "Member 3",
        sensor1Default = "Wrist (Forearm)",
        sensor2Default = "Hand (Dorsum)"
    ),
    ELBOW(
        displayName = "Elbow Component",
        memberLabel = "Member 4",
        sensor1Default = "Upper Arm (Humerus)",
        sensor2Default = "Forearm (Radius/Ulna)"
    )
}

enum class ExerciseType {
    // Member 1: Ankle Component
    DORSIFLEXION,
    PLANTARFLEXION,
    INVERSION,
    EVERSION,
    ANKLE_CIRCLES,
    ANKLE_ALPHABET,
    SEATED_HEEL_RAISES,
    STANDING_HEEL_RAISES,
    SINGLE_LEG_BALANCE,
    ANKLE_PUMPS,
    REST_STATIC,

    // Member 2: Knee Component
    QUADRICEPS_SETS,
    SEATED_KNEE_EXTENSIONS,
    STRAIGHT_LEG_RAISE,
    HEEL_SLIDE,
    HIP_ABDUCTION,
    HIP_FLEXION,
    MINI_SQUAT,
    SIT_TO_STAND,
    WALKING,
    STEP_UP,
    STANDING_BALANCE,

    // Member 3: Wrist Component
    WRIST_FLEXION_AND_EXTENSION,
    RADIAL_AND_ULNAR_DEVIATION,
    WRIST_PRONATION_AND_SUPINATION,
    GRIPS_RELATED_MOVEMENTS,

    // Member 4: Elbow Component
    ELBOW_FLEXION,
    ELBOW_EXTENSION,
    ASSISTED_ELBOW_RANGE_OF_MOTION,
    FOREARM_PRONATION,
    FOREARM_SUPINATION,
    BICEPS_CURL,
    WALL_PUSH_UP
}

val ExerciseType.component: RehabComponent
    get() = when (this) {
        ExerciseType.DORSIFLEXION,
        ExerciseType.PLANTARFLEXION,
        ExerciseType.INVERSION,
        ExerciseType.EVERSION,
        ExerciseType.ANKLE_CIRCLES,
        ExerciseType.ANKLE_ALPHABET,
        ExerciseType.SEATED_HEEL_RAISES,
        ExerciseType.STANDING_HEEL_RAISES,
        ExerciseType.SINGLE_LEG_BALANCE,
        ExerciseType.ANKLE_PUMPS,
        ExerciseType.REST_STATIC -> RehabComponent.ANKLE

        ExerciseType.QUADRICEPS_SETS,
        ExerciseType.SEATED_KNEE_EXTENSIONS,
        ExerciseType.STRAIGHT_LEG_RAISE,
        ExerciseType.HEEL_SLIDE,
        ExerciseType.HIP_ABDUCTION,
        ExerciseType.HIP_FLEXION,
        ExerciseType.MINI_SQUAT,
        ExerciseType.SIT_TO_STAND,
        ExerciseType.WALKING,
        ExerciseType.STEP_UP,
        ExerciseType.STANDING_BALANCE -> RehabComponent.KNEE

        ExerciseType.WRIST_FLEXION_AND_EXTENSION,
        ExerciseType.RADIAL_AND_ULNAR_DEVIATION,
        ExerciseType.WRIST_PRONATION_AND_SUPINATION,
        ExerciseType.GRIPS_RELATED_MOVEMENTS -> RehabComponent.WRIST

        ExerciseType.ELBOW_FLEXION,
        ExerciseType.ELBOW_EXTENSION,
        ExerciseType.ASSISTED_ELBOW_RANGE_OF_MOTION,
        ExerciseType.FOREARM_PRONATION,
        ExerciseType.FOREARM_SUPINATION,
        ExerciseType.BICEPS_CURL,
        ExerciseType.WALL_PUSH_UP -> RehabComponent.ELBOW
    }

enum class Difficulty { EASY, MEDIUM, HARD }

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
    val trialId: String = "Trial_1", // e.g. Trial_1, Trial_2, Trial_3
    val component: RehabComponent = RehabComponent.ANKLE,
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
    val sensor1Placement: String = component.sensor1Default, // e.g. Tibia, Thigh, Wrist, Upper Arm
    val sensor2Placement: String = component.sensor2Default, // e.g. Foot, Shank, Hand, Forearm
    val tibiaPlacement: String = sensor1Placement, // alias for backwards compatibility
    val footPlacement: String = sensor2Placement,  // alias for backwards compatibility
    val accelUnit: String = "g",
    val gyroUnit: String = "deg/s",
    val calibrationStatus: String = "Raw Uncalibrated",
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
