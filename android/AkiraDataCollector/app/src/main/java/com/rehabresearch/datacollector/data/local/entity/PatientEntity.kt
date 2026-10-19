package com.rehabresearch.datacollector.data.local.entity

import androidx.room.Entity
import androidx.room.PrimaryKey

enum class Gender { MALE, FEMALE, OTHER }

enum class SurgeryType(
    val displayName: String,
    val category: String,
    val component: RehabComponent?
) {
    // Member 1: Ankle Component Surgeries
    ANKLE_ORIF("Ankle ORIF (Fracture Fixation)", "Ankle (Member 1)", RehabComponent.ANKLE),
    ACHILLES_TENDON_REPAIR("Achilles Tendon Repair", "Ankle (Member 1)", RehabComponent.ANKLE),
    ANKLE_ARTHROSCOPY("Ankle Arthroscopy", "Ankle (Member 1)", RehabComponent.ANKLE),
    ANKLE_LIGAMENT_RECONSTRUCTION("Ankle Ligament (ATFL) Repair", "Ankle (Member 1)", RehabComponent.ANKLE),
    ANKLE_FUSION("Ankle Arthrodesis (Fusion)", "Ankle (Member 1)", RehabComponent.ANKLE),
    TOTAL_ANKLE_REPLACEMENT("Total Ankle Replacement", "Ankle (Member 1)", RehabComponent.ANKLE),

    // Member 2: Knee Component Surgeries
    TOTAL_KNEE_REPLACEMENT("Total Knee Replacement (TKR)", "Knee (Member 2)", RehabComponent.KNEE),
    PARTIAL_KNEE_REPLACEMENT("Partial Knee Replacement (PKR)", "Knee (Member 2)", RehabComponent.KNEE),
    ACL_RECONSTRUCTION("ACL Reconstruction", "Knee (Member 2)", RehabComponent.KNEE),
    MENISCUS_REPAIR("Meniscus Repair / Meniscectomy", "Knee (Member 2)", RehabComponent.KNEE),
    KNEE_ARTHROSCOPY("Knee Arthroscopy", "Knee (Member 2)", RehabComponent.KNEE),
    PATELLAR_REALIGNMENT("Patellar Realignment (MPFL)", "Knee (Member 2)", RehabComponent.KNEE),
    KNEE_REPLACEMENT("Knee Replacement (General)", "Knee (Member 2)", RehabComponent.KNEE),

    // Member 3: Wrist Component Surgeries
    DISTAL_RADIUS_ORIF("Distal Radius ORIF", "Wrist (Member 3)", RehabComponent.WRIST),
    CARPAL_TUNNEL_RELEASE("Carpal Tunnel Release (CTR)", "Wrist (Member 3)", RehabComponent.WRIST),
    SCAPHOID_FIXATION("Scaphoid Fixation / ORIF", "Wrist (Member 3)", RehabComponent.WRIST),
    TFCC_REPAIR("TFCC Repair", "Wrist (Member 3)", RehabComponent.WRIST),
    WRIST_ARTHROSCOPY("Wrist Arthroscopy", "Wrist (Member 3)", RehabComponent.WRIST),

    // Member 4: Elbow Component Surgeries
    OLECRANON_ORIF("Olecranon ORIF", "Elbow (Member 4)", RehabComponent.ELBOW),
    BICEPS_TENDON_REPAIR("Distal Biceps Tendon Repair", "Elbow (Member 4)", RehabComponent.ELBOW),
    TENNIS_ELBOW_RELEASE("Epicondyle Release (Tennis/Golfer)", "Elbow (Member 4)", RehabComponent.ELBOW),
    ELBOW_ARTHROSCOPY("Elbow Arthroscopy", "Elbow (Member 4)", RehabComponent.ELBOW),
    TOTAL_ELBOW_ARTHROPLASTY("Total Elbow Arthroplasty (TEA)", "Elbow (Member 4)", RehabComponent.ELBOW),

    // General / Conservative Rehabilitation
    CONSERVATIVE_REHAB("Conservative (Non-Surgical)", "General / Other", null),
    HIP_REPLACEMENT("Hip Replacement (THR)", "General / Other", null),
    OTHER("Other Orthopedic Surgery", "General / Other", null)
}

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
