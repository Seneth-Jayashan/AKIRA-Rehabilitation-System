package com.rehabresearch.datacollector.utils

import android.content.Context
import android.content.Intent
import androidx.core.content.FileProvider
import com.rehabresearch.datacollector.data.local.entity.PatientEntity
import com.rehabresearch.datacollector.data.local.entity.RehabComponent
import com.rehabresearch.datacollector.data.local.entity.SensorReadingEntity
import com.rehabresearch.datacollector.data.local.entity.SessionEntity
import java.io.File
import java.io.FileWriter

/**
 * Standardized research CSV exporter for the 4 AKIRA components:
 * - Member 1 (Ankle): Sensor 1 = Tibia (CH0),     Sensor 2 = Foot (CH1)
 * - Member 2 (Knee):  Sensor 1 = Thigh (CH0),     Sensor 2 = Shank (CH1)
 * - Member 3 (Wrist): Sensor 1 = Wrist (CH0),     Sensor 2 = Hand (CH1)
 * - Member 4 (Elbow): Sensor 1 = Upper Arm (CH0), Sensor 2 = Forearm (CH1)
 */
object CsvExporter {

    private const val CSV_HEADER =
        "timestamp_ms,s1_ax,s1_ay,s1_az,s1_gx,s1_gy,s1_gz," +
        "s2_ax,s2_ay,s2_az,s2_gx,s2_gy,s2_gz," +
        "tibia_ax,tibia_ay,tibia_az,tibia_gx,tibia_gy,tibia_gz," +
        "foot_ax,foot_ay,foot_az,foot_gx,foot_gy,foot_gz," +
        "participant_id,session_id,trial_id,component,exercise,side,difficulty,recovery_week," +
        "target_reps,actual_reps,duration_ms,sampling_rate_hz,sensor1_placement,sensor2_placement," +
        "units_accel,units_gyro,calibration_status,correct_movement,compensation_observed,pain_level,notes\n"

    /** Exports a single trial/session into a standalone CSV file. */
    fun exportSessionToCsv(
        context: Context,
        session: SessionEntity,
        patient: PatientEntity,
        readings: List<SensorReadingEntity>
    ): File {
        val exportDir = File(context.getExternalFilesDir(null), "exports").apply { mkdirs() }
        val filename = "${patient.patientId}_${session.component.name}_${session.exercise.name}_${session.trialId}_${session.sessionId.take(8)}.csv"
        val outFile = File(exportDir, filename)

        FileWriter(outFile).use { writer ->
            writer.append(CSV_HEADER)
            val notesClean = session.therapistNotes.replace(",", ";").replace("\n", " ").trim()
            val correctStr = session.correctMovement?.let { if (it) "Yes" else "No" } ?: ""

            for (r in readings) {
                writer.append(
                    listOf(
                        r.timestampMillis,
                        "%.4f".format(r.s1Ax), "%.4f".format(r.s1Ay), "%.4f".format(r.s1Az),
                        "%.4f".format(r.s1Gx), "%.4f".format(r.s1Gy), "%.4f".format(r.s1Gz),
                        "%.4f".format(r.s2Ax), "%.4f".format(r.s2Ay), "%.4f".format(r.s2Az),
                        "%.4f".format(r.s2Gx), "%.4f".format(r.s2Gy), "%.4f".format(r.s2Gz),
                        // Tibia / Foot columns (populated directly from S1 / S2)
                        "%.4f".format(r.s1Ax), "%.4f".format(r.s1Ay), "%.4f".format(r.s1Az),
                        "%.4f".format(r.s1Gx), "%.4f".format(r.s1Gy), "%.4f".format(r.s1Gz),
                        "%.4f".format(r.s2Ax), "%.4f".format(r.s2Ay), "%.4f".format(r.s2Az),
                        "%.4f".format(r.s2Gx), "%.4f".format(r.s2Gy), "%.4f".format(r.s2Gz),
                        patient.patientId,
                        session.sessionId,
                        session.trialId,
                        session.component.name,
                        session.exercise.name,
                        session.side.name,
                        session.difficulty.name,
                        session.recoveryWeek,
                        session.targetReps,
                        session.actualReps,
                        session.durationMillis,
                        "%.1f".format(session.avgSampleFrequencyHz),
                        "\"${session.sensor1Placement}\"",
                        "\"${session.sensor2Placement}\"",
                        session.accelUnit,
                        session.gyroUnit,
                        "\"${session.calibrationStatus}\"",
                        correctStr,
                        session.compensationObserved,
                        session.painLevel ?: "",
                        "\"$notesClean\""
                    ).joinToString(",")
                )
                writer.append("\n")
            }
        }
        return outFile
    }

    /**
     * Exports all sessions recorded for a specific participant and/or component.
     * Allows Member 1 to export solely Ankle data, Member 2 Knee data, etc.
     */
    fun exportFilteredDatasetToCsv(
        context: Context,
        exportName: String,
        sessionsWithReadings: List<Pair<SessionEntity, List<SensorReadingEntity>>>,
        patientMap: Map<String, PatientEntity>
    ): File {
        val exportDir = File(context.getExternalFilesDir(null), "exports").apply { mkdirs() }
        val outFile = File(exportDir, "${exportName}.csv")

        FileWriter(outFile).use { writer ->
            writer.append(CSV_HEADER)
            for ((session, readings) in sessionsWithReadings) {
                val patient = patientMap[session.patientId]
                val notesClean = session.therapistNotes.replace(",", ";").replace("\n", " ").trim()
                val correctStr = session.correctMovement?.let { if (it) "Yes" else "No" } ?: ""

                for (r in readings) {
                    writer.append(
                        listOf(
                            r.timestampMillis,
                            "%.4f".format(r.s1Ax), "%.4f".format(r.s1Ay), "%.4f".format(r.s1Az),
                            "%.4f".format(r.s1Gx), "%.4f".format(r.s1Gy), "%.4f".format(r.s1Gz),
                            "%.4f".format(r.s2Ax), "%.4f".format(r.s2Ay), "%.4f".format(r.s2Az),
                            "%.4f".format(r.s2Gx), "%.4f".format(r.s2Gy), "%.4f".format(r.s2Gz),
                            "%.4f".format(r.s1Ax), "%.4f".format(r.s1Ay), "%.4f".format(r.s1Az),
                            "%.4f".format(r.s1Gx), "%.4f".format(r.s1Gy), "%.4f".format(r.s1Gz),
                            "%.4f".format(r.s2Ax), "%.4f".format(r.s2Ay), "%.4f".format(r.s2Az),
                            "%.4f".format(r.s2Gx), "%.4f".format(r.s2Gy), "%.4f".format(r.s2Gz),
                            session.patientId,
                            session.sessionId,
                            session.trialId,
                            session.component.name,
                            session.exercise.name,
                            session.side.name,
                            session.difficulty.name,
                            session.recoveryWeek,
                            session.targetReps,
                            session.actualReps,
                            session.durationMillis,
                            "%.1f".format(session.avgSampleFrequencyHz),
                            "\"${session.sensor1Placement}\"",
                            "\"${session.sensor2Placement}\"",
                            session.accelUnit,
                            session.gyroUnit,
                            "\"${session.calibrationStatus}\"",
                            correctStr,
                            session.compensationObserved,
                            session.painLevel ?: "",
                            "\"$notesClean\""
                        ).joinToString(",")
                    )
                    writer.append("\n")
                }
            }
        }
        return outFile
    }

    /** Exports session metadata sidecar. */
    fun exportSessionMetadata(context: Context, session: SessionEntity, patient: PatientEntity): File {
        val exportDir = File(context.getExternalFilesDir(null), "exports").apply { mkdirs() }
        val outFile = File(exportDir, "${patient.patientId}_${session.component.name}_${session.exercise.name}_${session.trialId}_metadata.csv")
        FileWriter(outFile).use { writer ->
            writer.append("field,value\n")
            val rows = listOf(
                "sessionId" to session.sessionId,
                "patientId" to patient.patientId,
                "trialId" to session.trialId,
                "component" to session.component.name,
                "exercise" to session.exercise.name,
                "side" to session.side.name,
                "difficulty" to session.difficulty.name,
                "recoveryWeek" to session.recoveryWeek.toString(),
                "targetReps" to session.targetReps.toString(),
                "actualReps" to session.actualReps.toString(),
                "therapist" to session.therapistName,
                "startedAtEpochMillis" to session.startedAtEpochMillis.toString(),
                "endedAtEpochMillis" to (session.endedAtEpochMillis?.toString() ?: ""),
                "durationMillis" to session.durationMillis.toString(),
                "avgSampleFrequencyHz" to session.avgSampleFrequencyHz.toString(),
                "packetCount" to session.packetCount.toString(),
                "droppedPacketCount" to session.droppedPacketCount.toString(),
                "sensor1Placement" to session.sensor1Placement,
                "sensor2Placement" to session.sensor2Placement,
                "accelUnit" to session.accelUnit,
                "gyroUnit" to session.gyroUnit,
                "calibrationStatus" to session.calibrationStatus,
                "maxAngleDegrees" to (session.maxAngleDegrees?.toString() ?: ""),
                "painLevel" to (session.painLevel?.toString() ?: ""),
                "assistiveDevice" to (session.assistiveDevice ?: ""),
                "correctMovement" to (session.correctMovement?.toString() ?: ""),
                "compensationObserved" to session.compensationObserved.toString(),
                "notes" to session.therapistNotes.replace(",", ";").replace("\n", " ")
            )
            rows.forEach { (k, v) -> writer.append("$k,$v\n") }
        }
        return outFile
    }

    fun createShareIntent(context: Context, file: File): Intent {
        val uri = FileProvider.getUriForFile(
            context,
            "${context.packageName}.fileprovider",
            file
        )
        return Intent(Intent.ACTION_SEND).apply {
            type = "text/csv"
            putExtra(Intent.EXTRA_STREAM, uri)
            putExtra(Intent.EXTRA_SUBJECT, "AKIRA Research Dataset: ${file.name}")
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
    }
}
