package com.rehabresearch.datacollector.utils

import android.content.Context
import com.rehabresearch.datacollector.data.local.entity.PatientEntity
import com.rehabresearch.datacollector.data.local.entity.SensorReadingEntity
import com.rehabresearch.datacollector.data.local.entity.SessionEntity
import java.io.File
import java.io.FileWriter

/**
 * Writes one CSV per session in the exact shape the AI training pipeline expects
 * (see the "Dataset Output Example" table in the project spec):
 * Timestamp, Ax, Ay, Az, Gx, Gy, Gz, Exercise, Side, Patient, Week, Correct
 *
 * Files land in app-specific external storage (no permission needed on API 29+):
 *   /Android/data/com.rehabresearch.datacollector/files/exports/<sessionId>.csv
 * This app is local-only — pull files off the device manually (USB / file
 * manager / adb pull) to feed the AI training pipeline.
 */
object CsvExporter {

    fun exportSessionToCsv(
        context: Context,
        session: SessionEntity,
        patient: PatientEntity,
        readings: List<SensorReadingEntity>
    ): File {
        val exportDir = File(context.getExternalFilesDir(null), "exports").apply { mkdirs() }
        val outFile = File(exportDir, "${session.sessionId}.csv")

        FileWriter(outFile).use { writer ->
            writer.append("Timestamp,Ax,Ay,Az,Gx,Gy,Gz,Exercise,Side,Patient,Week,Correct\n")
            for (r in readings) {
                writer.append(
                    listOf(
                        r.timestampMillis,
                        r.ax, r.ay, r.az,
                        r.gx, r.gy, r.gz,
                        session.exercise.name,
                        session.side.name,
                        patient.patientId,
                        session.recoveryWeek,
                        session.correctMovement?.let { if (it) "Yes" else "No" } ?: ""
                    ).joinToString(",")
                )
                writer.append("\n")
            }
        }
        return outFile
    }

    /** Session-level metadata sidecar (JSON-ish key/value CSV) — pairs with the raw sample CSV. */
    fun exportSessionMetadata(context: Context, session: SessionEntity, patient: PatientEntity): File {
        val exportDir = File(context.getExternalFilesDir(null), "exports").apply { mkdirs() }
        val outFile = File(exportDir, "${session.sessionId}_metadata.csv")
        FileWriter(outFile).use { writer ->
            writer.append("field,value\n")
            val rows = listOf(
                "sessionId" to session.sessionId,
                "patientId" to patient.patientId,
                "exercise" to session.exercise.name,
                "side" to session.side.name,
                "difficulty" to session.difficulty.name,
                "recoveryWeek" to session.recoveryWeek.toString(),
                "targetReps" to session.targetReps.toString(),
                "actualReps" to session.actualReps.toString(),
                "therapist" to session.therapistName,
                "durationMillis" to session.durationMillis.toString(),
                "avgSampleFrequencyHz" to session.avgSampleFrequencyHz.toString(),
                "packetCount" to session.packetCount.toString(),
                "droppedPacketCount" to session.droppedPacketCount.toString(),
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
}
