package com.rehabresearch.datacollector.ui.history

import android.content.Context
import android.content.Intent
import android.widget.Toast
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.rehabresearch.datacollector.data.local.entity.PatientEntity
import com.rehabresearch.datacollector.data.local.entity.RehabComponent
import com.rehabresearch.datacollector.data.local.entity.SessionEntity
import com.rehabresearch.datacollector.data.repository.PatientRepository
import com.rehabresearch.datacollector.data.repository.SessionRepository
import com.rehabresearch.datacollector.utils.CsvExporter
import dagger.hilt.android.lifecycle.HiltViewModel
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch
import javax.inject.Inject

@HiltViewModel
class HistoryViewModel @Inject constructor(
    @ApplicationContext private val context: Context,
    private val sessionRepository: SessionRepository,
    private val patientRepository: PatientRepository
) : ViewModel() {

    val patients: StateFlow<List<PatientEntity>> = patientRepository.observeAll()
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5000), emptyList())

    private val _allSessions: StateFlow<List<SessionEntity>> = sessionRepository.observeAll()
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5000), emptyList())

    val selectedParticipantId = MutableStateFlow<String?>(null) // null = All Participants
    val selectedComponent = MutableStateFlow<RehabComponent?>(null) // null = All Components (Ankle, Knee, Wrist, Elbow)

    val sessions: StateFlow<List<SessionEntity>> = combine(
        _allSessions,
        selectedParticipantId,
        selectedComponent
    ) { all, filterPid, filterComp ->
        all.filter { session ->
            (filterPid == null || session.patientId == filterPid) &&
            (filterComp == null || session.component == filterComp)
        }
    }.stateIn(viewModelScope, SharingStarted.WhileSubscribed(5000), emptyList())

    fun selectParticipantFilter(patientId: String?) {
        selectedParticipantId.value = patientId
    }

    fun selectComponentFilter(component: RehabComponent?) {
        selectedComponent.value = component
    }

    /** Exports and shares an individual session CSV directly from the history list. */
    fun exportAndShareSession(context: Context, session: SessionEntity) {
        viewModelScope.launch {
            val patient = patientRepository.getById(session.patientId) ?: return@launch
            val readings = sessionRepository.getSensorReadings(session.sessionId)
            if (readings.isEmpty()) {
                Toast.makeText(context, "No sensor samples in this session.", Toast.LENGTH_SHORT).show()
                return@launch
            }
            val csvFile = CsvExporter.exportSessionToCsv(context, session, patient, readings)
            CsvExporter.exportSessionMetadata(context, session, patient)
            val shareIntent = CsvExporter.createShareIntent(context, csvFile)
            context.startActivity(Intent.createChooser(shareIntent, "Share Session CSV"))
        }
    }

    /**
     * Consolidates and exports all sessions matching the active filter.
     * Allows Member 1 to export solely their Ankle (Tibia + Foot) data,
     * Member 2 Knee data, Member 3 Wrist data, or Member 4 Elbow data.
     */
    fun exportAndShareActiveSelection(context: Context) {
        viewModelScope.launch {
            val targetSessions = sessions.value
            if (targetSessions.isEmpty()) {
                Toast.makeText(context, "No recordings to export.", Toast.LENGTH_SHORT).show()
                return@launch
            }

            val allPatients = patientRepository.observeAll()
            val patientMap = patients.value.associateBy { it.patientId }

            val sessionsWithReadings = targetSessions.map { s ->
                Pair(s, sessionRepository.getSensorReadings(s.sessionId))
            }.filter { it.second.isNotEmpty() }

            if (sessionsWithReadings.isEmpty()) {
                Toast.makeText(context, "No recorded samples found in selection.", Toast.LENGTH_SHORT).show()
                return@launch
            }

            val compName = selectedComponent.value?.name ?: "ALL_COMPONENTS"
            val pidName = selectedParticipantId.value ?: "ALL_MEMBERS"
            val exportName = "AKIRA_${compName}_${pidName}_dataset"

            val aggregatedCsv = CsvExporter.exportFilteredDatasetToCsv(
                context,
                exportName,
                sessionsWithReadings,
                patientMap
            )
            val shareIntent = CsvExporter.createShareIntent(context, aggregatedCsv)
            context.startActivity(Intent.createChooser(shareIntent, "Export $exportName"))
        }
    }
}
