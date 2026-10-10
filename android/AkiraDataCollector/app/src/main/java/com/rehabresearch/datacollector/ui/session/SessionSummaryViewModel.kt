package com.rehabresearch.datacollector.ui.session

import android.content.Context
import androidx.lifecycle.SavedStateHandle
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.rehabresearch.datacollector.data.local.entity.PatientEntity
import com.rehabresearch.datacollector.data.local.entity.SessionEntity
import com.rehabresearch.datacollector.data.local.entity.SessionStatus
import com.rehabresearch.datacollector.data.repository.PatientRepository
import com.rehabresearch.datacollector.data.repository.SessionRepository
import com.rehabresearch.datacollector.ui.NavRoutes
import com.rehabresearch.datacollector.utils.CsvExporter
import dagger.hilt.android.lifecycle.HiltViewModel
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import javax.inject.Inject

data class SessionSummaryUiState(
    val session: SessionEntity? = null,
    val patient: PatientEntity? = null,
    val sampleCount: Long = 0,
    val painLevel: Int = 0,
    val correctMovement: Boolean = true,
    val compensationObserved: Boolean = false,
    val assistiveDevice: String = "",
    val notes: String = "",
    val exportedFilePath: String? = null,
    val isExporting: Boolean = false
)

@HiltViewModel
class SessionSummaryViewModel @Inject constructor(
    @ApplicationContext private val context: Context,
    private val sessionRepository: SessionRepository,
    private val patientRepository: PatientRepository,
    savedStateHandle: SavedStateHandle
) : ViewModel() {

    private val sessionId: String = checkNotNull(savedStateHandle[NavRoutes.ARG_SESSION_ID])

    private val _uiState = MutableStateFlow(SessionSummaryUiState())
    val uiState: StateFlow<SessionSummaryUiState> = _uiState

    init {
        viewModelScope.launch {
            val session = sessionRepository.getById(sessionId) ?: return@launch
            val patient = patientRepository.getById(session.patientId)
            val count = sessionRepository.getSampleCount(sessionId)
            _uiState.value = _uiState.value.copy(session = session, patient = patient, sampleCount = count)
        }
    }

    fun updatePainLevel(v: Int) { _uiState.value = _uiState.value.copy(painLevel = v) }
    fun updateCorrectMovement(v: Boolean) { _uiState.value = _uiState.value.copy(correctMovement = v) }
    fun updateCompensation(v: Boolean) { _uiState.value = _uiState.value.copy(compensationObserved = v) }
    fun updateAssistiveDevice(v: String) { _uiState.value = _uiState.value.copy(assistiveDevice = v) }
    fun updateNotes(v: String) { _uiState.value = _uiState.value.copy(notes = v) }

    /** Saves labels to Room, then writes the CSV dataset row + metadata sidecar to disk. */
    fun saveLabelsAndExport() {
        val state = _uiState.value
        val session = state.session ?: return
        val patient = state.patient ?: return

        viewModelScope.launch {
            _uiState.value = state.copy(isExporting = true)
            val updatedSession = session.copy(
                painLevel = state.painLevel,
                correctMovement = state.correctMovement,
                compensationObserved = state.compensationObserved,
                assistiveDevice = state.assistiveDevice.ifBlank { null },
                therapistNotes = state.notes
            )
            sessionRepository.updateSession(updatedSession)

            val readings = sessionRepository.getSensorReadings(session.sessionId)
            val csvFile = CsvExporter.exportSessionToCsv(context, updatedSession, patient, readings)
            CsvExporter.exportSessionMetadata(context, updatedSession, patient)

            sessionRepository.updateSession(
                updatedSession.copy(csvFilePath = csvFile.absolutePath, status = SessionStatus.EXPORTED)
            )
            _uiState.value = _uiState.value.copy(isExporting = false, exportedFilePath = csvFile.absolutePath)
        }
    }
}
