package com.rehabresearch.datacollector.ui.dashboard

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.rehabresearch.datacollector.ble.BleConnectionState
import com.rehabresearch.datacollector.ble.BleManager
import com.rehabresearch.datacollector.data.repository.PatientRepository
import com.rehabresearch.datacollector.data.repository.SessionRepository
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.launch
import javax.inject.Inject

data class DashboardUiState(
    val patientsToday: Int = 0,
    val recordingsToday: Int = 0,
    val totalSessionsCollected: Int = 0,
    val bleState: BleConnectionState = BleConnectionState.Disconnected,
    val batteryPercent: Int? = null
)

@HiltViewModel
class DashboardViewModel @Inject constructor(
    private val patientRepository: PatientRepository,
    private val sessionRepository: SessionRepository,
    private val bleManager: BleManager
) : ViewModel() {

    private val _uiState = MutableStateFlow(DashboardUiState())
    val uiState: StateFlow<DashboardUiState> = _uiState

    init {
        refresh()
        viewModelScope.launch {
            combine(bleManager.connectionState, bleManager.linkStats) { state, stats ->
                _uiState.value = _uiState.value.copy(bleState = state, batteryPercent = stats.batteryPercent)
            }.collect {}
        }
    }

    fun refresh() {
        viewModelScope.launch {
            val patientsToday = patientRepository.countSeenToday()
            val recordingsToday = sessionRepository.countToday()
            _uiState.value = _uiState.value.copy(
                patientsToday = patientsToday,
                recordingsToday = recordingsToday
            )
        }
    }
}
