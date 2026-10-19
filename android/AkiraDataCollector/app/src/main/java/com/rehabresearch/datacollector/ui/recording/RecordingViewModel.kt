package com.rehabresearch.datacollector.ui.recording

import androidx.lifecycle.SavedStateHandle
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.rehabresearch.datacollector.ble.BleManager
import com.rehabresearch.datacollector.ble.ImuPacket
import com.rehabresearch.datacollector.data.local.entity.BodySide
import com.rehabresearch.datacollector.data.local.entity.Difficulty
import com.rehabresearch.datacollector.data.local.entity.ExerciseType
import com.rehabresearch.datacollector.data.local.entity.RehabComponent
import com.rehabresearch.datacollector.data.local.entity.SensorReadingEntity
import com.rehabresearch.datacollector.data.local.entity.SessionEntity
import com.rehabresearch.datacollector.data.local.entity.SessionStatus
import com.rehabresearch.datacollector.data.local.entity.component
import com.rehabresearch.datacollector.data.repository.PatientRepository
import com.rehabresearch.datacollector.data.repository.SessionRepository
import com.rehabresearch.datacollector.ui.NavRoutes
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import javax.inject.Inject

data class RecordingSetupState(
    val patientId: String = "",
    val trialNumber: Int = 1,
    val trialId: String = "Trial_1",
    val component: RehabComponent = RehabComponent.ANKLE,
    val exercise: ExerciseType = ExerciseType.DORSIFLEXION,
    val sensor1Placement: String = RehabComponent.ANKLE.sensor1Default,
    val sensor2Placement: String = RehabComponent.ANKLE.sensor2Default,
    val side: BodySide = BodySide.RIGHT,
    val difficulty: Difficulty = Difficulty.EASY,
    val targetReps: Int = 10,
    val therapistName: String = "",
    val recoveryWeek: Int = 1
)

data class LiveChartPoint(
    val t: Long,
    val s1Ax: Float, val s1Ay: Float, val s1Az: Float,
    val s1Gx: Float, val s1Gy: Float, val s1Gz: Float,
    val s2Ax: Float, val s2Ay: Float, val s2Az: Float,
    val s2Gx: Float, val s2Gy: Float, val s2Gz: Float,
    // Aliases for Tibia/Foot
    val tAx: Float = s1Ax, val tAy: Float = s1Ay, val tAz: Float = s1Az,
    val tGx: Float = s1Gx, val tGy: Float = s1Gy, val tGz: Float = s1Gz,
    val fAx: Float = s2Ax, val fAy: Float = s2Ay, val fAz: Float = s2Az,
    val fGx: Float = s2Gx, val fGy: Float = s2Gy, val fGz: Float = s2Gz,
    // Single-sensor aliases
    val ax: Float = s1Ax, val ay: Float = s1Ay, val az: Float = s1Az,
    val gx: Float = s1Gx, val gy: Float = s1Gy, val gz: Float = s1Gz
)

data class RecordingUiState(
    val isRecording: Boolean = false,
    val isCountingDown: Boolean = false,
    val countdownValue: Int = 3,
    val latestPacket: ImuPacket? = null,
    val chartBuffer: List<LiveChartPoint> = emptyList(),
    val elapsedMillis: Long = 0,
    val currentReps: Int = 0,
    val sampleRateHz: Float = 0f,
    val packetsReceived: Long = 0,
    val packetsDropped: Long = 0,
    val sessionId: String? = null,
    val finished: Boolean = false
)

private const val CHART_BUFFER_SIZE = 150 // ~1.5s of chart history at 100Hz

@HiltViewModel
class RecordingViewModel @Inject constructor(
    private val bleManager: BleManager,
    private val sessionRepository: SessionRepository,
    private val patientRepository: PatientRepository,
    savedStateHandle: SavedStateHandle
) : ViewModel() {

    val patientId: String = checkNotNull(savedStateHandle[NavRoutes.ARG_PATIENT_ID])

    private val _setup = MutableStateFlow(RecordingSetupState(patientId = patientId))
    val setup: StateFlow<RecordingSetupState> = _setup

    private val _uiState = MutableStateFlow(RecordingUiState())
    val uiState: StateFlow<RecordingUiState> = _uiState

    private var streamJob: Job? = null
    private var timerJob: Job? = null
    private var sessionStartMillis: Long = 0
    private val pendingBatch = mutableListOf<SensorReadingEntity>()
    private val batchFlushSize = 50

    fun updateSetup(transform: (RecordingSetupState) -> RecordingSetupState) {
        _setup.value = transform(_setup.value)
    }

    fun setComponent(c: RehabComponent) {
        val defaultEx = ExerciseType.entries.firstOrNull { it.component == c } ?: ExerciseType.DORSIFLEXION
        _setup.value = _setup.value.copy(
            component = c,
            exercise = defaultEx,
            sensor1Placement = c.sensor1Default,
            sensor2Placement = c.sensor2Default
        )
    }

    fun setTrialNumber(number: Int) {
        _setup.value = _setup.value.copy(
            trialNumber = number,
            trialId = "Trial_$number"
        )
    }

    /** 3-2-1-GO countdown, then begins BLE recording. */
    fun startCountdownThenRecord() {
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(isCountingDown = true, countdownValue = 3)
            for (n in 3 downTo 1) {
                _uiState.value = _uiState.value.copy(countdownValue = n)
                kotlinx.coroutines.delay(1000)
            }
            _uiState.value = _uiState.value.copy(isCountingDown = false)
            beginRecording()
        }
    }

    private fun beginRecording() {
        val s = _setup.value
        val sessionId = sessionRepository.newSessionId()
        sessionStartMillis = System.currentTimeMillis()

        viewModelScope.launch {
            sessionRepository.startSession(
                SessionEntity(
                    sessionId = sessionId,
                    patientId = s.patientId,
                    trialId = s.trialId,
                    component = s.component,
                    exercise = s.exercise,
                    side = s.side,
                    difficulty = s.difficulty,
                    recoveryWeek = s.recoveryWeek,
                    targetReps = s.targetReps,
                    therapistName = s.therapistName.ifBlank { "Participant" },
                    sensor1Placement = s.sensor1Placement,
                    sensor2Placement = s.sensor2Placement,
                    startedAtEpochMillis = sessionStartMillis,
                    status = SessionStatus.RECORDING
                )
            )

            _uiState.value = _uiState.value.copy(isRecording = true, sessionId = sessionId, elapsedMillis = 0)

            streamJob = launch {
                bleManager.imuPackets.collect { packet -> onPacket(sessionId, packet) }
            }
            timerJob = launch {
                while (true) {
                    kotlinx.coroutines.delay(200)
                    _uiState.value = _uiState.value.copy(elapsedMillis = System.currentTimeMillis() - sessionStartMillis)
                }
            }
        }
    }

    private suspend fun onPacket(sessionId: String, packet: ImuPacket) {
        val stats = bleManager.linkStats.value
        val point = LiveChartPoint(
            t = packet.timestampMillis,
            s1Ax = packet.s1Ax, s1Ay = packet.s1Ay, s1Az = packet.s1Az,
            s1Gx = packet.s1Gx, s1Gy = packet.s1Gy, s1Gz = packet.s1Gz,
            s2Ax = packet.s2Ax, s2Ay = packet.s2Ay, s2Az = packet.s2Az,
            s2Gx = packet.s2Gx, s2Gy = packet.s2Gy, s2Gz = packet.s2Gz
        )
        val newBuffer = (_uiState.value.chartBuffer + point).takeLast(CHART_BUFFER_SIZE)

        _uiState.value = _uiState.value.copy(
            latestPacket = packet,
            chartBuffer = newBuffer,
            sampleRateHz = stats.currentSampleRateHz,
            packetsReceived = stats.packetsReceived,
            packetsDropped = stats.packetsDropped
        )

        pendingBatch.add(
            SensorReadingEntity(
                sessionId = sessionId,
                timestampMillis = packet.timestampMillis,
                s1Ax = packet.s1Ax, s1Ay = packet.s1Ay, s1Az = packet.s1Az,
                s1Gx = packet.s1Gx, s1Gy = packet.s1Gy, s1Gz = packet.s1Gz,
                s2Ax = packet.s2Ax, s2Ay = packet.s2Ay, s2Az = packet.s2Az,
                s2Gx = packet.s2Gx, s2Gy = packet.s2Gy, s2Gz = packet.s2Gz,
                quatW = packet.quatW, quatX = packet.quatX, quatY = packet.quatY, quatZ = packet.quatZ,
                temperatureC = packet.temperatureC
            )
        )
        if (pendingBatch.size >= batchFlushSize) {
            flushBatch()
        }
    }

    private suspend fun flushBatch() {
        if (pendingBatch.isEmpty()) return
        val toWrite = pendingBatch.toList()
        pendingBatch.clear()
        sessionRepository.appendSensorBatch(toWrite)
    }

    fun incrementRep() {
        _uiState.value = _uiState.value.copy(currentReps = _uiState.value.currentReps + 1)
    }

    fun stopRecording() {
        streamJob?.cancel()
        timerJob?.cancel()
        val sessionId = _uiState.value.sessionId ?: return
        val endMillis = System.currentTimeMillis()

        viewModelScope.launch {
            flushBatch()
            val existing = sessionRepository.getById(sessionId) ?: return@launch
            sessionRepository.updateSession(
                existing.copy(
                    endedAtEpochMillis = endMillis,
                    durationMillis = endMillis - sessionStartMillis,
                    actualReps = _uiState.value.currentReps,
                    avgSampleFrequencyHz = _uiState.value.sampleRateHz,
                    packetCount = _uiState.value.packetsReceived,
                    droppedPacketCount = _uiState.value.packetsDropped,
                    status = SessionStatus.COMPLETED
                )
            )
            _uiState.value = _uiState.value.copy(isRecording = false, finished = true)
        }
    }

    override fun onCleared() {
        super.onCleared()
        streamJob?.cancel()
        timerJob?.cancel()
    }
}
