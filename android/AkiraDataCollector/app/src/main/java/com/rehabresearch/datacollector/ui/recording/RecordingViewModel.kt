package com.rehabresearch.datacollector.ui.recording

import androidx.lifecycle.SavedStateHandle
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.rehabresearch.datacollector.ble.BleManager
import com.rehabresearch.datacollector.ble.ImuPacket
import com.rehabresearch.datacollector.data.local.entity.BodySide
import com.rehabresearch.datacollector.data.local.entity.Difficulty
import com.rehabresearch.datacollector.data.local.entity.ExerciseType
import com.rehabresearch.datacollector.data.local.entity.SensorReadingEntity
import com.rehabresearch.datacollector.data.local.entity.SessionEntity
import com.rehabresearch.datacollector.data.local.entity.SessionStatus
import com.rehabresearch.datacollector.data.repository.PatientRepository
import com.rehabresearch.datacollector.data.repository.SessionRepository
import com.rehabresearch.datacollector.ui.NavRoutes
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import java.util.UUID
import javax.inject.Inject

data class RecordingSetupState(
    val patientId: String = "",
    val exercise: ExerciseType = ExerciseType.SIT_TO_STAND,
    val side: BodySide = BodySide.LEFT,
    val difficulty: Difficulty = Difficulty.EASY,
    val targetReps: Int = 10,
    val therapistName: String = "",
    val recoveryWeek: Int = 1
)

data class LiveChartPoint(val t: Long, val ax: Float, val ay: Float, val az: Float, val gx: Float, val gy: Float, val gz: Float)

data class RecordingUiState(
    val isRecording: Boolean = false,
    val isCountingDown: Boolean = false,
    val countdownValue: Int = 3,
    val latestPacket: ImuPacket? = null,
    val chartBuffer: List<LiveChartPoint> = emptyList(), // last N points for the live chart
    val elapsedMillis: Long = 0,
    val currentReps: Int = 0,
    val sampleRateHz: Float = 0f,
    val packetsReceived: Long = 0,
    val packetsDropped: Long = 0,
    val sessionId: String? = null,
    val finished: Boolean = false
)

private const val CHART_BUFFER_SIZE = 150 // ~1.5s of chart history at 100Hz, enough to look "live" without lagging Compose

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
    private val batchFlushSize = 50 // flush to Room every 50 samples (~0.5s @100Hz)

    fun updateSetup(transform: (RecordingSetupState) -> RecordingSetupState) {
        _setup.value = transform(_setup.value)
    }

    /** 3-2-1-GO countdown, then begins the actual BLE subscription + recording. */
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

        // IMPORTANT: the session row must exist in Room before any sensor_readings
        // batch is inserted, because sensor_readings has a FOREIGN KEY on sessionId.
        // Previously this ran as two independent launch{} blocks — the BLE collector
        // could start flushing samples before the session insert actually committed,
        // which risks a foreign-key constraint failure (or a crash) on the very first
        // batch. Sequencing them in one coroutine guarantees insert-then-stream order.
        viewModelScope.launch {
            sessionRepository.startSession(
                SessionEntity(
                    sessionId = sessionId,
                    patientId = s.patientId,
                    exercise = s.exercise,
                    side = s.side,
                    difficulty = s.difficulty,
                    recoveryWeek = s.recoveryWeek,
                    targetReps = s.targetReps,
                    therapistName = s.therapistName.ifBlank { "Unspecified" },
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
        val point = LiveChartPoint(packet.timestampMillis, packet.ax, packet.ay, packet.az, packet.gx, packet.gy, packet.gz)
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
                ax = packet.ax, ay = packet.ay, az = packet.az,
                gx = packet.gx, gy = packet.gy, gz = packet.gz,
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

    /** Stops the BLE stream and finalizes the session row (duration, packet totals). */
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
