package com.rehabresearch.datacollector.ui.livesensor

import android.graphics.Color as AndroidColor
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.ElevatedCard
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import com.github.mikephil.charting.charts.LineChart
import com.github.mikephil.charting.data.Entry
import com.github.mikephil.charting.data.LineData
import com.github.mikephil.charting.data.LineDataSet
import com.rehabresearch.datacollector.ui.recording.RecordingViewModel
import com.rehabresearch.datacollector.ui.theme.SensorReadoutStyle

enum class SensorChartView { SENSOR_1, SENSOR_2 }

/**
 * Live sensor display showing real-time 12-channel motion data:
 * - Sensor 1 (CH0): Proximal Segment (Tibia / Thigh / Wrist / Upper Arm)
 * - Sensor 2 (CH1): Distal Segment (Foot / Shank / Hand / Forearm)
 * With real-time waveform plotting, packet health, and trial rep counter.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun LiveSensorScreen(
    viewModel: RecordingViewModel,
    onSessionFinished: (String) -> Unit
) {
    val ui by viewModel.uiState.collectAsState()
    val setup by viewModel.setup.collectAsState()
    var selectedSensorView by remember { mutableStateOf(SensorChartView.SENSOR_1) }

    val s1ShortName = setup.sensor1Placement.substringBefore(" ")
    val s2ShortName = setup.sensor2Placement.substringBefore(" ")

    LaunchedEffect(Unit) {
        if (!ui.isRecording && !ui.isCountingDown && ui.sessionId == null) {
            viewModel.startCountdownThenRecord()
        }
    }
    LaunchedEffect(ui.finished, ui.sessionId) {
        if (ui.finished) ui.sessionId?.let(onSessionFinished)
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Text("${setup.exercise.name.replace("_", " ")} (${setup.trialId})")
                }
            )
        }
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .padding(16.dp)
                .verticalScroll(rememberScrollState())
        ) {

            if (ui.isCountingDown) {
                Column(
                    modifier = Modifier.fillParentMaxHeightOrFull(),
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.Center
                ) {
                    Text(
                        if (ui.countdownValue > 0) ui.countdownValue.toString() else "GO",
                        style = MaterialTheme.typography.headlineLarge.copy(
                            fontSize = 72.sp,
                            fontWeight = FontWeight.Bold
                        )
                    )
                }
                return@Scaffold
            }

            // Link & Stream Health
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                Text("%.0f Hz".format(ui.sampleRateHz), style = MaterialTheme.typography.bodyMedium)
                Text("Packets: ${ui.packetsReceived}", style = MaterialTheme.typography.bodyMedium)
                Text(
                    "Dropped: ${ui.packetsDropped}",
                    style = MaterialTheme.typography.bodyMedium,
                    color = if (ui.packetsDropped > 0) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.onSurfaceVariant
                )
                Text("%.1fs".format(ui.elapsedMillis / 1000f), style = MaterialTheme.typography.bodyMedium)
            }

            Spacer(Modifier.height(10.dp))

            // 1. Sensor 1 Card (CH0) - Proximal Segment
            ElevatedCard(Modifier.fillMaxWidth()) {
                Column(Modifier.padding(12.dp)) {
                    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                        Text(
                            "Sensor 1 (CH0): ${setup.sensor1Placement}",
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.SemiBold
                        )
                        Text("g & °/s", style = MaterialTheme.typography.labelSmall)
                    }
                    Spacer(Modifier.height(6.dp))
                    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                        AxisReadout("S1_Ax", ui.latestPacket?.s1Ax)
                        AxisReadout("S1_Ay", ui.latestPacket?.s1Ay)
                        AxisReadout("S1_Az", ui.latestPacket?.s1Az)
                        AxisReadout("S1_Gx", ui.latestPacket?.s1Gx)
                        AxisReadout("S1_Gy", ui.latestPacket?.s1Gy)
                        AxisReadout("S1_Gz", ui.latestPacket?.s1Gz)
                    }
                }
            }

            Spacer(Modifier.height(8.dp))

            // 2. Sensor 2 Card (CH1) - Distal Segment
            ElevatedCard(Modifier.fillMaxWidth()) {
                Column(Modifier.padding(12.dp)) {
                    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                        Text(
                            "Sensor 2 (CH1): ${setup.sensor2Placement}",
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.SemiBold
                        )
                        Text("g & °/s", style = MaterialTheme.typography.labelSmall)
                    }
                    Spacer(Modifier.height(6.dp))
                    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                        AxisReadout("S2_Ax", ui.latestPacket?.s2Ax)
                        AxisReadout("S2_Ay", ui.latestPacket?.s2Ay)
                        AxisReadout("S2_Az", ui.latestPacket?.s2Az)
                        AxisReadout("S2_Gx", ui.latestPacket?.s2Gx)
                        AxisReadout("S2_Gy", ui.latestPacket?.s2Gy)
                        AxisReadout("S2_Gz", ui.latestPacket?.s2Gz)
                    }
                }
            }

            Spacer(Modifier.height(12.dp))

            // Chart View Selector (Sensor 1 vs Sensor 2)
            Row(
                Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text("Waveform:", style = MaterialTheme.typography.bodyMedium, fontWeight = FontWeight.Medium)
                FilterChip(
                    selected = selectedSensorView == SensorChartView.SENSOR_1,
                    onClick = { selectedSensorView = SensorChartView.SENSOR_1 },
                    label = { Text("$s1ShortName IMU") }
                )
                FilterChip(
                    selected = selectedSensorView == SensorChartView.SENSOR_2,
                    onClick = { selectedSensorView = SensorChartView.SENSOR_2 },
                    label = { Text("$s2ShortName IMU") }
                )
            }

            Spacer(Modifier.height(8.dp))

            // Acceleration waveform
            val currentLabel = if (selectedSensorView == SensorChartView.SENSOR_1) s1ShortName else s2ShortName
            Text(
                "$currentLabel Acceleration (g)",
                style = MaterialTheme.typography.titleMedium
            )
            val accelValues = if (selectedSensorView == SensorChartView.SENSOR_1) {
                ui.chartBuffer.map { Triple(it.s1Ax, it.s1Ay, it.s1Az) }
            } else {
                ui.chartBuffer.map { Triple(it.s2Ax, it.s2Ay, it.s2Az) }
            }
            ImuLineChart(
                modifier = Modifier.fillMaxWidth().height(150.dp),
                seriesLabels = Triple("X", "Y", "Z"),
                colors = Triple(AndroidColor.rgb(228, 87, 46), AndroidColor.rgb(23, 190, 187), AndroidColor.rgb(46, 64, 87)),
                values = accelValues
            )

            Spacer(Modifier.height(8.dp))

            // Gyroscope waveform
            Text(
                "$currentLabel Gyroscope (°/s)",
                style = MaterialTheme.typography.titleMedium
            )
            val gyroValues = if (selectedSensorView == SensorChartView.SENSOR_1) {
                ui.chartBuffer.map { Triple(it.s1Gx, it.s1Gy, it.s1Gz) }
            } else {
                ui.chartBuffer.map { Triple(it.s2Gx, it.s2Gy, it.s2Gz) }
            }
            ImuLineChart(
                modifier = Modifier.fillMaxWidth().height(150.dp),
                seriesLabels = Triple("Gx", "Gy", "Gz"),
                colors = Triple(AndroidColor.rgb(228, 87, 46), AndroidColor.rgb(23, 190, 187), AndroidColor.rgb(46, 64, 87)),
                values = gyroValues
            )

            Spacer(Modifier.height(14.dp))
            Text("Reps: ${ui.currentReps} / ${setup.targetReps}", style = MaterialTheme.typography.titleLarge)
            Spacer(Modifier.height(8.dp))
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                Button(onClick = { viewModel.incrementRep() }, modifier = Modifier.weight(1f)) { Text("+1 Rep") }
                Button(onClick = { viewModel.stopRecording() }, modifier = Modifier.weight(1f)) { Text("End Session") }
            }
        }
    }
}

private fun Modifier.fillParentMaxHeightOrFull(): Modifier = this.fillMaxSize()

@Composable
private fun AxisReadout(label: String, value: Float?) {
    Column {
        Text(label, style = MaterialTheme.typography.labelSmall)
        Text(value?.let { "%.2f".format(it) } ?: "--", style = SensorReadoutStyle)
    }
}

@Composable
private fun ImuLineChart(
    modifier: Modifier,
    seriesLabels: Triple<String, String, String>,
    colors: Triple<Int, Int, Int>,
    values: List<Triple<Float, Float, Float>>
) {
    AndroidView(
        modifier = modifier,
        factory = { ctx ->
            LineChart(ctx).apply {
                description.isEnabled = false
                legend.isEnabled = true
                axisRight.isEnabled = false
                setTouchEnabled(false)
                setNoDataText("Waiting for sensor stream...")
            }
        },
        update = { lineChart ->
            val xEntries = values.mapIndexed { i, t -> Entry(i.toFloat(), t.first) }
            val yEntries = values.mapIndexed { i, t -> Entry(i.toFloat(), t.second) }
            val zEntries = values.mapIndexed { i, t -> Entry(i.toFloat(), t.third) }

            fun makeSet(entries: List<Entry>, label: String, color: Int) = LineDataSet(entries, label).apply {
                this.color = color
                lineWidth = 1.8f
                setDrawCircles(false)
                setDrawValues(false)
                mode = LineDataSet.Mode.CUBIC_BEZIER
            }

            lineChart.data = LineData(
                makeSet(xEntries, seriesLabels.first, colors.first),
                makeSet(yEntries, seriesLabels.second, colors.second),
                makeSet(zEntries, seriesLabels.third, colors.third)
            )
            lineChart.invalidate()
        }
    )
}
