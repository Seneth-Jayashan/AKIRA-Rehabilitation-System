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
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
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

/**
 * The most important screen in the app (per spec). Shows live accel/gyro
 * values, a rolling real-time chart, rep counter, and packet/frequency
 * health stats so the therapist can catch a bad BLE link mid-session
 * instead of finding out after the data is already useless.
 */
@Composable
fun LiveSensorScreen(
    viewModel: RecordingViewModel,
    onSessionFinished: (String) -> Unit
) {
    val ui by viewModel.uiState.collectAsState()
    val setup by viewModel.setup.collectAsState()

    LaunchedEffect(Unit) {
        if (!ui.isRecording && !ui.isCountingDown && ui.sessionId == null) {
            viewModel.startCountdownThenRecord()
        }
    }
    LaunchedEffect(ui.finished, ui.sessionId) {
        if (ui.finished) ui.sessionId?.let(onSessionFinished)
    }

    Scaffold(topBar = { TopAppBar(title = { Text(setup.exercise.name.replace("_", " ")) }) }) { padding ->
        Column(modifier = Modifier
            .fillMaxSize()
            .padding(padding)
            .padding(16.dp)) {

            if (ui.isCountingDown) {
                Column(
                    modifier = Modifier.fillMaxSize(),
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

            // Link health row
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                Text("%.0f Hz".format(ui.sampleRateHz), style = MaterialTheme.typography.bodyMedium)
                Text("Packets: ${ui.packetsReceived}", style = MaterialTheme.typography.bodyMedium)
                Text("Dropped: ${ui.packetsDropped}", style = MaterialTheme.typography.bodyMedium,
                    color = if (ui.packetsDropped > 0) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.onSurfaceVariant)
                Text("%.1fs".format(ui.elapsedMillis / 1000f), style = MaterialTheme.typography.bodyMedium)
            }

            Spacer(Modifier.height(12.dp))

            // Live numeric readouts
            Card(Modifier.fillMaxWidth()) {
                Column(Modifier.padding(16.dp)) {
                    Text("Acceleration (g)", style = MaterialTheme.typography.titleLarge)
                    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                        AxisReadout("Ax", ui.latestPacket?.ax)
                        AxisReadout("Ay", ui.latestPacket?.ay)
                        AxisReadout("Az", ui.latestPacket?.az)
                    }
                    Spacer(Modifier.height(8.dp))
                    Text("Gyroscope (°/s)", style = MaterialTheme.typography.titleLarge)
                    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                        AxisReadout("Gx", ui.latestPacket?.gx)
                        AxisReadout("Gy", ui.latestPacket?.gy)
                        AxisReadout("Gz", ui.latestPacket?.gz)
                    }
                }
            }

            Spacer(Modifier.height(12.dp))
            Text("Acceleration", style = MaterialTheme.typography.titleLarge)
            AccelChart(ui.chartBuffer, modifier = Modifier.fillMaxWidth().height(160.dp))

            Spacer(Modifier.height(8.dp))
            Text("Gyroscope", style = MaterialTheme.typography.titleLarge)
            GyroChart(ui.chartBuffer, modifier = Modifier.fillMaxWidth().height(160.dp))

            Spacer(Modifier.height(16.dp))
            Text("Reps: ${ui.currentReps} / ${setup.targetReps}", style = MaterialTheme.typography.titleLarge)
            Spacer(Modifier.height(8.dp))
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                Button(onClick = { viewModel.incrementRep() }, modifier = Modifier.weight(1f)) { Text("+1 Rep") }
                Button(onClick = { viewModel.stopRecording() }, modifier = Modifier.weight(1f)) { Text("End Session") }
            }
        }
    }
}

@Composable
private fun AxisReadout(label: String, value: Float?) {
    Column {
        Text(label, style = MaterialTheme.typography.labelSmall)
        Text(value?.let { "%.2f".format(it) } ?: "--", style = SensorReadoutStyle)
    }
}

@Composable
private fun AccelChart(points: List<com.rehabresearch.datacollector.ui.recording.LiveChartPoint>, modifier: Modifier) {
    ImuLineChart(
        modifier = modifier,
        seriesLabels = Triple("Ax", "Ay", "Az"),
        colors = Triple(AndroidColor.rgb(228, 87, 46), AndroidColor.rgb(23, 190, 187), AndroidColor.rgb(46, 64, 87)),
        values = points.map { Triple(it.ax, it.ay, it.az) }
    )
}

@Composable
private fun GyroChart(points: List<com.rehabresearch.datacollector.ui.recording.LiveChartPoint>, modifier: Modifier) {
    ImuLineChart(
        modifier = modifier,
        seriesLabels = Triple("Gx", "Gy", "Gz"),
        colors = Triple(AndroidColor.rgb(228, 87, 46), AndroidColor.rgb(23, 190, 187), AndroidColor.rgb(46, 64, 87)),
        values = points.map { Triple(it.gx, it.gy, it.gz) }
    )
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
                setNoDataText("Waiting for sensor data...")
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
