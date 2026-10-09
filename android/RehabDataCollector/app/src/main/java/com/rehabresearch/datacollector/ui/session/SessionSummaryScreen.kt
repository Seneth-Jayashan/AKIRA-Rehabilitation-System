package com.rehabresearch.datacollector.ui.session

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.rememberScrollState
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Slider
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel

@Composable
fun SessionSummaryScreen(
    onDone: () -> Unit,
    viewModel: SessionSummaryViewModel = hiltViewModel()
) {
    val state by viewModel.uiState.collectAsState()
    val session = state.session

    Scaffold(topBar = { TopAppBar(title = { Text("Session Summary") }) }) { padding ->
        Column(
            modifier = Modifier
                .padding(padding)
                .padding(20.dp)
                .verticalScroll(rememberScrollState())
        ) {
            if (session == null) {
                Text("Loading session...")
                return@Column
            }

            Card(Modifier.fillMaxWidth()) {
                Column(Modifier.padding(16.dp)) {
                    Text(session.exercise.name.replace("_", " "), style = MaterialTheme.typography.titleLarge)
                    Text("Duration: %.1fs".format(session.durationMillis / 1000f))
                    Text("Repetitions: ${session.actualReps} / ${session.targetReps}")
                    Text("Samples recorded: ${state.sampleCount}")
                    Text("Avg frequency: %.1f Hz".format(session.avgSampleFrequencyHz))
                    Text("Dropped packets: ${session.droppedPacketCount}")
                }
            }

            Spacer(Modifier.height(20.dp))
            Text("Pain Level: ${state.painLevel} / 10", style = MaterialTheme.typography.titleLarge)
            Slider(
                value = state.painLevel.toFloat(),
                onValueChange = { viewModel.updatePainLevel(it.toInt()) },
                valueRange = 0f..10f,
                steps = 9
            )

            Spacer(Modifier.height(12.dp))
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                Text("Correct Movement", style = MaterialTheme.typography.titleLarge)
                Switch(checked = state.correctMovement, onCheckedChange = viewModel::updateCorrectMovement)
            }
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                Text("Compensation Observed", style = MaterialTheme.typography.titleLarge)
                Switch(checked = state.compensationObserved, onCheckedChange = viewModel::updateCompensation)
            }

            Spacer(Modifier.height(12.dp))
            OutlinedTextField(
                value = state.assistiveDevice,
                onValueChange = viewModel::updateAssistiveDevice,
                label = { Text("Assistive Device (e.g. Walker, Cane, None)") },
                modifier = Modifier.fillMaxWidth()
            )

            Spacer(Modifier.height(12.dp))
            OutlinedTextField(
                value = state.notes,
                onValueChange = viewModel::updateNotes,
                label = { Text("Therapist Notes") },
                minLines = 3,
                modifier = Modifier.fillMaxWidth()
            )

            state.exportedFilePath?.let {
                Spacer(Modifier.height(12.dp))
                Text("Exported to: $it", style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.secondary)
            }

            Spacer(Modifier.height(24.dp))
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                Button(
                    onClick = viewModel::saveLabelsAndExport,
                    enabled = !state.isExporting,
                    modifier = Modifier.weight(1f)
                ) { Text(if (state.isExporting) "Exporting..." else "Save & Export CSV") }
                Button(onClick = onDone, modifier = Modifier.weight(1f)) { Text("Done") }
            }
        }
    }
}
