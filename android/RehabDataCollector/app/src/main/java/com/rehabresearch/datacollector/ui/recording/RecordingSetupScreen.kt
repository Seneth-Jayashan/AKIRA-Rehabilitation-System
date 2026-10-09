package com.rehabresearch.datacollector.ui.recording

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.selection.selectable
import androidx.compose.material3.Button
import androidx.compose.material3.ExposedDropdownMenuBox
import androidx.compose.material3.ExposedDropdownMenuDefaults
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.rehabresearch.datacollector.data.local.entity.BodySide
import com.rehabresearch.datacollector.data.local.entity.Difficulty
import com.rehabresearch.datacollector.data.local.entity.ExerciseType

@Composable
fun RecordingSetupScreen(
    viewModel: RecordingViewModel,
    onBeginSession: () -> Unit,
    onBack: () -> Unit
) {
    val setup by viewModel.setup.collectAsState()

    Scaffold(topBar = { TopAppBar(title = { Text("New Session — ${setup.patientId}") }) }) { padding ->
        Column(
            modifier = Modifier
                .padding(padding)
                .padding(20.dp)
                .verticalScroll(rememberScrollState())
        ) {
            Text("Exercise", style = MaterialTheme.typography.titleLarge)
            ExerciseDropdown(setup.exercise) { v -> viewModel.updateSetup { it.copy(exercise = v) } }

            Spacer(Modifier.height(16.dp))
            Text("Side", style = MaterialTheme.typography.titleLarge)
            Row {
                listOf(BodySide.LEFT, BodySide.RIGHT).forEach { side ->
                    Row(
                        Modifier
                            .selectable(selected = setup.side == side, onClick = { viewModel.updateSetup { it.copy(side = side) } })
                            .padding(end = 16.dp)
                    ) {
                        RadioButton(selected = setup.side == side, onClick = { viewModel.updateSetup { it.copy(side = side) } })
                        Text(side.name.lowercase().replaceFirstChar { it.uppercase() }, modifier = Modifier.padding(top = 12.dp))
                    }
                }
            }

            Spacer(Modifier.height(16.dp))
            Text("Difficulty", style = MaterialTheme.typography.titleLarge)
            Row {
                Difficulty.entries.forEach { d ->
                    Row(
                        Modifier
                            .selectable(selected = setup.difficulty == d, onClick = { viewModel.updateSetup { it.copy(difficulty = d) } })
                            .padding(end = 16.dp)
                    ) {
                        RadioButton(selected = setup.difficulty == d, onClick = { viewModel.updateSetup { it.copy(difficulty = d) } })
                        Text(d.name.lowercase().replaceFirstChar { it.uppercase() }, modifier = Modifier.padding(top = 12.dp))
                    }
                }
            }

            Spacer(Modifier.height(16.dp))
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                OutlinedTextField(
                    value = setup.targetReps.toString(),
                    onValueChange = { v -> v.toIntOrNull()?.let { n -> viewModel.updateSetup { it.copy(targetReps = n) } } },
                    label = { Text("Target Repetitions") },
                    modifier = Modifier.weight(1f)
                )
                OutlinedTextField(
                    value = setup.recoveryWeek.toString(),
                    onValueChange = { v -> v.toIntOrNull()?.let { n -> viewModel.updateSetup { it.copy(recoveryWeek = n) } } },
                    label = { Text("Recovery Week") },
                    modifier = Modifier.weight(1f)
                )
            }
            Spacer(Modifier.height(12.dp))
            OutlinedTextField(
                value = setup.therapistName,
                onValueChange = { v -> viewModel.updateSetup { it.copy(therapistName = v) } },
                label = { Text("Therapist") },
                modifier = Modifier.fillMaxWidth()
            )

            Spacer(Modifier.height(28.dp))
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                Button(onClick = onBack, modifier = Modifier.weight(1f)) { Text("Back") }
                Button(onClick = onBeginSession, modifier = Modifier.weight(1f)) { Text("Start Session") }
            }
        }
    }
}

@Composable
private fun ExerciseDropdown(selected: ExerciseType, onSelected: (ExerciseType) -> Unit) {
    var expanded by remember { mutableStateOf(false) }
    ExposedDropdownMenuBox(expanded = expanded, onExpandedChange = { expanded = it }) {
        OutlinedTextField(
            value = selected.name.replace("_", " ").lowercase().replaceFirstChar { it.uppercase() },
            onValueChange = {},
            readOnly = true,
            trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded = expanded) },
            modifier = Modifier.fillMaxWidth().menuAnchor()
        )
        ExposedDropdownMenu(expanded = expanded, onDismissRequest = { expanded = false }) {
            ExerciseType.entries.forEach { ex ->
                DropdownMenuItem(
                    text = { Text(ex.name.replace("_", " ").lowercase().replaceFirstChar { it.uppercase() }) },
                    onClick = { onSelected(ex); expanded = false }
                )
            }
        }
    }
}
