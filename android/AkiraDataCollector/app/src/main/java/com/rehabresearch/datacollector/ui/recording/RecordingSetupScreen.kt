package com.rehabresearch.datacollector.ui.recording

import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.selection.selectable
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ElevatedCard
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ExposedDropdownMenuBox
import androidx.compose.material3.ExposedDropdownMenuDefaults
import androidx.compose.material3.FilterChip
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
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
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.rehabresearch.datacollector.data.local.entity.BodySide
import com.rehabresearch.datacollector.data.local.entity.Difficulty
import com.rehabresearch.datacollector.data.local.entity.ExerciseType
import com.rehabresearch.datacollector.data.local.entity.RehabComponent
import com.rehabresearch.datacollector.data.local.entity.component

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun RecordingSetupScreen(
    viewModel: RecordingViewModel,
    onBeginSession: () -> Unit,
    onBack: () -> Unit
) {
    val setup by viewModel.setup.collectAsState()

    Scaffold(topBar = { TopAppBar(title = { Text("New Trial — ${setup.patientId}") }) }) { padding ->
        Column(
            modifier = Modifier
                .padding(padding)
                .padding(20.dp)
                .verticalScroll(rememberScrollState())
        ) {
            // 1. Select Rehabilitation Component (4 Members)
            Text("Rehabilitation Component", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
            Spacer(Modifier.height(6.dp))
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .horizontalScroll(rememberScrollState()),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                RehabComponent.entries.forEach { comp ->
                    FilterChip(
                        selected = setup.component == comp,
                        onClick = { viewModel.setComponent(comp) },
                        label = {
                            Text("${comp.displayName.substringBefore(" ")} (${comp.memberLabel})")
                        }
                    )
                }
            }

            Spacer(Modifier.height(10.dp))

            // Sensor Placement Info Card for Selected Member
            ElevatedCard(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.elevatedCardColors(containerColor = MaterialTheme.colorScheme.secondaryContainer)
            ) {
                Column(Modifier.padding(12.dp)) {
                    Text(
                        "${setup.component.displayName} (${setup.component.memberLabel}) Sensor Setup:",
                        style = MaterialTheme.typography.labelMedium,
                        fontWeight = FontWeight.Bold,
                        color = MaterialTheme.colorScheme.onSecondaryContainer
                    )
                    Spacer(Modifier.height(4.dp))
                    Text("• Sensor 1 (CH0): ${setup.sensor1Placement}", style = MaterialTheme.typography.bodySmall)
                    Text("• Sensor 2 (CH1): ${setup.sensor2Placement}", style = MaterialTheme.typography.bodySmall)
                }
            }

            Spacer(Modifier.height(16.dp))

            // 2. Select Exercise (Filtered by Component)
            Text("Exercise", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
            ExerciseDropdown(
                selected = setup.exercise,
                component = setup.component,
                onSelected = { v -> viewModel.updateSetup { it.copy(exercise = v) } }
            )

            Spacer(Modifier.height(16.dp))

            // 3. Trial Number and Repetitions
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                OutlinedTextField(
                    value = setup.trialNumber.toString(),
                    onValueChange = { v ->
                        v.toIntOrNull()?.let { n -> viewModel.setTrialNumber(n) }
                    },
                    label = { Text("Trial Number (e.g. 1, 2, 3)") },
                    modifier = Modifier.weight(1f)
                )
                OutlinedTextField(
                    value = setup.targetReps.toString(),
                    onValueChange = { v ->
                        v.toIntOrNull()?.let { n -> viewModel.updateSetup { it.copy(targetReps = n) } }
                    },
                    label = { Text("Target Reps") },
                    modifier = Modifier.weight(1f)
                )
            }

            Spacer(Modifier.height(16.dp))

            // 4. Body Side
            Text("Side", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
            Row {
                listOf(BodySide.RIGHT, BodySide.LEFT).forEach { side ->
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

            // 5. Difficulty Level
            Text("Difficulty", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
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

            // 6. Recovery Week & Evaluator
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                OutlinedTextField(
                    value = setup.recoveryWeek.toString(),
                    onValueChange = { v -> v.toIntOrNull()?.let { n -> viewModel.updateSetup { it.copy(recoveryWeek = n) } } },
                    label = { Text("Recovery Week") },
                    modifier = Modifier.weight(1f)
                )
                OutlinedTextField(
                    value = setup.therapistName,
                    onValueChange = { v -> viewModel.updateSetup { it.copy(therapistName = v) } },
                    label = { Text("Evaluator / Member") },
                    modifier = Modifier.weight(1f)
                )
            }

            Spacer(Modifier.height(28.dp))
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                OutlinedButton(onClick = onBack, modifier = Modifier.weight(1f)) { Text("Back") }
                Button(onClick = onBeginSession, modifier = Modifier.weight(1f)) { Text("Start Recording") }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun ExerciseDropdown(
    selected: ExerciseType,
    component: RehabComponent,
    onSelected: (ExerciseType) -> Unit
) {
    var expanded by remember { mutableStateOf(false) }
    val availableExercises = remember(component) {
        ExerciseType.entries.filter { it.component == component }
    }

    ExposedDropdownMenuBox(expanded = expanded, onExpandedChange = { expanded = it }) {
        OutlinedTextField(
            value = selected.name.replace("_", " ").lowercase().replaceFirstChar { it.uppercase() },
            onValueChange = {},
            readOnly = true,
            trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded = expanded) },
            modifier = Modifier.fillMaxWidth().menuAnchor()
        )
        ExposedDropdownMenu(expanded = expanded, onDismissRequest = { expanded = false }) {
            availableExercises.forEach { ex ->
                DropdownMenuItem(
                    text = { Text(ex.name.replace("_", " ").lowercase().replaceFirstChar { it.uppercase() }) },
                    onClick = { onSelected(ex); expanded = false }
                )
            }
        }
    }
}
