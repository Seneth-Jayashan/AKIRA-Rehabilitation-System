package com.rehabresearch.datacollector.ui.patient

import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ExposedDropdownMenuBox
import androidx.compose.material3.ExposedDropdownMenuDefaults
import androidx.compose.material3.FilterChip
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
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
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import com.rehabresearch.datacollector.data.local.entity.BodySide
import com.rehabresearch.datacollector.data.local.entity.Gender
import com.rehabresearch.datacollector.data.local.entity.RehabComponent
import com.rehabresearch.datacollector.data.local.entity.SurgeryType

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun AddPatientScreen(
    onSaved: () -> Unit,
    onBack: () -> Unit,
    viewModel: PatientViewModel = hiltViewModel()
) {
    val form by viewModel.form.collectAsState()
    var selectedComponentFilter by remember { mutableStateOf<RehabComponent?>(RehabComponent.ANKLE) }

    LaunchedEffect(form.saved) {
        if (form.saved) onSaved()
    }

    val availableSurgeries = remember(selectedComponentFilter) {
        if (selectedComponentFilter == null) {
            SurgeryType.entries.toList()
        } else {
            SurgeryType.entries.filter { it.component == selectedComponentFilter || it.component == null }
        }
    }

    Scaffold(topBar = { TopAppBar(title = { Text("New Patient") }) }) { padding ->
        Column(
            modifier = Modifier
                .padding(padding)
                .padding(20.dp)
                .verticalScroll(rememberScrollState())
        ) {
            OutlinedTextField(
                value = form.name,
                onValueChange = { v -> viewModel.updateForm { it.copy(name = v) } },
                label = { Text("Full name") },
                modifier = Modifier.fillMaxWidth()
            )
            Spacer(Modifier.height(10.dp))

            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                OutlinedTextField(
                    value = form.age,
                    onValueChange = { v -> viewModel.updateForm { it.copy(age = v.filter { c -> c.isDigit() }) } },
                    label = { Text("Age") },
                    keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                    modifier = Modifier.weight(1f)
                )
                EnumDropdown(
                    label = "Gender",
                    options = Gender.entries.toList(),
                    selected = form.gender,
                    display = { it.name.lowercase().replaceFirstChar { c -> c.uppercase() } },
                    onSelected = { v -> viewModel.updateForm { it.copy(gender = v) } },
                    modifier = Modifier.weight(1f)
                )
            }
            Spacer(Modifier.height(10.dp))

            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                OutlinedTextField(
                    value = form.heightCm,
                    onValueChange = { v -> viewModel.updateForm { it.copy(heightCm = v.filter { c -> c.isDigit() || c == '.' }) } },
                    label = { Text("Height (cm)") },
                    keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Decimal),
                    modifier = Modifier.weight(1f)
                )
                OutlinedTextField(
                    value = form.weightKg,
                    onValueChange = { v -> viewModel.updateForm { it.copy(weightKg = v.filter { c -> c.isDigit() || c == '.' }) } },
                    label = { Text("Weight (kg)") },
                    keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Decimal),
                    modifier = Modifier.weight(1f)
                )
            }
            Spacer(Modifier.height(14.dp))

            // Component Quick-Filter Chips
            Text(
                "Relevant Rehab Component",
                style = MaterialTheme.typography.labelMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            Spacer(Modifier.height(6.dp))
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .horizontalScroll(rememberScrollState()),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                FilterChip(
                    selected = selectedComponentFilter == RehabComponent.ANKLE,
                    onClick = {
                        selectedComponentFilter = RehabComponent.ANKLE
                        viewModel.updateForm { it.copy(surgeryType = SurgeryType.ANKLE_ORIF) }
                    },
                    label = { Text("Ankle (Member 1)") }
                )
                FilterChip(
                    selected = selectedComponentFilter == RehabComponent.KNEE,
                    onClick = {
                        selectedComponentFilter = RehabComponent.KNEE
                        viewModel.updateForm { it.copy(surgeryType = SurgeryType.TOTAL_KNEE_REPLACEMENT) }
                    },
                    label = { Text("Knee (Member 2)") }
                )
                FilterChip(
                    selected = selectedComponentFilter == RehabComponent.WRIST,
                    onClick = {
                        selectedComponentFilter = RehabComponent.WRIST
                        viewModel.updateForm { it.copy(surgeryType = SurgeryType.DISTAL_RADIUS_ORIF) }
                    },
                    label = { Text("Wrist (Member 3)") }
                )
                FilterChip(
                    selected = selectedComponentFilter == RehabComponent.ELBOW,
                    onClick = {
                        selectedComponentFilter = RehabComponent.ELBOW
                        viewModel.updateForm { it.copy(surgeryType = SurgeryType.OLECRANON_ORIF) }
                    },
                    label = { Text("Elbow (Member 4)") }
                )
                FilterChip(
                    selected = selectedComponentFilter == null,
                    onClick = { selectedComponentFilter = null },
                    label = { Text("All Surgeries") }
                )
            }
            Spacer(Modifier.height(10.dp))

            // Full-width Surgery Selector with Component Categories
            SurgeryDropdown(
                label = "Surgery / Orthopedic Condition",
                options = availableSurgeries,
                selected = form.surgeryType,
                onSelected = { v -> viewModel.updateForm { it.copy(surgeryType = v) } },
                modifier = Modifier.fillMaxWidth()
            )
            Spacer(Modifier.height(10.dp))

            // Side and Therapist Name
            EnumDropdown(
                label = "Affected Side",
                options = BodySide.entries.toList(),
                selected = form.surgerySide,
                display = { it.name.lowercase().replaceFirstChar { c -> c.uppercase() } },
                onSelected = { v -> viewModel.updateForm { it.copy(surgerySide = v) } },
                modifier = Modifier.fillMaxWidth()
            )
            Spacer(Modifier.height(10.dp))

            OutlinedTextField(
                value = form.therapistName,
                onValueChange = { v -> viewModel.updateForm { it.copy(therapistName = v) } },
                label = { Text("Physiotherapist name") },
                modifier = Modifier.fillMaxWidth()
            )

            form.errorMessage?.let {
                Spacer(Modifier.height(8.dp))
                Text(it, color = MaterialTheme.colorScheme.error)
            }

            Spacer(Modifier.height(24.dp))
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                OutlinedButton(onClick = onBack, modifier = Modifier.weight(1f)) { Text("Cancel") }
                Button(onClick = viewModel::savePatient, modifier = Modifier.weight(1f)) { Text("Save Patient") }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun SurgeryDropdown(
    label: String,
    options: List<SurgeryType>,
    selected: SurgeryType,
    onSelected: (SurgeryType) -> Unit,
    modifier: Modifier = Modifier
) {
    var expanded by remember { mutableStateOf(false) }
    ExposedDropdownMenuBox(
        expanded = expanded,
        onExpandedChange = { expanded = it },
        modifier = modifier
    ) {
        OutlinedTextField(
            value = selected.displayName,
            onValueChange = {},
            readOnly = true,
            label = { Text(label) },
            trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded = expanded) },
            modifier = Modifier
                .fillMaxWidth()
                .menuAnchor()
        )
        ExposedDropdownMenu(
            expanded = expanded,
            onDismissRequest = { expanded = false }
        ) {
            var lastCategory = ""
            options.forEach { surgery ->
                if (surgery.category != lastCategory) {
                    lastCategory = surgery.category
                    DropdownMenuItem(
                        text = {
                            Text(
                                text = "— ${lastCategory.uppercase()} —",
                                style = MaterialTheme.typography.labelSmall,
                                fontWeight = FontWeight.Bold,
                                color = MaterialTheme.colorScheme.primary
                            )
                        },
                        onClick = {},
                        enabled = false
                    )
                }
                DropdownMenuItem(
                    text = {
                        Column {
                            Text(surgery.displayName, style = MaterialTheme.typography.bodyMedium)
                            Text(
                                surgery.category,
                                style = MaterialTheme.typography.labelSmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                    },
                    onClick = {
                        onSelected(surgery)
                        expanded = false
                    }
                )
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun <T> EnumDropdown(
    label: String,
    options: List<T>,
    selected: T,
    display: (T) -> String,
    onSelected: (T) -> Unit,
    modifier: Modifier = Modifier
) {
    var expanded by remember { mutableStateOf(false) }
    ExposedDropdownMenuBox(expanded = expanded, onExpandedChange = { expanded = it }, modifier = modifier) {
        OutlinedTextField(
            value = display(selected),
            onValueChange = {},
            readOnly = true,
            label = { Text(label) },
            trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded = expanded) },
            modifier = Modifier.fillMaxWidth().menuAnchor()
        )
        ExposedDropdownMenu(expanded = expanded, onDismissRequest = { expanded = false }) {
            options.forEach { option ->
                DropdownMenuItem(
                    text = { Text(display(option)) },
                    onClick = { onSelected(option); expanded = false }
                )
            }
        }
    }
}
