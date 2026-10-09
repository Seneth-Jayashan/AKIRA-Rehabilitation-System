package com.rehabresearch.datacollector.ui.patient

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material3.Card
import androidx.compose.material3.FloatingActionButton
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import com.rehabresearch.datacollector.data.local.entity.PatientEntity

@Composable
fun PatientListScreen(
    onAddPatient: () -> Unit,
    onSelectPatient: (String) -> Unit,
    viewModel: PatientViewModel = hiltViewModel()
) {
    val patients by viewModel.patients.collectAsState()

    Scaffold(
        topBar = { TopAppBar(title = { Text("Patients") }) },
        floatingActionButton = {
            FloatingActionButton(onClick = onAddPatient) {
                Icon(Icons.Filled.Add, contentDescription = "Add patient")
            }
        }
    ) { padding ->
        if (patients.isEmpty()) {
            Column(modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .padding(24.dp)) {
                Text("No patients yet. Tap + to register your first patient.")
            }
        } else {
            LazyColumn(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(padding),
                contentPadding = PaddingValues(16.dp)
            ) {
                items(patients, key = { it.patientId }) { patient ->
                    PatientRow(patient, onClick = { onSelectPatient(patient.patientId) })
                }
            }
        }
    }
}

@Composable
private fun PatientRow(patient: PatientEntity, onClick: () -> Unit) {
    Card(
        onClick = onClick,
        modifier = Modifier
            .fillMaxSize()
            .padding(vertical = 6.dp)
    ) {
        Column(Modifier.padding(16.dp)) {
            Text("${patient.patientId} — ${patient.name}", style = MaterialTheme.typography.titleLarge)
            Text(
                "${patient.age} yrs · ${patient.gender.name.lowercase().replaceFirstChar { it.uppercase() }} · " +
                    "${patient.surgerySide.name.lowercase().replaceFirstChar { it.uppercase() }} " +
                    patient.surgeryType.name.replace("_", " ").lowercase().replaceFirstChar { it.uppercase() },
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            Text(
                "BMI: %.1f · Therapist: %s".format(patient.bmi, patient.physiotherapistName),
                style = MaterialTheme.typography.bodyMedium
            )
        }
    }
}
