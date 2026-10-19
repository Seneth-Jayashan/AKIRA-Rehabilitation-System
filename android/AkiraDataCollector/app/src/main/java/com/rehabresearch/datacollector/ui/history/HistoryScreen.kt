package com.rehabresearch.datacollector.ui.history

import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.FileDownload
import androidx.compose.material.icons.filled.Share
import androidx.compose.material3.AssistChip
import androidx.compose.material3.Button
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ElevatedCard
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import com.rehabresearch.datacollector.data.local.entity.RehabComponent
import com.rehabresearch.datacollector.data.local.entity.SessionEntity
import com.rehabresearch.datacollector.data.local.entity.SessionStatus

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun HistoryScreen(
    onBack: () -> Unit,
    viewModel: HistoryViewModel = hiltViewModel()
) {
    val sessions by viewModel.sessions.collectAsState()
    val patients by viewModel.patients.collectAsState()
    val selectedParticipantId by viewModel.selectedParticipantId.collectAsState()
    val selectedComponent by viewModel.selectedComponent.collectAsState()
    val context = LocalContext.current

    Scaffold(topBar = { TopAppBar(title = { Text("Recorded Dataset History") }) }) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
        ) {
            // 1. Filter by Component (4 Members)
            Column(modifier = Modifier.padding(horizontal = 16.dp, vertical = 4.dp)) {
                Text("Body Component (4 Members):", style = MaterialTheme.typography.labelMedium)
                Spacer(Modifier.height(4.dp))
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .horizontalScroll(rememberScrollState()),
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    FilterChip(
                        selected = selectedComponent == null,
                        onClick = { viewModel.selectComponentFilter(null) },
                        label = { Text("All Components") }
                    )
                    RehabComponent.entries.forEach { comp ->
                        FilterChip(
                            selected = selectedComponent == comp,
                            onClick = { viewModel.selectComponentFilter(comp) },
                            label = { Text("${comp.displayName.substringBefore(" ")} (${comp.memberLabel})") }
                        )
                    }
                }
            }

            // 2. Filter by Participant
            Column(modifier = Modifier.padding(horizontal = 16.dp, vertical = 4.dp)) {
                Text("Participant / Member ID:", style = MaterialTheme.typography.labelMedium)
                Spacer(Modifier.height(4.dp))
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .horizontalScroll(rememberScrollState()),
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    FilterChip(
                        selected = selectedParticipantId == null,
                        onClick = { viewModel.selectParticipantFilter(null) },
                        label = { Text("All Members") }
                    )
                    patients.forEach { p ->
                        FilterChip(
                            selected = selectedParticipantId == p.patientId,
                            onClick = { viewModel.selectParticipantFilter(p.patientId) },
                            label = { Text("${p.patientId} (${p.name})") }
                        )
                    }
                }
            }

            // 3. Export Active Selection Card
            ElevatedCard(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 16.dp, vertical = 6.dp),
                colors = CardDefaults.elevatedCardColors(containerColor = MaterialTheme.colorScheme.primaryContainer)
            ) {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(14.dp),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Column(modifier = Modifier.weight(1f)) {
                        val compText = selectedComponent?.displayName ?: "All Components"
                        val pidText = selectedParticipantId ?: "All Participants"
                        Text(
                            "$compText · $pidText",
                            style = MaterialTheme.typography.titleMedium,
                            color = MaterialTheme.colorScheme.onPrimaryContainer
                        )
                        Text(
                            "${sessions.size} recorded trials matching filter",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onPrimaryContainer
                        )
                    }
                    Button(
                        onClick = { viewModel.exportAndShareActiveSelection(context) },
                        enabled = sessions.isNotEmpty()
                    ) {
                        Icon(Icons.Filled.Share, contentDescription = null)
                        Spacer(Modifier.padding(horizontal = 4.dp))
                        Text(
                            if (selectedComponent == RehabComponent.ANKLE) "Export My Ankle Data"
                            else if (selectedComponent != null) "Export ${selectedComponent?.displayName?.substringBefore(" ")}"
                            else "Export Selection"
                        )
                    }
                }
            }

            if (sessions.isEmpty()) {
                Column(Modifier.padding(24.dp)) {
                    Text("No recordings found matching the selected component and participant filter.")
                }
            } else {
                LazyColumn(
                    modifier = Modifier.fillMaxSize(),
                    contentPadding = PaddingValues(16.dp),
                    verticalArrangement = Arrangement.spacedBy(10.dp)
                ) {
                    items(sessions, key = { it.sessionId }) { session ->
                        SessionRow(
                            session = session,
                            onExportShare = { viewModel.exportAndShareSession(context, session) }
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun SessionRow(
    session: SessionEntity,
    onExportShare: () -> Unit
) {
    ElevatedCard(
        modifier = Modifier.fillMaxWidth(),
        elevation = CardDefaults.elevatedCardElevation(defaultElevation = 2.dp),
        colors = CardDefaults.elevatedCardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)
    ) {
        Column(Modifier.padding(16.dp)) {
            Row(
                Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    "${session.exercise.name.replace("_", " ")} — ${session.trialId}",
                    style = MaterialTheme.typography.titleMedium
                )
                AssistChip(onClick = {}, label = { Text(statusLabel(session.status)) })
            }

            Text(
                "${session.component.displayName} (${session.component.memberLabel}) · ${session.patientId} · Side: ${session.side.name}",
                style = MaterialTheme.typography.bodyMedium
            )
            Text(
                "Sensors: ${session.sensor1Placement} (S1) + ${session.sensor2Placement} (S2)",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.primary
            )
            Text(
                "%.1fs duration · %.0f Hz avg · %d dropped packets".format(
                    session.durationMillis / 1000f, session.avgSampleFrequencyHz, session.droppedPacketCount
                ),
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )

            Spacer(Modifier.height(10.dp))

            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.End) {
                OutlinedButton(onClick = onExportShare) {
                    Icon(Icons.Filled.FileDownload, contentDescription = null)
                    Spacer(Modifier.padding(horizontal = 4.dp))
                    Text("Export / Share CSV")
                }
            }
        }
    }
}

private fun statusLabel(status: SessionStatus) = when (status) {
    SessionStatus.RECORDING -> "Recording"
    SessionStatus.COMPLETED -> "Completed"
    SessionStatus.EXPORTED -> "Exported"
}
