package com.rehabresearch.datacollector.ui.history

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.AssistChip
import androidx.compose.material3.Card
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
import com.rehabresearch.datacollector.data.local.entity.SessionEntity
import com.rehabresearch.datacollector.data.local.entity.SessionStatus

@Composable
fun HistoryScreen(
    onBack: () -> Unit,
    viewModel: HistoryViewModel = hiltViewModel()
) {
    val sessions by viewModel.sessions.collectAsState()

    Scaffold(topBar = { TopAppBar(title = { Text("Collected Sessions") }) }) { padding ->
        if (sessions.isEmpty()) {
            Column(Modifier.padding(padding).padding(24.dp)) {
                Text("No sessions recorded yet.")
            }
        } else {
            LazyColumn(
                modifier = Modifier.fillMaxSize().padding(padding),
                contentPadding = PaddingValues(16.dp)
            ) {
                items(sessions, key = { it.sessionId }) { session ->
                    SessionRow(session)
                }
            }
        }
    }
}

@Composable
private fun SessionRow(session: SessionEntity) {
    Card(modifier = Modifier.fillMaxWidth().padding(vertical = 6.dp)) {
        Column(Modifier.padding(16.dp)) {
            Text(
                "${session.exercise.name.replace("_", " ")} — ${session.patientId}",
                style = MaterialTheme.typography.titleLarge
            )
            Text(
                "${session.side.name} · Week ${session.recoveryWeek} · ${session.actualReps}/${session.targetReps} reps",
                style = MaterialTheme.typography.bodyMedium
            )
            Text(
                "%.1fs · %.0f Hz avg · %d dropped".format(
                    session.durationMillis / 1000f, session.avgSampleFrequencyHz, session.droppedPacketCount
                ),
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            androidx.compose.foundation.layout.Spacer(Modifier.padding(top = 4.dp))
            AssistChip(onClick = {}, label = { Text(statusLabel(session.status)) })
        }
    }
}

private fun statusLabel(status: SessionStatus) = when (status) {
    SessionStatus.RECORDING -> "Recording"
    SessionStatus.COMPLETED -> "Completed — not exported"
    SessionStatus.EXPORTED -> "Exported"
}
