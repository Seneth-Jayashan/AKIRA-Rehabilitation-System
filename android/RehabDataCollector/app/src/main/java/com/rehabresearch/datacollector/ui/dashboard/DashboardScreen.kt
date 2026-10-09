package com.rehabresearch.datacollector.ui.dashboard

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Bluetooth
import androidx.compose.material.icons.filled.BluetoothConnected
import androidx.compose.material.icons.filled.BluetoothDisabled
import androidx.compose.material.icons.filled.History
import androidx.compose.material.icons.filled.People
import androidx.compose.material.icons.filled.PlayCircle
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import com.rehabresearch.datacollector.ble.BleConnectionState

private data class DashboardAction(val label: String, val icon: ImageVector, val onClick: () -> Unit)

@Composable
fun DashboardScreen(
    onNewSession: () -> Unit,
    onPatients: () -> Unit,
    onHistory: () -> Unit,
    onSettings: () -> Unit,
    onBleDevice: () -> Unit,
    viewModel: DashboardViewModel = hiltViewModel()
) {
    val state by viewModel.uiState.collectAsState()

    Scaffold(
        topBar = {
            TopAppBar(title = { Text("Rehab Data Collector") })
        }
    ) { padding ->
        Column(modifier = Modifier
            .fillMaxSize()
            .padding(padding)
            .padding(16.dp)) {

            Row(modifier = Modifier.fillMaxWidth()) {
                StatCard("Today's Patients", state.patientsToday.toString(), Modifier.weight(1f))
                Spacer(Modifier.height(0.dp))
                StatCard("Today's Recordings", state.recordingsToday.toString(), Modifier.weight(1f))
            }
            Spacer(Modifier.height(12.dp))
            BleStatusCard(state.bleState, state.batteryPercent, onClick = onBleDevice)

            Spacer(Modifier.height(20.dp))
            Text("Quick Actions", style = MaterialTheme.typography.titleLarge)
            Spacer(Modifier.height(8.dp))

            val actions = listOf(
                DashboardAction("New Session", Icons.Filled.PlayCircle, onNewSession),
                DashboardAction("Patients", Icons.Filled.People, onPatients),
                DashboardAction("History", Icons.Filled.History, onHistory),
                DashboardAction("Settings", Icons.Filled.Settings, onSettings)
            )

            LazyVerticalGrid(
                columns = GridCells.Fixed(2),
                contentPadding = PaddingValues(vertical = 8.dp),
                horizontalArrangement = Arrangement.spacedBy(12.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                items(actions) { action ->
                    ActionCard(action)
                }
            }
        }
    }
}

@Composable
private fun StatCard(label: String, value: String, modifier: Modifier = Modifier) {
    Card(modifier = modifier.padding(4.dp)) {
        Column(Modifier.padding(16.dp)) {
            Text(value, style = MaterialTheme.typography.headlineLarge)
            Text(label, style = MaterialTheme.typography.bodyMedium)
        }
    }
}

@Composable
private fun BleStatusCard(state: BleConnectionState, batteryPercent: Int?, onClick: () -> Unit) {
    val (icon, label, color) = when (state) {
        is BleConnectionState.Connected -> Triple(Icons.Filled.BluetoothConnected, "Connected: ${state.name}", MaterialTheme.colorScheme.secondary)
        is BleConnectionState.Scanning -> Triple(Icons.Filled.Bluetooth, "Scanning...", MaterialTheme.colorScheme.primary)
        is BleConnectionState.Connecting -> Triple(Icons.Filled.Bluetooth, "Connecting...", MaterialTheme.colorScheme.primary)
        is BleConnectionState.DiscoveringServices -> Triple(Icons.Filled.Bluetooth, "Discovering services...", MaterialTheme.colorScheme.primary)
        is BleConnectionState.EnablingNotifications -> Triple(Icons.Filled.Bluetooth, "Enabling notifications...", MaterialTheme.colorScheme.primary)
        is BleConnectionState.Error -> Triple(Icons.Filled.BluetoothDisabled, state.message, MaterialTheme.colorScheme.error)
        else -> Triple(Icons.Filled.BluetoothDisabled, "Not connected — tap to connect ESP32", MaterialTheme.colorScheme.onSurfaceVariant)
    }
    Card(
        onClick = onClick,
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp)
        ) {
            Icon(icon, contentDescription = null, tint = color)
            Spacer(Modifier.height(0.dp))
            Column(Modifier.padding(start = 12.dp)) {
                Text(label, style = MaterialTheme.typography.bodyLarge)
                batteryPercent?.let { Text("Battery: $it%", style = MaterialTheme.typography.bodyMedium) }
            }
        }
    }
}

@Composable
private fun ActionCard(action: DashboardAction) {
    Card(onClick = action.onClick) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(20.dp)
        ) {
            Icon(action.icon, contentDescription = null, tint = MaterialTheme.colorScheme.primary)
            Spacer(Modifier.height(8.dp))
            Text(action.label, style = MaterialTheme.typography.titleLarge)
        }
    }
}
