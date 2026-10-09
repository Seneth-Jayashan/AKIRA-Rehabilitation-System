package com.rehabresearch.datacollector.ui.ble

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CircularProgressIndicator
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
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import com.rehabresearch.datacollector.ble.BleConnectionState

/**
 * NOTE ON PERMISSIONS: before this screen calls startScan()/connect(), the host
 * Activity must have already requested and been granted BLUETOOTH_SCAN +
 * BLUETOOTH_CONNECT (API 31+) or ACCESS_FINE_LOCATION (API <=30). Wire that
 * with rememberLauncherForActivityResult(RequestMultiplePermissions()) in
 * MainActivity and gate navigation to this screen on permissions being granted.
 */
@Composable
fun BleDeviceScreen(
    onBack: () -> Unit,
    viewModel: BleViewModel = hiltViewModel()
) {
    val connectionState by viewModel.connectionState.collectAsState()
    val linkStats by viewModel.linkStats.collectAsState()

    Scaffold(topBar = { TopAppBar(title = { Text("Connect ESP32 Sensor") }) }) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .padding(20.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            when (val state = connectionState) {
                is BleConnectionState.Disconnected -> {
                    Text("Not connected", style = MaterialTheme.typography.titleLarge)
                    Spacer(Modifier.height(16.dp))
                    Button(onClick = { viewModel.startScan() }) { Text("Search for ESP32") }
                }
                is BleConnectionState.Scanning -> {
                    CircularProgressIndicator()
                    Spacer(Modifier.height(12.dp))
                    Text("Searching...")
                    Spacer(Modifier.height(16.dp))
                    OutlinedButton(onClick = { viewModel.stopScan() }) { Text("Cancel") }
                }
                is BleConnectionState.DeviceFound -> {
                    Card(modifier = Modifier.fillMaxWidth().padding(vertical = 12.dp)) {
                        Column(Modifier.padding(16.dp)) {
                            Text(state.name, style = MaterialTheme.typography.titleLarge)
                            Text("Signal: ${state.rssi} dBm", style = MaterialTheme.typography.bodyMedium)
                        }
                    }
                    Button(onClick = { viewModel.connect(state.address) }) { Text("Connect") }
                }
                is BleConnectionState.Connecting -> {
                    CircularProgressIndicator()
                    Spacer(Modifier.height(12.dp))
                    Text("Connecting...")
                }
                is BleConnectionState.Connected -> {
                    Text("Connected", style = MaterialTheme.typography.titleLarge, color = MaterialTheme.colorScheme.secondary)
                    Text(state.name, style = MaterialTheme.typography.bodyLarge)
                    Spacer(Modifier.height(16.dp))
                    Card(modifier = Modifier.fillMaxWidth()) {
                        Column(Modifier.padding(16.dp)) {
                            Text("Signal Strength: ${linkStats.rssi} dBm")
                            Text("Sample Rate: %.1f Hz".format(linkStats.currentSampleRateHz))
                            Text("Packets Received: ${linkStats.packetsReceived}")
                            Text("Packets Dropped: ${linkStats.packetsDropped}")
                            linkStats.batteryPercent?.let { Text("Battery: $it%") }
                        }
                    }
                    Spacer(Modifier.height(16.dp))
                    OutlinedButton(onClick = { viewModel.disconnect() }) { Text("Disconnect") }
                }
                is BleConnectionState.DiscoveringServices -> {
                    CircularProgressIndicator()
                    Spacer(Modifier.height(12.dp))
                    Text("Discovering services...")
                }
                is BleConnectionState.EnablingNotifications -> {
                    CircularProgressIndicator()
                    Spacer(Modifier.height(12.dp))
                    Text("Enabling sensor notifications...")
                }
                is BleConnectionState.Error -> {
                    Text(state.message, color = MaterialTheme.colorScheme.error)
                    Spacer(Modifier.height(16.dp))
                    Button(onClick = { viewModel.startScan() }) { Text("Retry") }
                }
            }

            Spacer(Modifier.height(24.dp))
            OutlinedButton(onClick = onBack) { Text("Back") }
        }
    }
}
