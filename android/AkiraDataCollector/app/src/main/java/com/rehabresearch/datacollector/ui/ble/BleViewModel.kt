package com.rehabresearch.datacollector.ui.ble

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.rehabresearch.datacollector.ble.BleConnectionState
import com.rehabresearch.datacollector.ble.BleLinkStats
import com.rehabresearch.datacollector.ble.BleManager
import com.rehabresearch.datacollector.ble.DiscoveredBleDevice
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import javax.inject.Inject

@HiltViewModel
class BleViewModel @Inject constructor(
    val bleManager: BleManager
) : ViewModel() {

    val connectionState: StateFlow<BleConnectionState> = bleManager.connectionState
    val linkStats: StateFlow<BleLinkStats> = bleManager.linkStats
    val discoveredDevices: StateFlow<List<DiscoveredBleDevice>> = bleManager.discoveredDevices

    val isBluetoothEnabled: Boolean get() = bleManager.isBluetoothEnabled
    fun isLocationEnabled(): Boolean = bleManager.isLocationEnabled()

    init {
        // Keep RSSI fresh while this screen (or Live Sensor) is on screen
        viewModelScope.launch {
            while (true) {
                if (connectionState.value is BleConnectionState.Connected) {
                    bleManager.refreshRssi()
                }
                delay(2000)
            }
        }
    }

    fun startScan() = bleManager.startScan()
    fun stopScan() = bleManager.stopScan()
    fun loadBondedDevices() = bleManager.loadBondedAndConnectedDevices()
    fun connect(address: String) = bleManager.connect(address)
    fun disconnect() = bleManager.disconnect()
}
