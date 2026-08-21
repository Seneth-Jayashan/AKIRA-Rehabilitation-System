package com.rehabresearch.datacollector.ui.ble

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.rehabresearch.datacollector.ble.BleConnectionState
import com.rehabresearch.datacollector.ble.BleLinkStats
import com.rehabresearch.datacollector.ble.BleManager
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import javax.inject.Inject

@HiltViewModel
class BleViewModel @Inject constructor(
    val bleManager: BleManager
) : ViewModel() {

    val connectionState = bleManager.connectionState
    val linkStats = bleManager.linkStats

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
    fun connect(address: String) = bleManager.connect(address)
    fun disconnect() = bleManager.disconnect()
}
