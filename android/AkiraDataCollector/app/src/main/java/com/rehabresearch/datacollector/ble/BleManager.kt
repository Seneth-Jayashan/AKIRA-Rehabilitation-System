package com.rehabresearch.datacollector.ble

import android.Manifest
import android.annotation.SuppressLint
import android.bluetooth.BluetoothAdapter
import android.bluetooth.BluetoothDevice
import android.bluetooth.BluetoothGatt
import android.bluetooth.BluetoothGattCallback
import android.bluetooth.BluetoothGattCharacteristic
import android.bluetooth.BluetoothManager
import android.bluetooth.BluetoothProfile
import android.bluetooth.le.ScanCallback
import android.bluetooth.le.ScanFilter
import android.bluetooth.le.ScanResult
import android.bluetooth.le.ScanSettings
import android.content.Context
import android.content.pm.PackageManager
import android.location.LocationManager
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.os.ParcelUuid
import android.util.Log
import androidx.core.content.ContextCompat
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.flow.asStateFlow
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Owns the entire BLE lifecycle: scan -> connect -> discover services ->
 * subscribe to the IMU characteristic -> stream parsed packets out as a Flow.
 *
 * This class deliberately does NOT touch Room or any recording session logic —
 * it only knows about Bluetooth. RecordingViewModel is responsible for turning
 * the packet stream into stored SensorReadingEntity rows. Keeping these
 * separate means the BLE code can be tested/reused independent of storage.
 */
@SuppressLint("MissingPermission") // caller or hasRequiredPermissions() checks runtime permissions
@Singleton
class BleManager @Inject constructor(
    private val context: Context
) {
    private val bluetoothManager = context.getSystemService(Context.BLUETOOTH_SERVICE) as BluetoothManager
    private val adapter: BluetoothAdapter? get() = bluetoothManager.adapter
    private val mainHandler = Handler(Looper.getMainLooper())

    private var gatt: BluetoothGatt? = null
    private val packetTracker = PacketStatsTracker(targetHz = 100)

    private val _connectionState = MutableStateFlow<BleConnectionState>(BleConnectionState.Disconnected)
    val connectionState: StateFlow<BleConnectionState> = _connectionState.asStateFlow()

    private val _linkStats = MutableStateFlow(BleLinkStats())
    val linkStats: StateFlow<BleLinkStats> = _linkStats.asStateFlow()

    // SharedFlow (not StateFlow) because we want every packet delivered, not just the latest
    private val _imuPackets = MutableSharedFlow<ImuPacket>(extraBufferCapacity = 256)
    val imuPackets = _imuPackets.asSharedFlow()

    val isBluetoothEnabled: Boolean get() = adapter?.isEnabled == true

    /** Check if runtime permissions for BLE scanning are granted. */
    fun hasRequiredPermissions(): Boolean {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            ContextCompat.checkSelfPermission(context, Manifest.permission.BLUETOOTH_SCAN) == PackageManager.PERMISSION_GRANTED &&
            ContextCompat.checkSelfPermission(context, Manifest.permission.BLUETOOTH_CONNECT) == PackageManager.PERMISSION_GRANTED
        } else {
            ContextCompat.checkSelfPermission(context, Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED ||
            ContextCompat.checkSelfPermission(context, Manifest.permission.ACCESS_COARSE_LOCATION) == PackageManager.PERMISSION_GRANTED
        }
    }

    private val scanTimeoutRunnable = Runnable {
        if (_connectionState.value is BleConnectionState.Scanning) {
            Log.w(BleConstants.LOG_TAG, "Scan timed out after ${BleConstants.SCAN_TIMEOUT_MILLIS}ms")
            stopScan()
            _connectionState.value = BleConnectionState.Error(
                "No ESP32 sensor found. Ensure the sensor is powered on, in range, and advertising."
            )
        }
    }

    fun startScan() {
        val currentAdapter = adapter
        if (currentAdapter == null) {
            _connectionState.value = BleConnectionState.Error("Bluetooth hardware is not available on this device")
            return
        }
        if (!currentAdapter.isEnabled) {
            _connectionState.value = BleConnectionState.Error("Bluetooth is turned OFF. Please enable Bluetooth to search for sensors.")
            return
        }

        // On Android <= 11, Location Services (GPS) must be enabled for BLE scanning
        if (Build.VERSION.SDK_INT <= Build.VERSION_CODES.R) {
            val locationManager = context.getSystemService(Context.LOCATION_SERVICE) as? LocationManager
            if (locationManager != null && !locationManager.isLocationEnabled) {
                _connectionState.value = BleConnectionState.Error("Location (GPS) must be turned ON for Bluetooth scanning on Android 11 and below.")
                return
            }
        }

        // Verify runtime permissions before starting scan
        if (!hasRequiredPermissions()) {
            _connectionState.value = BleConnectionState.Error("Bluetooth scan permission is required. Please grant permissions in app settings.")
            return
        }

        // Stop any previous scan session cleanly
        stopScan()

        val bleScanner = currentAdapter.bluetoothLeScanner
        if (bleScanner == null) {
            _connectionState.value = BleConnectionState.Error("Bluetooth LE scanner unavailable. Try toggling Bluetooth.")
            return
        }

        _connectionState.value = BleConnectionState.Scanning
        Log.i(BleConstants.LOG_TAG, "Starting BLE scan (Low Latency)...")

        val settings = ScanSettings.Builder()
            .setScanMode(ScanSettings.SCAN_MODE_LOW_LATENCY)
            .setReportDelay(0)
            .build()

        try {
            // Unfiltered scan with in-code filtering allows discovering devices even if
            // the name or service UUID is delivered across separate advertisement / scan response packets.
            bleScanner.startScan(null, settings, scanCallback)
        } catch (e: SecurityException) {
            Log.e(BleConstants.LOG_TAG, "SecurityException starting BLE scan", e)
            _connectionState.value = BleConnectionState.Error("Bluetooth scan permission denied. Please grant permissions in app settings.")
            return
        } catch (e: Exception) {
            Log.e(BleConstants.LOG_TAG, "Exception starting BLE scan", e)
            _connectionState.value = BleConnectionState.Error("Failed to start scan: ${e.message}")
            return
        }

        mainHandler.removeCallbacks(scanTimeoutRunnable)
        mainHandler.postDelayed(scanTimeoutRunnable, BleConstants.SCAN_TIMEOUT_MILLIS)
    }

    fun stopScan() {
        mainHandler.removeCallbacks(scanTimeoutRunnable)
        try {
            adapter?.bluetoothLeScanner?.stopScan(scanCallback)
        } catch (e: Exception) {
            Log.w(BleConstants.LOG_TAG, "Error stopping scan: ${e.message}")
        }
    }

    private val scanCallback = object : ScanCallback() {
        override fun onScanResult(callbackType: Int, result: ScanResult) {
            processScanResult(result)
        }

        override fun onBatchScanResults(results: MutableList<ScanResult>) {
            for (result in results) {
                processScanResult(result)
                if (_connectionState.value is BleConnectionState.DeviceFound) break
            }
        }

        override fun onScanFailed(errorCode: Int) {
            val message = when (errorCode) {
                SCAN_FAILED_ALREADY_STARTED -> "Scan already in progress. Retrying..."
                SCAN_FAILED_APPLICATION_REGISTRATION_FAILED -> "Bluetooth stack registration error. Please toggle Bluetooth."
                SCAN_FAILED_INTERNAL_ERROR -> "Internal Bluetooth error. Please toggle Bluetooth."
                SCAN_FAILED_FEATURE_UNSUPPORTED -> "BLE scanning unsupported on this hardware."
                SCAN_FAILED_OUT_OF_HARDWARE_RESOURCES -> "Out of Bluetooth hardware resources. Close other BLE apps and retry."
                else -> "Scan failed (error code $errorCode)"
            }
            Log.e(BleConstants.LOG_TAG, "BLE scan failed: $message (code $errorCode)")
            _connectionState.value = BleConnectionState.Error(message)
        }
    }

    private fun processScanResult(result: ScanResult) {
        val device = result.device
        val scanRecord = result.scanRecord

        // CRITICAL FIX: device.name is often null on first discovery before OS bonding/caching.
        // The advertised local name is stored in scanRecord.deviceName.
        val advertisedName = scanRecord?.deviceName ?: runCatching { device.name }.getOrNull()

        // Check if advertisement contains our target IMU Service UUID
        val hasMatchingServiceUuid = scanRecord?.serviceUuids?.any {
            it.uuid == BleConstants.IMU_SERVICE_UUID
        } == true

        // Check if advertised name matches prefix or contains ESP32 / REHAB / AKIRA (case-insensitive)
        val matchesName = advertisedName != null && (
            advertisedName.startsWith(BleConstants.DEVICE_NAME_PREFIX, ignoreCase = true) ||
            advertisedName.contains("ESP32", ignoreCase = true) ||
            advertisedName.contains("REHAB", ignoreCase = true) ||
            advertisedName.contains("AKIRA", ignoreCase = true)
        )

        if (matchesName || hasMatchingServiceUuid) {
            val resolvedName = advertisedName ?: "ESP32-REHAB Sensor"
            Log.i(BleConstants.LOG_TAG, "Discovered target BLE device: '$resolvedName' [${device.address}] RSSI: ${result.rssi} dBm")
            stopScan()
            _connectionState.value = BleConnectionState.DeviceFound(
                name = resolvedName,
                address = device.address,
                rssi = result.rssi
            )
        }
    }

    fun connect(deviceAddress: String) {
        stopScan()
        val device: BluetoothDevice = adapter?.getRemoteDevice(deviceAddress) ?: run {
            _connectionState.value = BleConnectionState.Error("Unknown device address")
            return
        }
        _connectionState.value = BleConnectionState.Connecting
        packetTracker.reset()
        gatt = device.connectGatt(context, false, gattCallback, BluetoothDevice.TRANSPORT_LE)
    }

    fun disconnect() {
        gatt?.disconnect()
        gatt?.close()
        gatt = null
        _connectionState.value = BleConnectionState.Disconnected
    }

    private val gattCallback = object : BluetoothGattCallback() {
        override fun onConnectionStateChange(g: BluetoothGatt, status: Int, newState: Int) {
            when (newState) {
                BluetoothProfile.STATE_CONNECTED -> {

                    Log.d("BLE", "Connected")

                    g.requestMtu(BleConstants.MTU_SIZE)
                }
                BluetoothProfile.STATE_DISCONNECTED -> {
                    _connectionState.value = BleConnectionState.Disconnected
                    g.close()
                }
            }
        }

        override fun onMtuChanged(
            gatt: BluetoothGatt,
            mtu: Int,
            status: Int
        ) {

            Log.d("BLE", "MTU = $mtu")

            _connectionState.value = BleConnectionState.DiscoveringServices
            gatt.discoverServices()
        }

        override fun onServicesDiscovered(g: BluetoothGatt, status: Int) {
            Log.d("BLE", "Services discovered")
            if (status != BluetoothGatt.GATT_SUCCESS) {
                _connectionState.value = BleConnectionState.Error("Service discovery failed")
                return
            }
            val service = g.getService(BleConstants.IMU_SERVICE_UUID)
            val characteristic = service?.getCharacteristic(BleConstants.IMU_DATA_CHARACTERISTIC_UUID)
            Log.d(
                "BLE",
                "Characteristic = ${characteristic?.uuid}"
            )
            if (characteristic == null) {
                _connectionState.value = BleConnectionState.Error("IMU characteristic not found — check firmware UUIDs")
                return
            }
            _connectionState.value = BleConnectionState.EnablingNotifications
            subscribeToNotifications(g, characteristic)
            // NOTE: we deliberately do NOT set Connected here. The descriptor write to
            // enable notifications is async — declaring "Connected" before it actually
            // succeeds would show a green light on screen while packets might still be
            // silently not flowing. onDescriptorWrite() below is the real signal.
        }
        override fun onDescriptorWrite(
            gatt: BluetoothGatt,
            descriptor: android.bluetooth.BluetoothGattDescriptor,
            status: Int
        ) {
            super.onDescriptorWrite(gatt, descriptor, status)

            if (status == BluetoothGatt.GATT_SUCCESS) {

                Log.d(
                    "BLE",
                    "Notification descriptor written successfully"
                )

                _connectionState.value = BleConnectionState.Connected(
                    name = gatt.device.name ?: "ESP32",
                    address = gatt.device.address
                )

            } else {

                Log.e(
                    "BLE",
                    "Descriptor write failed : $status"
                )

                _connectionState.value = BleConnectionState.Error("Failed to enable notifications (status $status)")
            }
        }

        // NOTE: this 3-arg onCharacteristicChanged(gatt, characteristic, value) overload is
        // API 33+. Since minSdk is 26, also override the legacy 2-arg version below and read
        // characteristic.value there; guard with Build.VERSION.SDK_INT so only one path fires
        // per device (the legacy callback is unused when the framework calls the new one).
        override fun onCharacteristicChanged(
            g: BluetoothGatt,
            characteristic: BluetoothGattCharacteristic,
            value: ByteArray
        ) {
            handleIncomingPacket(value)
        }

        @Deprecated("Kept for API < 33 devices; framework calls this instead of the 3-arg overload")
        @Suppress("DEPRECATION", "OVERRIDE_DEPRECATION")
        override fun onCharacteristicChanged(
            g: BluetoothGatt,
            characteristic: BluetoothGattCharacteristic
        ) {
            handleIncomingPacket(characteristic.value ?: return)
        }

        private fun handleIncomingPacket(value: ByteArray) {

            Log.d(
                "BLE",
                "Received ${value.size} bytes"
            )

            Log.d(
                "BLE",
                value.joinToString(" ") {
                    "%02X".format(it)
                }
            )

            val packet = ImuPacketParser.parse(value) ?: return

            Log.d(
                "BLE",
                packet.toString()
            )

            _imuPackets.tryEmit(packet)

            _linkStats.value =
                packetTracker.onPacket(packet)
                    .copy(rssi = _linkStats.value.rssi)
        }

        override fun onReadRemoteRssi(g: BluetoothGatt, rssi: Int, status: Int) {
            _linkStats.value = _linkStats.value.copy(rssi = rssi)
        }
    }

    private fun subscribeToNotifications(g: BluetoothGatt, characteristic: BluetoothGattCharacteristic) {
        Log.d("BLE", "Enabling notifications")
        g.setCharacteristicNotification(characteristic, true)
        val descriptor = characteristic.getDescriptor(BleConstants.CLIENT_CHARACTERISTIC_CONFIG_UUID)
        if (descriptor == null) return

        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.TIRAMISU) {
            // API 33+: writeDescriptor(descriptor, value) — descriptor.value is deprecated/unused here
            g.writeDescriptor(descriptor, BluetoothGattDescriptorValue.ENABLE_NOTIFICATION_VALUE)
        } else {
            // Pre-API 33: must set descriptor.value directly, then call the 1-arg writeDescriptor
            @Suppress("DEPRECATION")
            descriptor.value = BluetoothGattDescriptorValue.ENABLE_NOTIFICATION_VALUE
            @Suppress("DEPRECATION")
            g.writeDescriptor(descriptor)
        }
    }

    /** Poll RSSI periodically (e.g. every 2s from a ViewModel) to keep the signal bar fresh. */
    fun refreshRssi() {
        gatt?.readRemoteRssi()
    }
}

/**
 * BluetoothGatt.writeDescriptor(descriptor, value) is the API-33+ signature;
 * on older APIs you'd instead set descriptor.value and call writeDescriptor(descriptor).
 * This small shim keeps BleManager readable — swap the branch based on Build.VERSION.SDK_INT
 * in a real build, or use one code path if your minSdk allows it.
 */
private object BluetoothGattDescriptorValue {
    val ENABLE_NOTIFICATION_VALUE: ByteArray =
        android.bluetooth.BluetoothGattDescriptor.ENABLE_NOTIFICATION_VALUE
}
