package com.rehabresearch.datacollector

import android.Manifest
import android.os.Build
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import com.rehabresearch.datacollector.ui.RehabNavHost
import com.rehabresearch.datacollector.ui.theme.RehabDataCollectorTheme
import dagger.hilt.android.AndroidEntryPoint

@AndroidEntryPoint
class MainActivity : ComponentActivity() {

    /**
     * BLE runtime permissions required before BleManager.startScan()/connect()
     * can be called. Android 12+ (API 31+) uses BLUETOOTH_SCAN/BLUETOOTH_CONNECT;
     * earlier versions need location permission because BLE scan results can be
     * used to infer location.
     */
    private val blePermissions: Array<String> =
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            arrayOf(Manifest.permission.BLUETOOTH_SCAN, Manifest.permission.BLUETOOTH_CONNECT)
        } else {
            arrayOf(Manifest.permission.ACCESS_FINE_LOCATION)
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContent {
            val permissionsGranted = remember { mutableStateOf(false) }

            val permissionLauncher = rememberLauncherForActivityResult(
                ActivityResultContracts.RequestMultiplePermissions()
            ) { results ->
                permissionsGranted.value = results.values.all { it }
            }

            LaunchedEffect(Unit) {
                permissionLauncher.launch(blePermissions)
            }

            RehabDataCollectorTheme {
                // The BLE screen itself checks connectionState; if permissions were denied,
                // BleManager.startScan() will simply fail silently on some OEMs or throw
                // SecurityException on stricter ones. A production build should surface
                // permissionsGranted.value here (e.g. a blocking rationale screen) before
                // ever reaching the BLE Device screen. Kept minimal here for clarity.
                RehabNavHost()
            }
        }
    }
}
