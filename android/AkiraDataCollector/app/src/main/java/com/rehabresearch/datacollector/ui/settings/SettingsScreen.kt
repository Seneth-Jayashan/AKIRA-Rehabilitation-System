package com.rehabresearch.datacollector.ui.settings

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.ListItem
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier

/**
 * This is a local-only research tool — there is no login and no cloud sync.
 * All patient data, sessions, and sensor readings live in the on-device Room
 * database; CSV exports land in app-private external storage
 * (Android/data/com.rehabresearch.datacollector/files/exports/) and are pulled
 * off the device manually (USB / file manager) for the AI training pipeline.
 *
 * Real settings to add here later:
 * - Target sample rate (50/100 Hz) sent to firmware via a config characteristic
 * - Data retention / local storage usage + clear-cache
 * - Export-all-sessions-as-zip convenience action
 */
@Composable
fun SettingsScreen() {
    Scaffold(topBar = { TopAppBar(title = { Text("Settings") }) }) { padding ->
        Column(Modifier.fillMaxSize().padding(padding)) {
            ListItem(headlineContent = { Text("App version") }, supportingContent = { Text("0.1.0-data-collector") })
            ListItem(headlineContent = { Text("Storage") }, supportingContent = { Text("Local only — no cloud sync") })
            ListItem(headlineContent = { Text("Sample rate") }, supportingContent = { Text("Set by firmware (default 100 Hz)") })
        }
    }
}
