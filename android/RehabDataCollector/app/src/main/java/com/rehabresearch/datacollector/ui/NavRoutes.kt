package com.rehabresearch.datacollector.ui

/**
 * Central place for every navigation destination in the app.
 * Keeping these as sealed routes avoids magic strings scattered across screens.
 */
sealed class NavRoutes(val route: String) {
    data object Dashboard : NavRoutes("dashboard")

    data object PatientList : NavRoutes("patients")
    data object AddPatient : NavRoutes("patients/add")

    data object BleDevice : NavRoutes("ble")

    // Setup and LiveSensor both live inside this nested graph and share ONE
    // RecordingViewModel instance (scoped to the graph's own backstack entry).
    // That's essential: setup picks the exercise/side/reps, and only once
    // recording actually starts does a sessionId get generated — LiveSensor
    // needs to see that same in-memory state, not receive it as a nav arg.
    data object RecordingGraph : NavRoutes("recording_graph/{patientId}") {
        fun build(patientId: String) = "recording_graph/$patientId"
    }
    data object RecordingSetup : NavRoutes("setup")
    data object LiveSensor : NavRoutes("live")

    data object SessionSummary : NavRoutes("session/summary/{sessionId}") {
        fun build(sessionId: String) = "session/summary/$sessionId"
    }

    data object History : NavRoutes("history")
    data object Settings : NavRoutes("settings")

    companion object {
        const val ARG_PATIENT_ID = "patientId"
        const val ARG_SESSION_ID = "sessionId"
    }
}
