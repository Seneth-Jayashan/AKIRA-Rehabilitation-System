package com.rehabresearch.datacollector.ui

import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.navigation.NavHostController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.navigation
import androidx.navigation.compose.rememberNavController
import com.rehabresearch.datacollector.ui.ble.BleDeviceScreen
import com.rehabresearch.datacollector.ui.dashboard.DashboardScreen
import com.rehabresearch.datacollector.ui.history.HistoryScreen
import com.rehabresearch.datacollector.ui.livesensor.LiveSensorScreen
import com.rehabresearch.datacollector.ui.patient.AddPatientScreen
import com.rehabresearch.datacollector.ui.patient.PatientListScreen
import com.rehabresearch.datacollector.ui.recording.RecordingSetupScreen
import com.rehabresearch.datacollector.ui.recording.RecordingViewModel
import com.rehabresearch.datacollector.ui.session.SessionSummaryScreen
import com.rehabresearch.datacollector.ui.settings.SettingsScreen

@Composable
fun RehabNavHost(navController: NavHostController = rememberNavController()) {
    NavHost(navController = navController, startDestination = NavRoutes.Dashboard.route) {

        composable(NavRoutes.Dashboard.route) {
            DashboardScreen(
                onNewSession = { navController.navigate(NavRoutes.PatientList.route) },
                onPatients = { navController.navigate(NavRoutes.PatientList.route) },
                onHistory = { navController.navigate(NavRoutes.History.route) },
                onSettings = { navController.navigate(NavRoutes.Settings.route) },
                onBleDevice = { navController.navigate(NavRoutes.BleDevice.route) }
            )
        }

        composable(NavRoutes.BleDevice.route) {
            BleDeviceScreen(onBack = { navController.popBackStack() })
        }

        composable(NavRoutes.PatientList.route) {
            PatientListScreen(
                onAddPatient = { navController.navigate(NavRoutes.AddPatient.route) },
                onSelectPatient = { patientId ->
                    navController.navigate(NavRoutes.RecordingGraph.build(patientId))
                }
            )
        }

        composable(NavRoutes.AddPatient.route) {
            AddPatientScreen(
                onSaved = { navController.popBackStack() },
                onBack = { navController.popBackStack() }
            )
        }

        navigation(
            route = NavRoutes.RecordingGraph.route,
            startDestination = NavRoutes.RecordingSetup.route
        ) {
            composable(NavRoutes.RecordingSetup.route) { backStackEntry ->
                val parentEntry = remember(backStackEntry) {
                    navController.getBackStackEntry(NavRoutes.RecordingGraph.route)
                }
                val viewModel: RecordingViewModel = hiltViewModel(parentEntry)
                RecordingSetupScreen(
                    viewModel = viewModel,
                    onBeginSession = { navController.navigate(NavRoutes.LiveSensor.route) },
                    onBack = { navController.popBackStack() }
                )
            }

            composable(NavRoutes.LiveSensor.route) { backStackEntry ->
                val parentEntry = remember(backStackEntry) {
                    navController.getBackStackEntry(NavRoutes.RecordingGraph.route)
                }
                val viewModel: RecordingViewModel = hiltViewModel(parentEntry)
                LiveSensorScreen(
                    viewModel = viewModel,
                    onSessionFinished = { sessionId ->
                        navController.navigate(NavRoutes.SessionSummary.build(sessionId)) {
                            popUpTo(NavRoutes.PatientList.route)
                        }
                    }
                )
            }
        }

        composable(NavRoutes.SessionSummary.route) {
            SessionSummaryScreen(onDone = {
                navController.navigate(NavRoutes.Dashboard.route) {
                    popUpTo(NavRoutes.Dashboard.route) { inclusive = true }
                }
            })
        }

        composable(NavRoutes.History.route) {
            HistoryScreen(onBack = { navController.popBackStack() })
        }

        composable(NavRoutes.Settings.route) {
            SettingsScreen()
        }
    }
}
