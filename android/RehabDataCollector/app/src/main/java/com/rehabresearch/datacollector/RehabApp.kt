package com.rehabresearch.datacollector

import android.app.Application
import dagger.hilt.android.HiltAndroidApp

/**
 * Application entry point. @HiltAndroidApp triggers Hilt's code generation,
 * including a base class for the app that serves as the application-level
 * dependency container.
 */
@HiltAndroidApp
class RehabApp : Application()
