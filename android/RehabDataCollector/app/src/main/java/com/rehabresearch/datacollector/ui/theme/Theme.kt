package com.rehabresearch.datacollector.ui.theme

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable

private val LightColors = lightColorScheme(
    primary = ClinicalBlue,
    secondary = ClinicalTeal,
    background = BackgroundGray,
    surface = SurfaceWhite,
    error = AlertRed,
    onPrimary = SurfaceWhite,
    onBackground = TextPrimary,
    onSurface = TextPrimary
)

private val DarkColors = darkColorScheme(
    primary = ClinicalBlue,
    secondary = ClinicalTeal,
    background = Color2(),
    surface = Color3(),
    error = AlertRed
)

// Small helpers kept private; dark mode isn't the priority for a clinic-floor tool
// but we don't want a broken theme if the device is set to dark.
private fun Color2() = androidx.compose.ui.graphics.Color(0xFF11161C)
private fun Color3() = androidx.compose.ui.graphics.Color(0xFF1B222B)

@Composable
fun RehabDataCollectorTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    content: @Composable () -> Unit
) {
    val colors = if (darkTheme) DarkColors else LightColors
    MaterialTheme(
        colorScheme = colors,
        typography = AppTypography,
        content = content
    )
}
