package com.rehabresearch.datacollector.ui.theme

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable

private val LightColors = lightColorScheme(
    primary = PrimaryBlue,
    secondary = SecondaryBlue,
    background = BackgroundLight,
    surface = SurfaceLight,
    error = AlertRed,
    onPrimary = SurfaceLight,
    onBackground = TextPrimary,
    onSurface = TextPrimary,
    surfaceVariant = androidx.compose.ui.graphics.Color(0xFFF1F5F9),
    onSurfaceVariant = androidx.compose.ui.graphics.Color(0xFF475569)
)

private val DarkColors = darkColorScheme(
    primary = SecondaryBlue,
    secondary = PrimaryBlue,
    background = androidx.compose.ui.graphics.Color(0xFF0F172A),
    surface = androidx.compose.ui.graphics.Color(0xFF1E293B),
    error = androidx.compose.ui.graphics.Color(0xFFF87171),
    onPrimary = androidx.compose.ui.graphics.Color(0xFFF8FAFC),
    onBackground = androidx.compose.ui.graphics.Color(0xFFF8FAFC),
    onSurface = androidx.compose.ui.graphics.Color(0xFFF8FAFC),
    surfaceVariant = androidx.compose.ui.graphics.Color(0xFF334155),
    onSurfaceVariant = androidx.compose.ui.graphics.Color(0xFFCBD5E1)
)

@Composable
fun AkiraDataCollectorTheme(
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
