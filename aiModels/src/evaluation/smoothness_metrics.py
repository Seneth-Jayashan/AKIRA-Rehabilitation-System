import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.signal import welch

def calculate_jerk(acc_signal, fs=100):
    """
    Computes Jerk (derivative of acceleration).
    """
    dt = 1.0 / fs
    jerk = np.diff(acc_signal) / dt
    return jerk

def calculate_smoothness(acc_signal, fs=100):
    """
    Calculates Kinematic Smoothness Metrics from an IMU acceleration signal.
    """
    # 1. Root Mean Square (RMS) of Jerk
    # A higher RMS Jerk indicates a less smooth, more tremorous movement.
    jerk = calculate_jerk(acc_signal, fs)
    rms_jerk = np.sqrt(np.mean(jerk**2))
    
    # 2. Spectral Arc Length (SPARC)
    # Measures the complexity of the signal's frequency spectrum.
    # Smoother movements have simpler spectra (concentrated at low frequencies).
    # Less smooth movements (tremors) have long, complex spectra.
    # Higher SPARC (closer to 0) = Smoother. Lower SPARC (more negative) = Less Smooth.
    freqs, psd = welch(acc_signal, fs=fs, nperseg=min(len(acc_signal), 256))
    
    # Normalize PSD
    psd_norm = psd / max(psd)
    
    # Calculate Arc Length of the spectrum curve
    df = freqs[1] - freqs[0]
    dp = np.diff(psd_norm)
    arc_length = np.sum(np.sqrt((df)**2 + dp**2))
    
    sparc = -arc_length
    
    return rms_jerk, sparc

def simulate_and_evaluate():
    """
    Simulates a smooth rehab movement (e.g., slow Ankle Dorsiflexion) 
    vs a rigid/spastic movement to demonstrate the smoothness metrics.
    """
    os.makedirs('aiModels/results', exist_ok=True)
    fs = 100
    t = np.linspace(0, 5, fs * 5)
    
    # Smooth movement (low frequency sine wave)
    smooth_acc = np.sin(2 * np.pi * 0.5 * t)
    
    # Spastic/Rigid movement (base movement + high frequency jitter)
    spastic_acc = smooth_acc + 0.3 * np.sin(2 * np.pi * 8.0 * t) + np.random.normal(0, 0.1, len(t))
    
    # Calculate metrics
    smooth_rms_jerk, smooth_sparc = calculate_smoothness(smooth_acc, fs)
    spastic_rms_jerk, spastic_sparc = calculate_smoothness(spastic_acc, fs)
    
    print("--- Kinematic Smoothness Metrics ---")
    print("\n[Healthy / Smooth Movement]")
    print(f"RMS Jerk: {smooth_rms_jerk:.2f} (lower is better)")
    print(f"SPARC:    {smooth_sparc:.2f} (closer to 0 is better)")
    
    print("\n[Post-ORIF / Spastic Movement]")
    print(f"RMS Jerk: {spastic_rms_jerk:.2f} (lower is better)")
    print(f"SPARC:    {spastic_sparc:.2f} (closer to 0 is better)")
    
    # Visualizing Jerk
    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 2, 1)
    plt.plot(t, smooth_acc, 'b')
    plt.title("Smooth Acceleration Profile")
    plt.ylabel("Acc (g)")
    
    plt.subplot(2, 2, 2)
    plt.plot(t[:-1], calculate_jerk(smooth_acc, fs), 'b')
    plt.title("Smooth Jerk Profile (Low Amplitude)")
    plt.ylabel("Jerk (g/s)")
    
    plt.subplot(2, 2, 3)
    plt.plot(t, spastic_acc, 'r')
    plt.title("Spastic Acceleration Profile")
    plt.ylabel("Acc (g)")
    plt.xlabel("Time (s)")
    
    plt.subplot(2, 2, 4)
    plt.plot(t[:-1], calculate_jerk(spastic_acc, fs), 'r')
    plt.title("Spastic Jerk Profile (High Amplitude)")
    plt.ylabel("Jerk (g/s)")
    plt.xlabel("Time (s)")
    
    plt.tight_layout()
    plt.savefig('aiModels/results/smoothness_comparison.png')
    plt.close()
    print("\nSaved chart to aiModels/results/smoothness_comparison.png")

if __name__ == "__main__":
    simulate_and_evaluate()
