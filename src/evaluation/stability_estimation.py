import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import os

def simulate_force_plate_data(fs=100, duration=10, condition='stable'):
    """
    Simulates Center of Pressure (CoP) data for a rehabilitation task.
    In a real scenario, this data comes from the AnkleImage dataset.
    """
    t = np.linspace(0, duration, fs * duration)
    
    # Base postural sway (low frequency human sway)
    base_sway_x = 0.5 * np.sin(2 * np.pi * 0.2 * t)
    base_sway_y = 0.8 * np.sin(2 * np.pi * 0.15 * t + 0.5)
    
    if condition == 'stable':
        noise_level = 0.2
        tremor = 0.0
    elif condition == 'unstable':
        noise_level = 1.5
        tremor = 0.8 * np.sin(2 * np.pi * 5.0 * t) # 5Hz tremor
    else:
        raise ValueError("Condition must be 'stable' or 'unstable'")
        
    cop_x = base_sway_x + tremor + np.random.normal(0, noise_level, len(t))
    cop_y = base_sway_y + tremor + np.random.normal(0, noise_level, len(t))
    
    # Force Z (Vertical ground reaction force)
    # E.g. body weight ~700N, varying slightly
    fz = 700 + 20 * np.sin(2 * np.pi * 1.0 * t) + np.random.normal(0, 5, len(t))
    
    df = pd.DataFrame({
        'timestamp': t,
        'cop_x': cop_x,
        'cop_y': cop_y,
        'fz': fz
    })
    return df

def calculate_stability_metrics(df, fs=100):
    """
    Extracts clinical stability indices from CoP data.
    """
    cop_x = df['cop_x'].values
    cop_y = df['cop_y'].values
    
    # 1. Mean Velocity (Total excursion / time)
    dx = np.diff(cop_x)
    dy = np.diff(cop_y)
    path_length = np.sum(np.sqrt(dx**2 + dy**2))
    duration = len(cop_x) / fs
    mean_velocity = path_length / duration
    
    # 2. RMS Displacement (from center)
    mean_x = np.mean(cop_x)
    mean_y = np.mean(cop_y)
    rms_x = np.sqrt(np.mean((cop_x - mean_x)**2))
    rms_y = np.sqrt(np.mean((cop_y - mean_y)**2))
    rms_displacement = np.sqrt(rms_x**2 + rms_y**2)
    
    # 3. 95% Confidence Ellipse Area (or Convex Hull as proxy for Sway Area)
    points = np.column_stack((cop_x, cop_y))
    hull = ConvexHull(points)
    sway_area = hull.volume # For 2D points, 'volume' is the area of the polygon
    
    # 4. Stability Index (Custom normalized score, 0-100, higher is better)
    # Inversely proportional to sway area and velocity
    # Penalize high velocity and large sway areas
    raw_instability = (mean_velocity * 0.5) + (sway_area * 0.5)
    
    # Sigmoid-like normalization to 0-100 bounds
    stability_index = max(0, 100 - (raw_instability * 2))
    
    return {
        'mean_velocity_mm_s': mean_velocity,
        'rms_displacement_mm': rms_displacement,
        'sway_area_mm2': sway_area,
        'stability_index': stability_index
    }

def visualize_sway(stable_df, unstable_df):
    os.makedirs('results', exist_ok=True)
    
    plt.figure(figsize=(10, 5))
    
    # Stable Plot
    plt.subplot(1, 2, 1)
    plt.plot(stable_df['cop_x'], stable_df['cop_y'], alpha=0.6, color='blue', linewidth=0.5)
    # Draw convex hull
    points = np.column_stack((stable_df['cop_x'], stable_df['cop_y']))
    hull = ConvexHull(points)
    for simplex in hull.simplices:
        plt.plot(points[simplex, 0], points[simplex, 1], 'k-', alpha=0.5)
    plt.title("Stable CoP Sway (Simulated)")
    plt.xlabel("CoP X (mm)")
    plt.ylabel("CoP Y (mm)")
    plt.grid(True)
    
    # Unstable Plot
    plt.subplot(1, 2, 2)
    plt.plot(unstable_df['cop_x'], unstable_df['cop_y'], alpha=0.6, color='red', linewidth=0.5)
    # Draw convex hull
    points = np.column_stack((unstable_df['cop_x'], unstable_df['cop_y']))
    hull = ConvexHull(points)
    for simplex in hull.simplices:
        plt.plot(points[simplex, 0], points[simplex, 1], 'k-', alpha=0.5)
    plt.title("Unstable CoP Sway (Simulated)")
    plt.xlabel("CoP X (mm)")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('results/cop_sway_comparison.png')
    plt.close()
    print("Saved results/cop_sway_comparison.png")

if __name__ == "__main__":
    print("Generating simulated AnkleImage Force Plate data...")
    stable_data = simulate_force_plate_data(condition='stable')
    unstable_data = simulate_force_plate_data(condition='unstable')
    
    print("\nCalculating Stability Metrics for STABLE patient:")
    stable_metrics = calculate_stability_metrics(stable_data)
    for k, v in stable_metrics.items():
        print(f"  {k}: {v:.2f}")
        
    print("\nCalculating Stability Metrics for UNSTABLE patient (Post-ORIF simulated):")
    unstable_metrics = calculate_stability_metrics(unstable_data)
    for k, v in unstable_metrics.items():
        print(f"  {k}: {v:.2f}")
        
    visualize_sway(stable_data, unstable_data)
    print("\nPhase 6 Stability Estimation core logic complete.")
