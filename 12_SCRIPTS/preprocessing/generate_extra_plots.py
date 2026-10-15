import pandas as pd
import matplotlib.pyplot as plt

def generate_extra_plots():
    print("Generating additional dashboard plots...")
    # Load some features
    try:
        df = pd.read_csv('06_FEATURES/segmented_movements.csv')
        feat_df = pd.read_csv('06_FEATURES/wrist_features.csv')
        rep_df = pd.read_csv('08_RESULTS/repetition_analysis.csv')
        
        trial = df['trial_id'].unique()[0]
        group = df[df['trial_id'] == trial]
        
        # 5. Acceleration magnitude
        plt.figure(figsize=(10,4))
        plt.plot(group['acc_mag'], label='Acc Mag')
        plt.title("Acceleration Magnitude")
        plt.savefig("07_VISUALIZATIONS/acceleration_magnitude.png")
        plt.close()
        
        # 6. Angular velocity magnitude
        plt.figure(figsize=(10,4))
        plt.plot(group['gyro_mag'], label='Gyro Mag (Angular Velocity)')
        plt.title("Angular Velocity Magnitude")
        plt.savefig("07_VISUALIZATIONS/angular_velocity_magnitude.png")
        plt.close()
        
        # 11. Feature distributions
        plt.figure(figsize=(10,4))
        plt.hist(feat_df['acc_mag_mean'], bins=20, alpha=0.5, label='Acc Mag Mean')
        plt.hist(feat_df['gyro_mag_mean'], bins=20, alpha=0.5, label='Gyro Mag Mean')
        plt.title("Feature Distributions")
        plt.legend()
        plt.savefig("07_VISUALIZATIONS/feature_distributions.png")
        plt.close()
        
        # 12. Movement quality indicators
        plt.figure(figsize=(10,4))
        plt.bar(rep_df['trial_id'].astype(str), rep_df['movement_consistency'])
        plt.title("Movement Quality (Consistency)")
        plt.xlabel("Trial")
        plt.ylabel("Std Dev of Peak Velocity")
        plt.savefig("07_VISUALIZATIONS/movement_quality.png")
        plt.close()
        
        print("Extra plots generated successfully.")
    except Exception as e:
        print("Error generating extra plots:", str(e))

if __name__ == '__main__':
    generate_extra_plots()
