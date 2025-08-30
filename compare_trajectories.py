import pandas as pd
import matplotlib.pyplot as plt

# Load data for multiple trajectories
df_1 = pd.read_csv('data/colored_frames/trajectory_1/hybrid_scores.csv')
df_2 = pd.read_csv('data/colored_frames/trajectory_2/hybrid_scores.csv')

# Plot B-factor progression for both trajectories
plt.figure(figsize=(10, 6))

# Plot for Trajectory 1
plt.plot(df_1['frame'], df_1['b_factor'], label='Trajectory 1', color='blue')

# Plot for Trajectory 2
plt.plot(df_2['frame'], df_2['b_factor'], label='Trajectory 2', color='red')

plt.title('Comparison of B-factor (Anomaly Score) Across Trajectories')
plt.xlabel('Frame')
plt.ylabel('B-factor (Anomaly Score)')
plt.legend()
plt.grid(True)
plt.show()
