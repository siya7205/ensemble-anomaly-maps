import pandas as pd
import matplotlib.pyplot as plt
import os

# Define the path to the hybrid scores CSV file
run_dir = 'outputs/run-traj-20250827-015400'
hybrid_scores_path = os.path.join(run_dir, 'deep', 'hybrid_scores.csv')

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(hybrid_scores_path)

# Check the columns in the DataFrame (useful to confirm the correct column name)
print(df.columns)

# Extract the B-factor (anomaly score) for each frame
b_factor = df['A_hybrid'].values  

# Plot the B-factor progression over the frames
plt.plot(b_factor)
plt.title('B-factor (Anomaly) Progression Across Frames')
plt.xlabel('Frame')
plt.ylabel('Anomaly Score (B-factor)')
plt.grid(True)
plt.show()
