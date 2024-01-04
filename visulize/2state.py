import matplotlib.pyplot as plt
import numpy as np

# Sample data for two stages
stage1_data = np.array([1, 2, 3, 4, 5])
stage2_data = np.array([5, 4, 3, 2, 1])

# Create subplots
plt.figure(figsize=(10, 5))

# Plot data for the first stage
plt.subplot(1, 2, 1)  # 1 row, 2 columns, subplot 1
plt.plot(stage1_data, label='Stage 1', marker='o')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Stage 1 Data')
plt.legend()

# Plot data for the second stage
plt.subplot(1, 2, 2)  # 1 row, 2 columns, subplot 2
plt.plot(stage2_data, label='Stage 2', marker='x')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Stage 2 Data')
plt.legend()

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the combined plot
plt.show()
