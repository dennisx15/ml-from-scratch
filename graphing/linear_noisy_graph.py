import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)  # For reproducibility

# Heights
heights = np.array([150, 155, 160, 165, 170, 175], dtype=float)

# Add random noise to weights
noise = np.random.normal(loc=0, scale=5.0, size=heights.shape)
weights_noisy = 2.5 * heights - 100 + noise

# Print data
for h, w in zip(heights, weights_noisy):
    print(f"Height: {h} cm, Weight: {round(w, 1)} kg")

# Plot
plt.scatter(heights, weights_noisy, color='blue', label='Noisy data')

plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.title('Noisy Dataset: Height vs Weight')
plt.legend()
plt.grid(True)
plt.show()
