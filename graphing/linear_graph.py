import numpy as np
import matplotlib.pyplot as plt

# Heights in cm
heights = np.array([150, 155, 160, 165, 170, 175], dtype=float)

# Perfect linear relation: weight = 2.5 * height - 100
weights = 2.5 * heights - 100

# Print dataset
for h, w in zip(heights, weights):
    print(f"Height: {h} cm, Weight: {w} kg")

# Optional: plot the dataset
plt.scatter(heights, weights, color='blue', label='Data points')
plt.plot(heights, weights, color='red', label='Perfect linear fit')
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.title('Perfectly Linear Dataset: Height vs Weight')
plt.legend()
plt.grid(True)
plt.show()
