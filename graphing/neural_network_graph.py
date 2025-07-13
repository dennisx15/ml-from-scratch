import neural_networks
from neural_networks import *
import matplotlib.pyplot as plt


neural_networks.train(200000, x, y)


y_pred = [predict(i/10) for i in range(10)]
x_test = [i/10 for i in range(10)]  # 0.0 to 0.9
y_test = [xi**2 + 2 for xi in x_test]


plt.figure(figsize=(8, 5))
plt.plot(x_test, y_test, label='True Values (Target)', marker='o')
plt.plot(x_test, y_pred, label='Predictions', marker='x')
plt.title("Neural Network Predictions vs True Curve")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# If this code does not work, right click on ml_models, hover over mark directory as, and choose Sources Root

