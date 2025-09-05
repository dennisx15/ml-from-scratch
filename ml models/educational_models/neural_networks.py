"""
==========================================================
Simple Non-Linear Curve Fitting Neural Network
==========================================================

This script demonstrates a very basic neural network implementation
to fit a simple quadratic function (y = x^2 + 2) over a small input range.

Key Features:
- Single hidden neuron per mini-network (two sub-networks combined)
- Non-linear activation function: softplus
- Manual computation of derivatives for gradient descent
- Training loop with full-batch gradient updates
- Demonstrates early exploration of neural network concepts

Author: Dennis Alacahanli
Purpose: My earliest implementation of a single neuron. The beginning of non-linear predictions
"""

import math

# ----------------- Dataset ----------------- #
x = [0, 0.5, 1.0, 1.5, 2.0]   # Input values
y = [xi**2 + 2 for xi in x]   # Target values (quadratic function)

# ----------------- Model Parameters ----------------- #
lr = 0.01                      # Learning rate
w1, w2, w3, w4 = 1, 1, 1, 1   # Initial weights
b1, b2, b3 = 0, 0, 0          # Initial biases

# ----------------- Activation Function ----------------- #
def softplus(number):
    """
    Softplus activation function: log(1 + e^x)
    """
    return math.log(1 + math.e**number)

# ----------------- Mini Neural Networks ----------------- #
def neural_network1(number):
    """
    First mini-network: single hidden neuron with softplus activation.
    """
    hidden_layer = w1 * number + b1
    return softplus(hidden_layer) * w2

def neural_network2(number):
    """
    Second mini-network: single hidden neuron with softplus activation.
    """
    hidden_layer = w3 * number + b2
    return softplus(hidden_layer) * w4

# ----------------- Prediction ----------------- #
def predict(number):
    """
    Overall network prediction as the sum of both mini-networks plus output bias.
    """
    return neural_network1(number) + neural_network2(number) + b3

# ----------------- Loss Function ----------------- #
def sum_of_squared_residuals(inputs, answers):
    """
    Compute total squared error across all inputs.
    """
    error = 0
    for i in range(len(inputs)):
        error += (answers[i] - predict(inputs[i])) ** 2
    return error

# ----------------- Derivatives ----------------- #
def derivative_with_respect_to_b3(xi, yi):
    """
    Base derivative of loss with respect to output bias b3.
    """
    return -2*(yi - neural_network1(xi) - neural_network2(xi) - b3)

def derivative_with_respect_to_w2(xi, yi):
    hidden = w1 * xi + b1
    return derivative_with_respect_to_b3(xi, yi) * softplus(hidden)

def derivative_with_respect_to_w4(xi, yi):
    hidden = w3 * xi + b2
    return derivative_with_respect_to_b3(xi, yi) * softplus(hidden)

def derivative_with_respect_to_w1(xi, yi):
    return -2 * (yi - neural_network1(xi) - neural_network2(xi) - b3) * w2 * (math.e ** (w1 * xi + b1)) / (
            1 + math.e**(w1 * xi + b1)) * xi

def derivative_with_respect_to_w3(xi, yi):
    return -2 * (yi - neural_network1(xi) - neural_network2(xi) - b3) * w4 * (math.e ** (w3 * xi + b2)) / (
                1 + math.e ** (w3 * xi + b2)) * xi

def derivative_with_respect_to_b1(xi, yi):
    return -2 * (yi - neural_network1(xi) - neural_network2(xi) - b3) * w2 * (math.e ** (w1 * xi + b1)) / (
            1 + math.e**(w1 * xi + b1))

def derivative_with_respect_to_b2(xi, yi):
    return -2 * (yi - neural_network1(xi) - neural_network2(xi) - b3) * w4 * (math.e ** (w3 * xi + b2)) / (
                1 + math.e ** (w3 * xi + b2))

# ----------------- Training Loop ----------------- #
def train(epochs, x, y):
    """
    Train the network using gradient descent for the specified number of epochs.
    """
    global w1, w2, w3, w4, b1, b2, b3
    for epoch in range(epochs):
        # Initialize gradients
        dw1 = dw2 = dw3 = dw4 = db1 = db2 = db3 = 0

        # Compute gradients for all samples
        for xi, yi in zip(x, y):
            dw1 += derivative_with_respect_to_w1(xi, yi)
            dw2 += derivative_with_respect_to_w2(xi, yi)
            dw3 += derivative_with_respect_to_w3(xi, yi)
            dw4 += derivative_with_respect_to_w4(xi, yi)
            db1 += derivative_with_respect_to_b1(xi, yi)
            db2 += derivative_with_respect_to_b2(xi, yi)
            db3 += derivative_with_respect_to_b3(xi, yi)

        # Average gradients and update parameters
        n = len(x)
        w1 -= lr * dw1 / n
        w2 -= lr * dw2 / n
        w3 -= lr * dw3 / n
        w4 -= lr * dw4 / n
        b1 -= lr * db1 / n
        b2 -= lr * db2 / n
        b3 -= lr * db3 / n

        # Print loss every 100 epochs
        if epoch % 100 == 0:
            print(f"Epoch {epoch} | Loss: {sum_of_squared_residuals(x, y):.4f}")

# ----------------- Train the Model ----------------- #
train(200000, x, y)
