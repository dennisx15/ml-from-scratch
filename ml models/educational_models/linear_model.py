"""
=====================================================
 Linear Regression from Scratch
=====================================================

This is an introductory project demonstrating the basics of
gradient descent and linear regression implemented from scratch
without using any machine learning libraries.

It includes:
 - A small dataset of x and y values
 - Prediction function for a linear model
 - Calculation of sum of squared residuals (loss)
 - Derivatives of the loss with respect to slope and intercept
 - Gradient descent loop to optimize slope and intercept

Author: Dennis Alacahanli
Purpose: My very first machine learning model. A very simple 2-D linear regression model
"""

# ----------------- Dataset ----------------- #
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Input features
y = [i * 2 + 6 for i in x]            # Target values using the function y = 2x + 6

# ----------------- Hyperparameters ----------------- #
batch_size = 2       # Not used in this simple example
intercept = 0.0      # Initial guess for intercept (b)
slope = 1            # Initial guess for slope (m)
learning_rate = 0.001  # Learning rate for gradient descent

# ----------------- Prediction Function ----------------- #
def predict(value, slope, intercept):
    """
    Compute predicted y value using current slope and intercept.
    y = m*x + b
    """
    return intercept + value*slope

# ----------------- Loss Function ----------------- #
def sum_of_squared_residuals(x, y):
    """
    Compute the sum of squared differences between predicted and actual values.
    """
    result = 0
    for i in range(len(x)):
        result += (predict(x[i], slope, intercept) - y[i]) ** 2
    return result

# ----------------- Gradients ----------------- #
def sum_of_predict_derivatives_w_respect_to_slope(x, y):
    """
    Compute derivative of loss with respect to slope (m).
    """
    result = 0.0
    for i in range(len(x)):
        result += 2 * (predict(x[i], slope, intercept) - y[i]) * x[i]
    return result

def sum_of_predict_derivatives_w_respect_to_intercept(x, y):
    """
    Compute derivative of loss with respect to intercept (b).
    """
    result = 0.0
    for i in range(len(x)):
        result += 2 * (predict(x[i], slope, intercept) - y[i])
    return result

# ----------------- Gradient Descent ----------------- #
while sum_of_squared_residuals(x, y) > 1e-6:
    slope -= sum_of_predict_derivatives_w_respect_to_slope(x, y) * learning_rate
    intercept -= sum_of_predict_derivatives_w_respect_to_intercept(x, y) * learning_rate

# ----------------- Results ----------------- #
print(slope)      # Optimized slope
print(intercept)  # Optimized intercept
