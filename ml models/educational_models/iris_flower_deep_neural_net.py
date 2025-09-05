"""
=====================================================
 Iris Dataset Neural Network Example
=====================================================

This script demonstrates training a simple feedforward neural network
from scratch (using the CPU framework) on the Iris dataset.

It includes:
 - Data loading and normalization
 - One-hot encoding of target labels
 - Training a single hidden layer network with backpropagation
 - Evaluation of accuracy on test data

Author: Dennis Alacahanli
Purpose: This is an early implementation of training on the iris dataset made for learning purposes.
This model uses the deep neural network framework.
"""

import numpy as np
import pandas as pd
import math
from ml_frameworks import deep_neural_net_framework_cpu as dnnc

# ----------------- Random Seed ----------------- #
np.random.seed(42)

# ----------------- Load and Normalize Data ----------------- #
data = pd.read_csv("../../datasets/Iris.csv")
df_norm = data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].apply(
    lambda x: (x - x.min()) / (x.max() - x.min())  # normalize to [0, 1]
)

# Encode target species as 0, 1, 2
target = data[['Species']].replace(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], [0, 1, 2])

# Combine features and targets
df = pd.concat([df_norm, target], axis=1)

# ----------------- Train/Test Split ----------------- #
train_test_per = 90 / 100.0
df['train'] = np.random.rand(len(df)) < train_test_per  # boolean mask for training

# Separate train and test sets
train = df[df.train == 1].drop('train', axis=1).sample(frac=1)  # shuffle training data
test = df[df.train == 0].drop('train', axis=1)

# Input features
X = train.values[:, :4]

# One-hot encoding for target labels
targets = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
Y = np.array([targets[int(x)] for x in train.values[:, 4:5]])

num_inputs = len(X[0])
num_outputs = len(Y[0])
print(num_inputs)
print(num_outputs)

# ----------------- Hyperparameters ----------------- #
lr = 0.005

# ----------------- Neural Network Layers ----------------- #
hidden_layer = dnnc.HiddenLayer(num_inputs=num_inputs, num_layers=50, activation_function="relu")
output_layer = dnnc.OutputLayer(num_inputs=50, num_outputs=num_outputs, num_layers=20, activation_function="softplus")

# ----------------- Prediction Function ----------------- #
def predict(x):
    return output_layer.predict(hidden_layer.predict(x))

# ----------------- Cross-Entropy Loss ----------------- #
def cross_entropy(yi, predicted):
    val = 0
    predicted = predicted.flatten()
    for i in zip(yi, predicted):
        val -= i[0] * math.log(i[1] + 1e-15)  # small epsilon to avoid log(0)
    return val

# ----------------- Accuracy Function ----------------- #
def accuracy(x, y):
    correct = 0
    for xi, yi in zip(x, y):
        pred = predict(xi)
        if np.argmax(pred) == np.argmax(yi):
            correct += 1
    return correct / len(X)

# ----------------- Convert Prediction to One-Hot ----------------- #
def argmax_one_hot(prediction):
    max_val = max(prediction[0])
    return [1.0 if i == max_val else 0.0 for i in prediction[0]]

# ----------------- Training Function ----------------- #
def train(epochs, x, y):
    for epoch in range(epochs):
        # Forward pass through hidden and output layers
        hidden_out = hidden_layer.predict(x)

        # Backpropagation
        output_w1_grads, output_w2_grads, output_b1_grads, output_b2_grads, output_delta_hidden = \
            output_layer.back_propagate(hidden_out, y)
        hidden_w_grads, hidden_b_grads, _ = hidden_layer.back_propagate(x, output_delta_hidden, output_layer.input_weights)

        # Update weights and biases
        output_layer.input_weights -= lr * output_w1_grads
        output_layer.output_weights -= lr * output_w2_grads
        output_layer.input_biases -= lr * output_b1_grads
        output_layer.output_biases -= lr * output_b2_grads
        hidden_layer.input_weights -= lr * hidden_w_grads
        hidden_layer.input_biases -= lr * hidden_b_grads

        # Logging every 100 epochs
        if epoch % 100 == 0:
            acc = accuracy(x, y)
            loss = np.mean([cross_entropy(y[i], predict(x[i:i+1])) for i in range(len(x))])
            print(f"Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {acc:.2%}")

# ----------------- Train the Model ----------------- #
train(900, X, Y)

# ----------------- Test Model ----------------- #
test_x = test.values[:, :4]
test_y = np.array([targets[int(x)] for x in test.values[:, 4:5]])
corrects = 0

print("comparing predictions and test results:")
for i in range(len(test_x)):
    pred = argmax_one_hot(predict(test_x[i]))
    result = np.array_equal(pred, test_y[i])
    print(pred, test_y[i])
    if result:
        corrects += 1

print(f'test accuracy: {corrects/len(test_x)*100:.2f}%')
