"""
==========================================================
Simple Neural Network for Iris Classification (From Scratch)
==========================================================

This script demonstrates a single hidden layer neural network
implemented entirely from scratch (no high-level ML libraries)
to classify the Iris dataset.

Key Features:
- Single hidden layer with ReLU activation
- Output layer with softmax activation for multi-class classification
- Manual implementation of forward pass, loss calculation (cross-entropy),
  and backpropagation for weight and bias updates
- Demonstrates early hands-on exploration of neural networks concepts
- Uses only NumPy and pandas for computations

Author: Dennis Alacahanli
Purpose: This is an early implementation of training on the iris dataset made for learning purposes. It is slower than other models as it is not optimized
"""


import pandas as pd
import numpy as np
import math
# ------------------------
# Data Loading & Preprocessing
# ------------------------
# Read Iris dataset
data = pd.read_csv("../../datasets/Iris.csv")

# Normalize numeric features to [0, 1]
df_norm = data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

# Convert categorical labels to integers
target = data[['Species']].replace(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], [0, 1, 2])

# Combine normalized features with labels
df = pd.concat([df_norm, target], axis=1)

# Split into train and test sets (90% train)
train_test_per = 90/100.0
df['train'] = np.random.rand(len(df)) < train_test_per
train = df[df.train == 1].drop('train', axis=1).sample(frac=1)  # Shuffle training set
test = df[df.train == 0].drop('train', axis=1)

# Separate features and one-hot encode labels
X = train.values[:, :4]
targets = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
Y = np.array([targets[int(x)] for x in train.values[:, 4:5]])

# ------------------------
# Neural Network Initialization
# ------------------------
num_inputs = len(X[0])
num_outputs = len(Y[0])
hidden_layer_neurons = 10  # Number of neurons in hidden layer

# Random initialization of weights [-1, 1] and biases [0]
w1 = 2 * np.random.random((num_inputs, hidden_layer_neurons)) - 1
w2 = 2 * np.random.random((hidden_layer_neurons, num_outputs)) - 1
b1 = np.zeros((1, hidden_layer_neurons))
b2 = np.zeros((1, num_outputs))

# ------------------------
# Forward Pass Function
# ------------------------
def predict(xi, w_1, w_2, b_1, b_2):
    xi = xi.reshape(1, -1)  # Ensure input is 2D

    # Hidden layer with ReLU activation
    hidden = np.maximum(0, np.dot(xi, w_1) + b_1)

    # Output layer (logits)
    logits = np.dot(hidden, w_2) + b_2

    # Softmax for probabilities (with numerical stability)
    logits_shifted = logits - np.max(logits)
    exp_scores = np.exp(logits_shifted)
    softmax = exp_scores / np.sum(exp_scores)

    return softmax.flatten()  # Flatten to 1D array

# ------------------------
# Helper Functions
# ------------------------
def argmax_one_hot(prediction):
    max_val = max(prediction)
    return [1.0 if i == max_val else 0.0 for i in prediction]

def cross_entropy(yi, predicted):
    val = 0
    for i in zip(yi, predicted):
        val -= i[0] * math.log(i[1] + 1e-15)  # Clip for numerical stability
    return val

def accuracy(x, y):
    correct = 0
    for xi, yi in zip(x, y):
        pred = predict(xi, w1, w2, b1, b2)
        if np.argmax(pred) == np.argmax(yi):
            correct += 1
    return correct / len(X)

# ------------------------
# Manual Backpropagation
# ------------------------
def back_propagation(x, y, w_1, w_2, b_1, b_2):
    # Store gradients for each training sample
    b2_grads, w2_grads, b1_grads, w1_grads = [], [], [], []

    for i in range(len(x)):
        pred = predict(x[i], w_1, w_2, b_1, b_2)
        b2_grad = pred - y[i]  # Gradient at output

        # Hidden layer computations
        hidden = np.dot(x[i], w_1) + b_1
        relu_out = np.maximum(0, hidden)
        relu_grad = (hidden > 0).astype(float)

        delta_hidden = np.dot(w_2, b2_grad).reshape(1, -1) * relu_grad

        # Compute gradients
        w2_grad = np.outer(relu_out.flatten(), b2_grad)
        b1_grad = delta_hidden.flatten()
        w1_grad = np.outer(x[i], b1_grad)

        b1_grads.append(b1_grad)
        w2_grads.append(w2_grad)
        b2_grads.append(b2_grad)
        w1_grads.append(w1_grad)

    return np.array(w1_grads), np.array(w2_grads), np.array(b1_grads), np.array(b2_grads)

# ------------------------
# Training Loop
# ------------------------
lr = 0.01 / len(X)  # Learning rate adjusted by dataset size

def train(epochs, x, y, w_1, w_2, b_1, b_2):
    for epoch in range(epochs):
        w1_grads, w2_grads, b1_grads, b2_grads = back_propagation(x, y, w_1, w_2, b_1, b_2)

        # Update weights and biases using average gradients
        w_1 += -lr * np.sum(w1_grads, axis=0)
        w_2 += -lr * np.sum(w2_grads, axis=0)
        b_1 += -lr * np.sum(b1_grads, axis=0)
        b_2 += -lr * np.sum(b2_grads, axis=0)

        if epoch % 100 == 0:
            acc = accuracy(X, Y)
            loss = np.mean([cross_entropy(x[i], predict(x[i], w_1, w_2, b_1, b_2)) for i in range(len(X))])
            print(f"Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {acc:.2%}")
    return w_1, w_2, b_1, b_2

# Train the network
w1, w2, b1, b2 = train(10000, X, Y, w1, w2, b1, b2)

# ------------------------
# Testing & Evaluation
# ------------------------
test_x = test.values[:, :4]
test_y = np.array([targets[int(x)] for x in test.values[:, 4:5]])

corrects = 0
for i in range(len(test_x)):
    pred = argmax_one_hot(predict(test_x[i], w1, w2, b1, b2))
    result = np.array_equal(pred, test_y[i])
    print(pred, test_y[i])
    if result:
        corrects += 1

print(f'Test accuracy: {corrects/len(test_y):.2%}')

