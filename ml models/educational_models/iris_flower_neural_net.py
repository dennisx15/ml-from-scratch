"""
==========================================================
Simple Neural Network for Iris Classification (From Scratch)
==========================================================

This script implements a single hidden layer neural network to classify
the Iris dataset, written entirely from scratch using only NumPy and pandas.

Key Features:
- Hidden layer with ReLU activation function
- Output layer with softmax activation for multi-class classification
- Manual implementation of forward pass, cross-entropy loss, and backpropagation
- Supports batch gradient computation for hidden and output layers
- Demonstrates hands-on neural network building for educational purposes

Author: Dennis Alacahanli
Purpose: This is an early implementation of training on the iris dataset made for learning purposes.
This model is vectorized per sample so it is faster than the earlier model.
"""

import pandas as pd
import numpy as np
import math

# ----------------- Seed and Dataset ----------------- #
np.random.seed(42)
data = pd.read_csv("../../datasets/Iris.csv")

# Normalize features to [0,1] range
df_norm = data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].apply(
    lambda x: (x - x.min()) / (x.max() - x.min())
)

# Encode target labels as integers
target = data[['Species']].replace(
    ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], [0, 1, 2]
)

# Combine normalized features and target labels
df = pd.concat([df_norm, target], axis=1)

# Split into training and testing sets
train_test_per = 90/100.0
df['train'] = np.random.rand(len(df)) < train_test_per
train = df[df.train == 1].drop('train', axis=1).sample(frac=1)
test = df[df.train == 0].drop('train', axis=1)

# Training input features and one-hot encoded targets
X = train.values[:, :4]
targets = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
Y = np.array([targets[int(x)] for x in train.values[:, 4:5]])

# ----------------- Model Parameters ----------------- #
num_inputs = len(X[0])
num_outputs = len(Y[0])
hidden_layer_neurons = 10

# Initialize weights and biases
w1 = 2 * np.random.random((num_inputs, hidden_layer_neurons)) - 1
w2 = 2 * np.random.random((hidden_layer_neurons, num_outputs)) - 1
b1 = np.zeros((1, hidden_layer_neurons))
b2 = np.zeros((1, num_outputs))
lr = 1 / len(X)  # learning rate scaled by training set size

# ----------------- Activation Functions ----------------- #
def relu(number):
    return np.maximum(0, number)

def relu_gradient(number):
    return (number > 0).astype(float)

def softplus(number):
    return np.log1p(np.exp(number))

def softplus_gradient(number):
    return (1 / (1 + math.e ** (-number))).astype(float)

def softmax(logits):
    logits_shifted = logits - np.max(logits, axis=1, keepdims=True)  # numerical stability
    exp_scores = np.exp(logits_shifted)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

# ----------------- Forward Pass ----------------- #
def predict(xi, w_1, w_2, b_1, b_2):
    hidden = relu(np.dot(xi, w_1) + b_1)  # Hidden layer activation
    logits = np.dot(hidden, w_2) + b_2     # Output logits
    return softmax(logits)                  # Softmax probabilities

# ----------------- Loss and Evaluation ----------------- #
def cross_entropy(yi, predicted):
    val = 0
    predicted = predicted.flatten()
    for i in zip(yi, predicted):
        val -= i[0] * math.log(i[1] + 1e-15)
    return val

def accuracy(x, y):
    correct = 0
    for xi, yi in zip(x, y):
        pred = predict(xi, w1, w2, b1, b2)
        if np.argmax(pred) == np.argmax(yi):
            correct += 1
    return correct / len(X)

def argmax_one_hot(prediction):
    max_val = max(prediction[0])
    return [1.0 if i == max_val else 0.0 for i in prediction[0]]

# ----------------- Backpropagation ----------------- #
def back_propagation_relu(x, y, w_1, w_2, b_1, b_2):
    z1 = np.dot(x, w_1) + b_1          # Hidden layer pre-activation
    relu_out = relu(z1)                # ReLU activation
    relu_grad = relu_gradient(z1)      # ReLU derivative
    probs = predict(x, w_1, w_2, b_1, b_2)

    # Gradients for output layer
    b2_grads = probs - y
    w2_grads = np.einsum('bi,bj->bij', relu_out, b2_grads)

    # Gradients for hidden layer
    delta_hidden = np.dot(b2_grads, w_2.T) * relu_grad
    w1_grads = np.einsum('bi,bj->bij', x, delta_hidden)
    b1_grads = delta_hidden

    return w1_grads, w2_grads, b1_grads, b2_grads

# ----------------- Training Loop ----------------- #
def train(epochs, x, y, w_1, w_2, b_1, b_2):
    for epoch in range(epochs):
        w1_grads, w2_grads, b1_grads, b2_grads = back_propagation_relu(x, y, w_1, w_2, b_1, b_2)

        # Update weights and biases
        w_1 += -lr * np.sum(w1_grads, axis=0)
        w_2 += -lr * np.sum(w2_grads, axis=0)
        b_1 += -lr * np.sum(b1_grads, axis=0)
        b_2 += -lr * np.sum(b2_grads, axis=0)

        # Logging
        if epoch % 100 == 0:
            acc = accuracy(X, Y)
            loss = np.mean([cross_entropy(x[i], predict(x[i], w_1, w_2, b_1, b_2)) for i in range(len(X))])
            print(f"Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {acc:.2%}")

    return w_1, w_2, b_1, b_2

# ----------------- Train Model ----------------- #
w1, w2, b1, b2 = train(1000, X, Y, w1, w2, b1, b2)

# ----------------- Evaluate on Test Set ----------------- #
test_x = test.values[:, :4]
test_y = np.array([targets[int(x)] for x in test.values[:, 4:5]])
corrects = 0
print('Comparing predictions with test data:')

for i in range(len(test_x)):
    pred = argmax_one_hot(predict(test_x[i], w1, w2, b1, b2))
    result = np.array_equal(pred, test_y[i])
    print(pred, test_y[i])
    if result:
        corrects += 1

print(f'Test accuracy: {corrects/len(test_x)*100:.2f}%')
