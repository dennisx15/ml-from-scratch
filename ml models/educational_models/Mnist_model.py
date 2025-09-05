"""
=====================================================
 MNIST Deep Neural Network from Scratch
=====================================================

This project implements a fully connected deep neural network
to classify handwritten digits from the MNIST dataset.
All layers, activations, forward and backward passes are
implemented from scratch using NumPy.

It includes:
 - Loading and preprocessing the MNIST dataset
 - Definition of hidden layers and output layer
 - Forward pass function for predictions
 - Cross-entropy loss function
 - Accuracy calculation
 - Batch gradient descent training loop
 - Backpropagation for weight updates
 - Evaluation on test set

Author: Dennis Alacahanli
Purpose: Learning and demonstrating a multi-layer neural network
from scratch without using high-level ML libraries.
"""

from ml_frameworks import deep_neural_net_framework_cpu as dnnc
from datasets.DataLoaders import mnist_digits
import numpy as np

# --------------------------
# Hyperparameters & Settings
# --------------------------
corrects = 0               # Counter for correct predictions on test set
num_classes = 10           # Number of output classes (digits 0-9)
batch_size = 20            # Number of samples per batch
epochs = 30                # Number of training epochs
lr = 0.04                  # Learning rate
np.random.seed(42)         # Seed for reproducibility

# --------------------------
# Prediction & Evaluation Functions
# --------------------------
def predict(x):
    """Forward pass through all layers to get output prediction."""
    vals1 = hidden_layer.predict(x)
    vals2 = hidden_layer2.predict(vals1)
    vals3 = hidden_layer3.predict(vals2)
    return output_layer.predict(vals3)


def cross_entropy(y_true, y_pred):
    """Compute cross-entropy loss between true and predicted labels."""
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)  # Prevent log(0)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))


def accuracy(x_batches, y_batches):
    """Compute accuracy over batches."""
    correct = 0
    total = 0
    for batch_x, batch_y in zip(x_batches, y_batches):
        for xi, yi in zip(batch_x, batch_y):
            pred = predict(xi)
            if np.argmax(pred) == np.argmax(yi):
                correct += 1
            total += 1
    return correct / total


def argmax_one_hot(prediction):
    """Convert raw predictions to one-hot encoding of the max value."""
    max_val = max(prediction[0])
    return [1.0 if i == max_val else 0.0 for i in prediction[0]]

# --------------------------
# Filepaths for MNIST dataset
# --------------------------
training_images_filepath = 'C:\\Users\\denni\\PycharmProjects\\ml-from-scratch\\datasets\\Mnist\\train-images-idx3-ubyte\\train-images-idx3-ubyte'
training_labels_filepath = 'C:\\Users\\denni\\PycharmProjects\\ml-from-scratch\\datasets\\Mnist\\train-labels-idx1-ubyte\\train-labels-idx1-ubyte'
test_images_filepath = 'C:\\Users\\denni\\PycharmProjects\\ml-from-scratch\\datasets\\Mnist\\t10k-images-idx3-ubyte\\t10k-images-idx3-ubyte'
test_labels_filepath = 'C:\\Users\\denni\\PycharmProjects\\ml-from-scratch\\datasets\\Mnist\\t10k-labels-idx1-ubyte\\t10k-labels-idx1-ubyte'

# --------------------------
# Load MNIST dataset
# --------------------------
mnist = MnistDigits.MnistDataloader(training_images_filepath, training_labels_filepath,
                        test_images_filepath, test_labels_filepath)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize and reshape images
x_train = x_train.reshape(x_train.shape[0], -1).astype(np.float32) / 255.0
x_test = x_test.reshape(x_test.shape[0], -1).astype(np.float32) / 255.0

# One-hot encode labels
y_train = np.eye(num_classes)[y_train]
y_test = np.eye(num_classes)[y_test]

# --------------------------
# Prepare data in batches
# --------------------------
X_batches = []
y_batches = []
for X_batch, y_batch in dnnc.batch_data(x_train, y_train, batch_size=20, shuffle=True, drop_last=False):
    X_batches.append(X_batch)
    y_batches.append(y_batch)
X_batches = np.array(X_batches)

# --------------------------
# Define network dimensions
# --------------------------
num_inputs = len(X_batches[0][0])
num_outputs = len(y_batches[0][0])

# --------------------------
# Initialize layers
# --------------------------
hidden_layer = dnnc.HiddenLayer(num_inputs=num_inputs, num_layers=140, activation_function="swish")
hidden_layer2 = dnnc.HiddenLayer(num_inputs=140, num_layers=112, activation_function="swish")
hidden_layer3 = dnnc.HiddenLayer(num_inputs=112, num_layers=84, activation_function="swish")
output_layer = dnnc.OutputLayer(num_inputs=84, num_outputs=num_outputs, num_layers=56, activation_function="swish")

# --------------------------
# Training Function
# --------------------------
def train(epochs, X_batches, y_batches):
    """Train the network using batch gradient descent."""
    for epoch in range(epochs):
        for X_batch, y_batch in zip(X_batches, y_batches):
            batch_size = X_batch.shape[0]

            # Forward pass
            hidden_out = hidden_layer.predict(X_batch)
            hidden2_out = hidden_layer2.predict(hidden_out)
            hidden3_out = hidden_layer3.predict(hidden2_out)

            # Backpropagation
            w1_grads, w2_grads, b1_grads, b2_grads, delta_hidden = output_layer.back_propagate(hidden3_out, y_batch)
            hidden3_w_grads, hidden3_b_grads, delta_hidden3 = hidden_layer3.back_propagate(hidden2_out, delta_hidden, output_layer.input_weights)
            hidden2_w_grads, hidden2_b_grads, delta_hidden2 = hidden_layer2.back_propagate(hidden_out, delta_hidden3, hidden_layer3.input_weights)
            hidden1_w_grads, hidden1_b_grads, _ = hidden_layer.back_propagate(X_batch, delta_hidden2, hidden_layer2.input_weights)

            # Update weights and biases
            output_layer.input_weights -= lr * w1_grads / batch_size
            output_layer.output_weights -= lr * w2_grads / batch_size
            output_layer.input_biases -= lr * b1_grads / batch_size
            output_layer.output_biases -= lr * b2_grads / batch_size

            hidden_layer3.input_weights -= lr * hidden3_w_grads / batch_size
            hidden_layer3.input_biases -= lr * hidden3_b_grads / batch_size

            hidden_layer2.input_weights -= lr * hidden2_w_grads / batch_size
            hidden_layer2.input_biases -= lr * hidden2_b_grads / batch_size

            hidden_layer.input_weights -= lr * hidden1_w_grads / batch_size
            hidden_layer.input_biases -= lr * hidden1_b_grads / batch_size

        # Compute loss and accuracy after each epoch
        acc = accuracy(X_batches, y_batches)
        loss = np.mean([cross_entropy(y, predict(x)) for x, y in zip(X_batches, y_batches)])
        print(f"Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {acc:.2%}")


# --------------------------
# Train the network
# --------------------------
train(epochs, X_batches, y_batches)

# --------------------------
# Evaluate on test set
# --------------------------
test_x = x_test
test_y = y_test
for i in range(len(test_x)):
    pred = argmax_one_hot(predict(test_x[i]))
    result = np.array_equal(pred, test_y[i])
    if result:
        corrects += 1

# Print final test accuracy
print(corrects / len(test_y))
