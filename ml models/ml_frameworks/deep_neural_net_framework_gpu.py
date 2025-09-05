"""
Simple GPU accelerated Neural Network Framework (from scratch with CuPy)
---------------------------------------------------------
Implements:
  - Hidden and output layers with backpropagation
  - ReLU, Swish, and Softplus activations
  - Mini-batch training with cross-entropy loss
  - Model save/load functionality

Notes:
    -Requires a computer with a dedicated Nvidia GPU.
    -Requires the installation of cuda toolkit.
Author: Dennis Alacahanli
Purpose: Educational project to understand neural networks at a low level.

"""

import cupy as cp  # GPU-accelerated NumPy-like library
from ml_frameworks import optimizers

cp.random.seed(42)
# ---------------- Activation Functions ---------------- #

def relu(number):
    # ReLU: outputs input if positive, else 0
    return cp.maximum(0, number)


def relu_gradient(number):
    # Gradient of ReLU: 1 for positive values, 0 otherwise
    return (number > 0).astype(cp.float32)


def swish(x):
    # Swish: smooth activation (x * sigmoid(x))
    return x / (1 + cp.exp(-x))


def swish_grad(x):
    # Derivative of Swish
    sig = 1 / (1 + cp.exp(-x))
    return sig + x * sig * (1 - sig)


def softplus(x):
    # Softplus: smooth version of ReLU
    # For large x, softplus(x) ≈ x (to avoid overflow in exp)
    return cp.where(x > 20, x, cp.log1p(cp.exp(x)))


def softplus_gradient(x):
    # Derivative of Softplus = sigmoid(x)
    # Input is clipped to avoid numerical overflow
    x = cp.clip(x, -500, 500)
    return 1 / (1 + cp.exp(-x))


# ---------------- Data Batching ---------------- #

def batch_data(X, y, batch_size, shuffle=False, drop_last=False):
    """
    Splits features (X) and labels (y) into batches (on GPU).
    """

    if shuffle:
        indices = cp.arange(len(X))
        cp.random.shuffle(indices)
        X = X[indices]
        y = y[indices]

    for i in range(0, len(X), batch_size):
        if drop_last and i + batch_size > len(X):
            break
        yield X[i:i + batch_size], y[i:i + batch_size]


def softmax(logits):
    # Stable softmax: subtract max to prevent overflow
    logits_shifted = logits - cp.max(logits, axis=1, keepdims=True)
    exp_scores = cp.exp(logits_shifted)
    return exp_scores / cp.sum(exp_scores, axis=1, keepdims=True)


# ---------------- Hidden Layer ---------------- #

class HiddenLayer:
    def __init__(self, num_inputs, num_layers, activation_function):
        # Initialize weights and biases (small random values)
        self.input_weights = (2 * cp.random.random((num_inputs, num_layers)) - 1) * 0.1
        self.input_biases = (2 * cp.random.random((1, num_layers)) - 1) * 0.1
        self.activation_function = activation_function

    def predict(self, x):
        # Forward pass: Linear transformation
        self.z1 = cp.dot(x, self.input_weights) + self.input_biases

        # Apply chosen activation
        if self.activation_function == "relu":
            self.h1 = relu(self.z1)
        elif self.activation_function == "swish":
            self.h1 = swish(self.z1)
        else:
            self.h1 = softplus(self.z1)

        return self.h1

    def back_propagate(self, x, delta_next, weights_next):
        # Select derivative of chosen activation
        if self.activation_function == "relu":
            activation_grad = relu_gradient(self.z1)
        elif self.activation_function == "swish":
            activation_grad = swish_grad(self.z1)
        else:
            activation_grad = softplus_gradient(self.z1)

        # Compute error for this layer (delta)
        delta_hidden = cp.dot(delta_next, weights_next.T) * activation_grad

        # Compute gradients for weights and biases
        w_grads = cp.dot(x.T, delta_hidden)
        b_grads = cp.sum(delta_hidden, axis=0, keepdims=True)

        return w_grads, b_grads, delta_hidden


# ---------------- Output Layer ---------------- #

class OutputLayer:
    def __init__(self, num_inputs, num_outputs, num_layers, activation_function):
        # First transformation (hidden)
        self.input_weights = (2 * cp.random.random((num_inputs, num_layers)) - 1) * 0.1
        self.input_biases = (2 * cp.random.random((1, num_layers)) - 1) * 0.1
        # Second transformation (to outputs)
        self.output_weights = (2 * cp.random.random((num_layers, num_outputs)) - 1) * 0.1
        self.output_biases = (2 * cp.random.random((1, num_outputs)) - 1) * 0.1
        self.activation_function = activation_function

    def predict(self, x):
        # Hidden layer transform
        z1 = cp.dot(x, self.input_weights) + self.input_biases

        # Apply activation
        if self.activation_function == "relu":
            h1 = relu(z1)
        elif self.activation_function == "swish":
            h1 = swish(z1)
        else:
            h1 = softplus(z1)

        # Output logits → softmax
        logits = cp.dot(h1, self.output_weights) + self.output_biases
        return softmax(logits)

    def back_propagate(self, x, y):
        # Hidden transform
        z1 = cp.dot(x, self.input_weights) + self.input_biases

        # Apply chosen activation + gradient
        if self.activation_function == "relu":
            hidden_out = relu(z1)
            hidden_grad = relu_gradient(z1)
        elif self.activation_function == "swish":
            hidden_out = swish(z1)
            hidden_grad = swish_grad(z1)
        else:
            hidden_out = softplus(z1)
            hidden_grad = softplus_gradient(z1)

        # Forward pass (probabilities)
        probs = self.predict(x)

        # Error at output (softmax - one-hot labels)
        delta_output = probs - y

        # Gradients for output weights and biases
        w2_grads = cp.dot(hidden_out.T, delta_output)
        b2_grads = cp.sum(delta_output, axis=0, keepdims=True)

        # Backprop error into hidden
        delta_hidden = cp.dot(delta_output, self.output_weights.T) * hidden_grad

        # Gradients for first layer (input weights/biases)
        w1_grads = cp.dot(x.T, delta_hidden)
        b1_grads = cp.sum(delta_hidden, axis=0, keepdims=True)

        return w1_grads, w2_grads, b1_grads, b2_grads, delta_hidden


# ---------------- Neural Network ---------------- #

class NeuralNetwork:
    def __init__(self, hidden_layers, output_layer, lr, optimizer):
        self.hidden_layers = hidden_layers
        self.output_layer = output_layer
        self.accuracies = []  # Track training accuracy over epochs
        self.grads = []
        self.lr = lr
        if optimizer == "SGD":
            self.optimizer = optimizers.SGD(lr=self.lr)
        elif optimizer == "MomentumSGD":
            self.optimizer = optimizers.MomentumSGD(lr=self.lr)
        else:
            raise Exception("Invalid optimizer")

    def predict(self, x):
        # Forward pass through all hidden layers
        out = x
        for hidden_layer in self.hidden_layers:
            out = hidden_layer.predict(out)
        # Then output layer
        return self.output_layer.predict(out)

    def accuracy(self, x, y):
        # Prediction → class index
        preds = self.predict(x)
        pred_classes = cp.argmax(preds, axis=1)
        true_classes = cp.argmax(y, axis=1)
        return cp.mean(pred_classes == true_classes).item()

    def cross_entropy(self, y_true, y_pred):
        # Cross-entropy loss
        y_pred = cp.clip(y_pred, 1e-15, 1 - 1e-15)  # avoid log(0)
        return -cp.mean(cp.sum(y_true * cp.log(y_pred), axis=1)).item()

    def train(self, x, y, epochs, batch_size, shuffle=True, drop_last=False, seed=42):
        # Seed for reproducibility
        if seed is not None:
            cp.random.seed(seed)

        for epoch in range(epochs):
            # Iterate over batches
            for X_batch, y_batch in batch_data(x, y, batch_size=batch_size,
                                               shuffle=shuffle, drop_last=drop_last):

                batch_size = X_batch.shape[0]
                out = X_batch
                hidden_outs = [X_batch]

                # Forward pass through hidden layers
                for hidden_layer in self.hidden_layers:
                    out = hidden_layer.predict(out)
                    hidden_outs.append(out)

                # Store grads for the current epoch
                self.grads = []

                # Backpropagation: compute gradients in output layer
                w1_grads, w2_grads, b1_grads, b2_grads, delta_hidden = \
                    self.output_layer.back_propagate(hidden_outs[-1], y_batch)

                # Backpropagate into hidden layers
                next_delta = delta_hidden
                next_weights = self.output_layer.input_weights


                for i in reversed(range(len(self.hidden_layers))):
                    h = self.hidden_layers[i]
                    prev_out = hidden_outs[i]
                    w_grads, b_grads, delta_hidden = h.back_propagate(prev_out, next_delta, next_weights)

                    self.grads.append([w_grads, b_grads])

                    next_delta = delta_hidden
                    next_weights = h.input_weights

                # Reverse updates so they match forward order
                self.grads.reverse()

                # Apply updates to hidden layers
                if type(self.optimizer) == optimizers.MomentumSGD:
                    for i in range(len(self.hidden_layers)):
                        self.hidden_layers[i].input_weights -= self.optimizer.update(f"h{i}_w", self.grads[i][0]) / batch_size
                        self.hidden_layers[i].input_biases -= self.optimizer.update(f"h{i}_b", self.grads[i][1]) / batch_size

                else:
                    for i in range(len(self.hidden_layers)):
                        self.hidden_layers[i].input_weights -= self.optimizer.update(self.grads[i][0]) / batch_size
                        self.hidden_layers[i].input_biases -= self.optimizer.update(self.grads[i][1]) / batch_size



                # Apply updates to output layer

                if type(self.optimizer) == optimizers.MomentumSGD:
                    self.output_layer.input_weights -= self.optimizer.update("out_w1", w1_grads) / batch_size
                    self.output_layer.input_biases -= self.optimizer.update("out_b1", b1_grads) / batch_size
                    self.output_layer.output_weights -= self.optimizer.update("out_w2", w2_grads) / batch_size
                    self.output_layer.output_biases -= self.optimizer.update("out_b2", b2_grads) / batch_size
                else:
                    self.output_layer.input_weights -= self.optimizer.update(w1_grads) / batch_size
                    self.output_layer.input_biases -= self.optimizer.update(b1_grads) / batch_size
                    self.output_layer.output_weights -= self.optimizer.update(w2_grads) / batch_size
                    self.output_layer.output_biases -= self.optimizer.update(b2_grads) / batch_size

            # Every 5 epochs → log loss & accuracy
            if epoch % 5 == 0:
                preds = self.predict(x)
                acc = self.accuracy(x, y)
                self.accuracies.append(acc)
                loss = self.cross_entropy(y, preds)
                print(f"Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {acc:.2%}")

    # --- Saving Model Parameters ---
    def save_model(self, filename="model_weights.npz", printing=True):
        params = {}
        # Save hidden layers
        for i, layer in enumerate(self.hidden_layers):
            params[f"h{i}_weights"] = layer.input_weights
            params[f"h{i}_biases"] = layer.input_biases
        # Save output layer
        params["out_input_weights"] = self.output_layer.input_weights
        params["out_input_biases"] = self.output_layer.input_biases
        params["out_output_weights"] = self.output_layer.output_weights
        params["out_output_biases"] = self.output_layer.output_biases

        cp.savez(filename, **params)
        if printing:
            print(f"Model saved to {filename}")

    # --- Loading Model Parameters ---
    def load_model(self, filename="model_weights.npz", printing=True):
        params = cp.load(filename)
        for i, layer in enumerate(self.hidden_layers):
            layer.input_weights = params[f"h{i}_weights"]
            layer.input_biases = params[f"h{i}_biases"]
        self.output_layer.input_weights = params["out_input_weights"]
        self.output_layer.input_biases = params["out_input_biases"]
        self.output_layer.output_weights = params["out_output_weights"]
        self.output_layer.output_biases = params["out_output_biases"]
        if printing:
            print(f"Model loaded from {filename}")
