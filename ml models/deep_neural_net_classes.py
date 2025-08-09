import numpy as np


def relu(number):
    return np.maximum(0, number)


def relu_gradient(number):
    return (number > 0).astype(float)


def swish(x):
    return x / (1 + np.exp(-x))  # x * sigmoid(x)


def swish_grad(x):
    sig = 1 / (1 + np.exp(-x))
    return sig + x * sig * (1 - sig)


def softplus(x):
    # For large x, softplus(x) ≈ x to avoid overflow
    return np.where(x > 20, x, np.log1p(np.exp(x)))


def softplus_gradient(x):
    # sigmoid function, clipped input to avoid overflow in exp
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


def batch_data(X, y, batch_size, shuffle=False, drop_last=False):
    """
    Splits features (X) and labels (y) into batches.

    Parameters:
        X (np.ndarray): Feature data (e.g., images)
        y (np.ndarray): Labels
        batch_size (int): Number of samples per batch
        shuffle (bool): Whether to shuffle before batching
        drop_last (bool): Whether to drop last batch if incomplete

    Returns:
        Generator yielding (X_batch, y_batch)
    """

    if shuffle:
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]

    for i in range(0, len(X), batch_size):
        if drop_last and i + batch_size > len(X):
            break
        yield X[i:i + batch_size], y[i:i + batch_size]


def softmax(logits):
    logits_shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_scores = np.exp(logits_shifted)
    softmax = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return softmax


class HiddenLayer:
    def __init__(self, num_inputs, num_layers, activation_function):
        self.input_weights = (2 * np.random.random((num_inputs, num_layers)) - 1) * 0.1
        self.input_biases = (2 * np.random.random((1, num_layers)) - 1) * 0.1
        self.activation_function = activation_function

    def predict(self, x):
        self.z1 = np.dot(x, self.input_weights) + self.input_biases

        if self.activation_function == "relu":
            self.h1 = relu(self.z1)
        elif self.activation_function == "swish":
            self.h1 = swish(self.z1)
        else:
            self.h1 = softplus(self.z1)

        return self.h1

    def back_propagate(self, x, delta_next, weights_next):
        if self.activation_function == "relu":
            activation_grad = relu_gradient(self.z1)
        elif self.activation_function == "swish":
            activation_grad = swish_grad(self.z1)
        else:
            activation_grad = softplus_gradient(self.z1)

        delta_hidden = np.dot(delta_next, weights_next.T) * activation_grad
        w_grads = np.dot(x.T, delta_hidden)  # shape: (input_units, hidden_units)
        b_grads = np.sum(delta_hidden, axis=0, keepdims=True)  # shape: (1, hidden_units)

        return w_grads, b_grads, delta_hidden


class OutputLayer:
    def __init__(self, num_inputs, num_outputs, num_layers, activation_function):
        self.input_weights = (2 * np.random.random((num_inputs, num_layers)) - 1) * 0.1
        self.input_biases = (2 * np.random.random((1, num_layers)) - 1) * 0.1
        self.output_weights = (2 * np.random.random((num_layers, num_outputs)) - 1) * 0.1
        self.output_biases = (2 * np.random.random((1, num_outputs)) - 1) * 0.1
        self.activation_function = activation_function

    def predict(self, x):
        z1 = np.dot(x, self.input_weights) + self.input_biases

        if self.activation_function == "relu":
            h1 = relu(z1)
        elif self.activation_function == "swish":
            h1 = swish(z1)
        else:
            h1 = softplus(z1)

        logits = np.dot(h1, self.output_weights) + self.output_biases
        return softmax(logits)

    def back_propagate(self, x, y):
        z1 = np.dot(x, self.input_weights) + self.input_biases  # (batch_size, hidden_units)

        if self.activation_function == "relu":
            hidden_out = relu(z1)
            hidden_grad = relu_gradient(z1)
        elif self.activation_function == "swish":
            hidden_out = swish(z1)
            hidden_grad = swish_grad(z1)
        else:
            hidden_out = softplus(z1)
            hidden_grad = softplus_gradient(z1)

        probs = self.predict(x)  # (batch_size, output_units)

        # Output layer error
        delta_output = probs - y  # (batch_size, output_units)

        # Gradient for output weights and biases
        w2_grads = np.dot(hidden_out.T, delta_output)  # (hidden_units, output_units)
        b2_grads = np.sum(delta_output, axis=0, keepdims=True)  # (1, output_units)

        # Error propagated to hidden layer
        delta_hidden = np.dot(delta_output, self.output_weights.T) * hidden_grad  # (batch_size, hidden_units)

        # Gradient for input weights and biases
        w1_grads = np.dot(x.T, delta_hidden)  # (input_units, hidden_units)
        b1_grads = np.sum(delta_hidden, axis=0, keepdims=True)  # (1, hidden_units)

        return w1_grads, w2_grads, b1_grads, b2_grads, delta_hidden


