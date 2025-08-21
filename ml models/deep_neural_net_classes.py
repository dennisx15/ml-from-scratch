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
    return (exp_scores / np.sum(exp_scores, axis=1, keepdims=True))


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


class NeuralNetwork:
    def __init__(self, hidden_layers, output_layer):
        self.hidden_layers = hidden_layers
        self.output_layer = output_layer


    def predict(self, x):
        out = x
        for hidden_layer in self.hidden_layers:
            out = hidden_layer.predict(out)
        return self.output_layer.predict(out)

    def accuracy(self, x, y):
        preds = self.predict(x)  # shape: (num_samples, num_classes)
        pred_classes = np.argmax(preds, axis=1)
        true_classes = np.argmax(y, axis=1)
        return np.mean(pred_classes == true_classes)

    def cross_entropy(self, y_true, y_pred):
        # Clip to prevent log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        # Mean over samples
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

    def train(self, x, y, epochs, lr, batch_size, shuffle=True, drop_last=False, seed=42):
        if seed is not None:
            np.random.seed(seed)

        for epoch in range(epochs):
            for X_batch, y_batch in batch_data(x, y, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last):

                batch_size = X_batch.shape[0]
                out = X_batch
                hidden_outs = [X_batch]
                for hidden_layer in self.hidden_layers:
                    out = hidden_layer.predict(out)
                    hidden_outs.append(out)

                probs = self.output_layer.predict(out)

                w1_grads, w2_grads, b1_grads, b2_grads, delta_hidden = \
                    self.output_layer.back_propagate(hidden_outs[-1], y_batch)

                # Update output layer weights


                next_delta = delta_hidden
                next_weights = self.output_layer.input_weights
                grads = []
                for i in reversed(range(len(self.hidden_layers))):

                    h = self.hidden_layers[i]
                    prev_out = hidden_outs[i]
                    w_grads, b_grads, delta_hidden = h.back_propagate(prev_out, next_delta, next_weights)


                    input_weight_update = lr * w_grads / batch_size
                    input_bias_update = lr * b_grads / batch_size
                    grads.append([input_weight_update, input_bias_update])

                    next_delta = delta_hidden
                    next_weights = h.input_weights

                grads.reverse()
                for i in range(len(self.hidden_layers)):
                    self.hidden_layers[i].input_weights -= grads[i][0]
                    self.hidden_layers[i].input_biases -= grads[i][1]

                self.output_layer.input_weights -= lr * w1_grads / batch_size
                self.output_layer.input_biases -= lr * b1_grads / batch_size
                self.output_layer.output_weights -= lr * w2_grads / batch_size
                self.output_layer.output_biases -= lr * b2_grads / batch_size

            if epoch % 5 == 0:
                preds = self.predict(np.vstack(x))
                acc = self.accuracy(np.vstack(x), np.vstack(y))
                loss = self.cross_entropy(np.vstack(y), preds)
                print(f"Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {acc:.2%}")


    # --- Saving ---
    def save_model(self, filename="model_weights.npz"):
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

        np.savez(filename, **params)
        print(f"Model saved to {filename}")

    # --- Loading ---
    def load_model(self,  filename="model_weights.npz"):
        params = np.load(filename)
        for i, layer in enumerate(self.hidden_layers):
            layer.input_weights = params[f"h{i}_weights"]
            layer.input_biases = params[f"h{i}_biases"]
        self.output_layer.input_weights = params["out_input_weights"]
        self.output_layer.input_biases = params["out_input_biases"]
        self.output_layer.output_weights = params["out_output_weights"]
        self.output_layer.output_biases = params["out_output_biases"]
        print(f"Model loaded from {filename}")



