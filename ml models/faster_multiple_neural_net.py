import pandas as pd
import numpy as np
import math


np.random.seed(42)

data = pd.read_csv("../datasets/Iris.csv")
df_norm = data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].apply(lambda x: (x - x.min()) / (x.max() - x.min()))


# The replace feature is removed in newer versions
target = data[['Species']].replace(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], [0, 1, 2])


df = pd.concat([df_norm, target], axis=1)

train_test_per = 90/100.0
df['train'] = np.random.rand(len(df)) < train_test_per

train = df[df.train == 1]
train = train.drop('train', axis=1).sample(frac=1)

test = df[df.train == 0]
test = test.drop('train', axis=1)

X = train.values[:, :4]

targets = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
Y = np.array([targets[int(x)] for x in train.values[:, 4:5]])


num_inputs = len(X[0])
num_outputs = len(Y[0])
hidden_layer_neurons = 10

w1 = 2 * np.random.random((num_inputs, hidden_layer_neurons)) - 1
w2 = 2 * np.random.random((hidden_layer_neurons, num_outputs)) - 1

b1 = np.zeros((1, hidden_layer_neurons))
b2 = np.zeros((1, num_outputs))

lr = 1 / len(X)


def relu(number):
    return np.maximum(0, number)


def relu_gradient(number):
    return (number > 0).astype(float)


def softplus(number):
    return np.log1p(np.exp(number))


def softplus_gradient(number):
    return (1 / (1 + math.e ** (-1 * number))).astype(float)


def softmax(logits):
    logits_shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_scores = np.exp(logits_shifted)
    softmax = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return softmax


def predict(xi, w_1, w_2, b_1, b_2):

    # Hidden layer (ReLU)
    hidden = np.maximum(0, np.dot(xi, w_1) + b_1)

    # Output layer (raw logits)
    logits = np.dot(hidden, w_2) + b_2

    return softmax(logits)


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


def back_propagation_relu(x, y, w_1, w_2, b_1, b_2):
    # Forward pass in batch
    z1 = np.dot(x, w_1) + b_1                # (batch_size, hidden_neurons)
    relu_out = relu(z1)             # ReLU activation
    relu_grad = relu_gradient(z1)    # ReLU derivative

    probs = predict(x, w_1, w_2, b_1, b_2)

    # Output layer gradient
    b2_grads = probs - y                     # (batch_size, num_outputs)
    w2_grads = np.einsum('bi,bj->bij', relu_out, b2_grads)  # (batch_size, hidden, output)

    # Hidden layer gradient
    delta_hidden = np.dot(b2_grads, w_2.T) * relu_grad      # (batch_size, hidden)
    w1_grads = np.einsum('bi,bj->bij', x, delta_hidden)     # (batch_size, input, hidden)

    b1_grads = delta_hidden                 # (batch_size, hidden)

    return w1_grads, w2_grads, b1_grads, b2_grads


def back_propagation_softplus(x, y, w_1, w_2, b_1, b_2):
    # Forward pass in batch
    z1 = np.dot(x, w_1) + b_1                # (batch_size, hidden_neurons)
    softplus_out = softplus(z1)             # ReLU activation
    softplus_grad = softplus_gradient(z1)    # ReLU derivative

    probs = predict(x, w_1, w_2, b_1, b_2)

    # Output layer gradient
    b2_grads = probs - y                     # (batch_size, num_outputs)
    w2_grads = np.einsum('bi,bj->bij', softplus_out, b2_grads)  # (batch_size, hidden, output)

    # Hidden layer gradient
    delta_hidden = np.dot(b2_grads, w_2.T) * softplus_grad      # (batch_size, hidden)
    w1_grads = np.einsum('bi,bj->bij', x, delta_hidden)     # (batch_size, input, hidden)

    b1_grads = delta_hidden                 # (batch_size, hidden)

    return w1_grads, w2_grads, b1_grads, b2_grads


def train(epochs, x, y, w_1, w_2, b_1, b_2):
    for epoch in range(epochs):
        w1_grads, w2_grads, b1_grads, b2_grads = back_propagation_relu(x, y, w_1, w_2, b_1, b_2)

        w1_update = -lr * np.sum(w1_grads, axis=0)
        w2_update = -lr * np.sum(w2_grads, axis=0)
        b1_update = -lr * np.sum(b1_grads, axis=0)
        b2_update = -lr * np.sum(b2_grads, axis=0)

        w_1 += w1_update
        w_2 += w2_update
        b_1 += b1_update
        b_2 += b2_update

        if epoch % 100 == 0:
            acc = accuracy(X, Y)
            loss = np.mean([cross_entropy(x[i], predict(x[i], w_1, w_2, b_1, b_2)) for i in range(len(X))])
            print(f"Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {acc:.2%}")
    return w_1, w_2, b_1, b_2


w1, w2, b1, b2 = train(10000, X, Y, w1, w2, b1, b2)
test_x = test.values[:, :4]
test_y = np.array([targets[int(x)] for x in test.values[:, 4:5]])

corrects = 0
for i in range(len(test_x)):
    pred = argmax_one_hot(predict(test_x[i], w1, w2, b1, b2))
    result = np.array_equal(pred, test_y[i])
    print(pred, test_y[i])
    if result:
        corrects += 1
#prediction = predict(X, w1, w2, b1, b2)
#print(prediction)

#print(prediction)
#print(np.shape(prediction))


print(corrects/len(test_y))

