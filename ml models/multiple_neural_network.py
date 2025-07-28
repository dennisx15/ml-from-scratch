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
hidden_layer_neurons = 5


w1 = 2 * np.random.random((num_inputs, hidden_layer_neurons)) - 1
w2 = 2 * np.random.random((hidden_layer_neurons, num_outputs)) - 1

b1 = np.zeros((1, hidden_layer_neurons))
b2 = np.zeros((1, num_outputs))


def predict(xi, w_1, w_2, b_1, b_2):
    xi = xi.reshape(1, -1)

    # Hidden layer (ReLU)
    hidden = np.maximum(0, np.dot(xi, w_1) + b_1)

    # Output layer (raw logits)
    logits = np.dot(hidden, w_2) + b_2

    # Softmax with numerical stability
    logits_shifted = logits - np.max(logits)  # stability trick
    exp_scores = np.exp(logits_shifted)
    softmax = exp_scores / np.sum(exp_scores)

    return softmax.flatten()  # return as 1D array


def argmax_one_hot(prediction):
    max_val = max(prediction)
    return [1.0 if i == max_val else 0.0 for i in prediction]


def cross_entropy(yi, predicted):

    val = 0
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


def back_propagation(x, y, w_1, w_2, b_1, b_2):
    b2_grads = []
    w2_grads = []
    b1_grads = []
    w1_grads = []

    for i in range(len(x)):
        pred = predict(x[i], w_1, w_2, b_1, b_2)
        b2_grad = pred - y[i]

        hidden = np.dot(x[i], w_1) + b_1             # shape (1, hidden_layer_neurons)
        relu_out = np.maximum(0, hidden)
        relu_grad = (hidden > 0).astype(float)       # convert bool to float, shape (1, hidden_layer_neurons)

        delta_hidden = np.dot(w_2, b2_grad)          # shape (hidden_layer_neurons,)
        delta_hidden = delta_hidden.reshape(1, -1)   # reshape to (1, hidden_layer_neurons)

        delta_hidden *= relu_grad                     # element-wise multiply, shapes match now

        w2_grad = np.outer(relu_out.flatten(), b2_grad)
        b1_grad = delta_hidden.flatten()              # flatten back to 1D for convenience
        w1_grad = np.outer(x[i], b1_grad)

        b1_grads.append(b1_grad)
        w2_grads.append(w2_grad)
        b2_grads.append(b2_grad)
        w1_grads.append(w1_grad)

    b1_grads = np.array(b1_grads)
    b2_grads = np.array(b2_grads)
    w2_grads = np.array(w2_grads)
    w1_grads = np.array(w1_grads)

    return w1_grads, w2_grads, b1_grads, b2_grads


lr = 0.01 / len(X)


def train(epochs, x, y, w_1, w_2, b_1, b_2):
    for epoch in range(epochs):
        w1_grads, w2_grads, b1_grads, b2_grads = back_propagation(x, y, w_1, w_2, b_1, b_2)

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


print(corrects/len(test_y))

