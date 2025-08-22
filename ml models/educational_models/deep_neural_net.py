from practical_models import deep_neural_net_classes as dnnc
import pandas as pd
import numpy as np
import math


np.random.seed(42)

data = pd.read_csv("../../datasets/Iris.csv")
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

print(num_inputs)
print(num_outputs)
lr = 0.005

hidden_layer = dnnc.HiddenLayer(num_inputs=num_inputs, num_layers=50, activation_function="relu")
output_layer = dnnc.OutputLayer(num_inputs=50, num_outputs=num_outputs, num_layers=20, activation_function="softplus")


def predict(x):
    return output_layer.predict(hidden_layer.predict(x))


def cross_entropy(yi, predicted):

    val = 0
    predicted = predicted.flatten()
    for i in zip(yi, predicted):
        val -= i[0] * math.log(i[1] + 1e-15)
    return val


def accuracy(x, y):
    correct = 0
    for xi, yi in zip(x, y):
        pred = predict(xi)
        if np.argmax(pred) == np.argmax(yi):
            correct += 1
    return correct / len(X)


def argmax_one_hot(prediction):
    max_val = max(prediction[0])
    return [1.0 if i == max_val else 0.0 for i in prediction[0]]


def train(epochs, x, y):
    for epoch in range(epochs):
        # Backpropagation
        hidden_out = hidden_layer.predict(x)
        output_w1_grads, output_w2_grads, output_b1_grads, output_b2_grads, output_delta_hidden = output_layer.back_propagate(hidden_out, y)
        hidden_w_grads, hidden_b_grads, _ = hidden_layer.back_propagate(x, output_delta_hidden, output_layer.input_weights)

        # Update weights and biases
        output_layer.input_weights -= lr * output_w1_grads
        output_layer.output_weights -= lr * output_w2_grads
        output_layer.input_biases -= lr * output_b1_grads
        output_layer.output_biases -= lr * output_b2_grads
        hidden_layer.input_weights -= lr * hidden_w_grads
        hidden_layer.input_biases -= lr * hidden_b_grads

        # Logging
        if epoch % 100 == 0:
            acc = accuracy(x, y)
            loss = np.mean([cross_entropy(y[i], predict(x[i:i+1])) for i in range(len(x))])
            print(f"Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {acc:.2%}")


train(10000, X, Y)

test_x = test.values[:, :4]
test_y = np.array([targets[int(x)] for x in test.values[:, 4:5]])
corrects = 0
for i in range(len(test_x)):
    pred = argmax_one_hot(predict(test_x[i]))
    result = np.array_equal(pred, test_y[i])
    print(pred, test_y[i])
    if result:
        corrects += 1

print(corrects/len(test_y))

