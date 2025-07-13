import math

x = [0, 0.5, 1.0, 1.5, 2.0]
y = [xi**2 + 2 for xi in x]

lr = 0.01
w1, w2, w3, w4 = 1, 1, 1, 1
b1, b2, b3 = 0, 0, 0


def softplus(number):
    return math.log(1 + math.e**number)


def neural_network1(number):
    hidden_layer = w1 * number + b1
    return softplus(hidden_layer) * w2


def neural_network2(number):
    hidden_layer = w3 * number + b2
    return softplus(hidden_layer) * w4


def predict(number):
    return neural_network1(number) + neural_network2(number) + b3


def sum_of_squared_residuals(inputs, answers):
    error = 0
    for i in range(len(inputs)):
        error += (answers[i] - predict(inputs[i])) ** 2
    return error


# This derivative also appears as a part of the derivative of every variable. You could call it the base derivative

def derivative_with_respect_to_b3(xi, yi):
    return -2*(yi - neural_network1(xi) - neural_network2(xi) - b3)


def derivative_with_respect_to_w2(xi, yi):
    hidden = w1 * xi + b1
    return derivative_with_respect_to_b3(xi, yi) * softplus(hidden)


def derivative_with_respect_to_w4(xi, yi):
    hidden = w3 * xi + b2
    return derivative_with_respect_to_b3(xi, yi) * softplus(hidden)


def derivative_with_respect_to_w1(xi, yi):
    return -2 * (yi - neural_network1(xi) - neural_network2(xi) - b3) * w2 * (math.e ** (w1 * xi + b1)) / (
            1 + math.e**(w1 * xi + b1)) * xi


def derivative_with_respect_to_w3(xi, yi):
    return -2 * (yi - neural_network1(xi) - neural_network2(xi) - b3) * w4 * (math.e ** (w3 * xi + b2)) / (
                1 + math.e ** (w3 * xi + b2)) * xi


def derivative_with_respect_to_b1(xi, yi):
    return -2 * (yi - neural_network1(xi) - neural_network2(xi) - b3) * w2 * (math.e ** (w1 * xi + b1)) / (
            1 + math.e**(w1 * xi + b1))


def derivative_with_respect_to_b2(xi, yi):
    return -2 * (yi - neural_network1(xi) - neural_network2(xi) - b3) * w4 * (math.e ** (w3 * xi + b2)) / (
                1 + math.e ** (w3 * xi + b2))


def train(epochs, x, y):
    global w1, w2, w3, w4, b1, b2, b3
    for epoch in range(epochs):
        dw1 = dw2 = dw3 = dw4 = db1 = db2 = db3 = 0

        for xi, yi in zip(x, y):
            dw1 += derivative_with_respect_to_w1(xi, yi)
            dw2 += derivative_with_respect_to_w2(xi, yi)
            dw3 += derivative_with_respect_to_w3(xi, yi)
            dw4 += derivative_with_respect_to_w4(xi, yi)
            db1 += derivative_with_respect_to_b1(xi, yi)
            db2 += derivative_with_respect_to_b2(xi, yi)
            db3 += derivative_with_respect_to_b3(xi, yi)

        # Average gradients
        n = len(x)
        w1 -= lr * dw1 / n
        w2 -= lr * dw2 / n
        w3 -= lr * dw3 / n
        w4 -= lr * dw4 / n
        b1 -= lr * db1 / n
        b2 -= lr * db2 / n
        b3 -= lr * db3 / n

        if epoch % 100 == 0:
            print(f"Epoch {epoch} | Loss: {sum_of_squared_residuals(x, y):.4f}")


