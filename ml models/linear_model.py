x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = [i * 2 + 6 for i in x]
batch_size = 2
intercept = 0.0
slope = 1
learning_rate = 0.001


def predict(value, slope, intercept):
    return intercept + value*slope


def sum_of_squared_residuals(x, y):
    result = 0
    for i in range(len(x)):
        result += (predict(x[i], slope, intercept) - y[i]) ** 2
    return result


def sum_of_predict_derivatives_w_respect_to_slope(x, y):
    result = 0.0
    for i in range(len(x)):
        result += 2 * (predict(x[i], slope, intercept) - y[i]) * x[i]
    return result


def sum_of_predict_derivatives_w_respect_to_intercept(x, y):
    result = 0.0
    for i in range(len(x)):
        result += 2 * (predict(x[i], slope, intercept) - y[i])
    return result


while sum_of_squared_residuals(x, y) > 1e-6:
    slope -= sum_of_predict_derivatives_w_respect_to_slope(x, y) * learning_rate
    intercept -= sum_of_predict_derivatives_w_respect_to_intercept(x, y) * learning_rate


print(slope)
print(intercept)

