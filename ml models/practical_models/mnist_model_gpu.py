from datasets.DataLoaders import MnistDigits
import cupy as cp
import numpy as np
import time
from practical_models import deep_neural_net_classes_gpu as dnncg
from graphing import graphing_framework

np.random.seed(42)
corrects = 0
num_inputs = 784
num_outputs = 10
num_classes = 10
lr = 0.15
epochs = 95
batch_size = 1000

#Adjust these paths accordingly
input_path = '../input'
training_images_filepath = 'C:\\Users\\denni\\PycharmProjects\\ml-from-scratch\\datasets\\Mnist\\train-images-idx3-ubyte\\train-images-idx3-ubyte'
training_labels_filepath = 'C:\\Users\\denni\\PycharmProjects\\ml-from-scratch\\datasets\\Mnist\\train-labels-idx1-ubyte\\train-labels-idx1-ubyte'
test_images_filepath = 'C:\\Users\\denni\\PycharmProjects\\ml-from-scratch\\datasets\\Mnist\\t10k-images-idx3-ubyte\\t10k-images-idx3-ubyte'
test_labels_filepath = 'C:\\Users\\denni\\PycharmProjects\\ml-from-scratch\\datasets\\Mnist\\t10k-labels-idx1-ubyte\\t10k-labels-idx1-ubyte'

mnist = MnistDigits.MnistDataloader(training_images_filepath, training_labels_filepath,
                        test_images_filepath, test_labels_filepath)

start = time.time()
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = cp.asarray(x_train)
y_train = cp.asarray(y_train)
x_test = cp.asarray(x_test)
y_test = cp.asarray(y_test)

x_train = x_train.reshape(x_train.shape[0], -1).astype(cp.float32) / 255.0
x_test = x_test.reshape(x_test.shape[0], -1).astype(cp.float32) / 255.0

y_train = cp.eye(num_classes)[y_train]
y_test = cp.eye(num_classes)[y_test]


#create a nueral network model
hidden_layer = dnncg.HiddenLayer(num_inputs=num_inputs, num_layers=140, activation_function="swish")
hidden_layer2 = dnncg.HiddenLayer(num_inputs=140, num_layers=112, activation_function="swish")
hidden_layer3 = dnncg.HiddenLayer(num_inputs=112, num_layers=84, activation_function="swish")
output_layer = dnncg.OutputLayer(num_inputs=84, num_outputs=num_outputs, num_layers=56, activation_function="softplus")


model = dnncg.NeuralNetwork(
    [
    hidden_layer,
    hidden_layer2,
    hidden_layer3
    ],
    output_layer=output_layer)

model.train(x_train, y_train, epochs=epochs, lr=lr, batch_size=batch_size, shuffle=True, drop_last=False, seed=42)
model.save_model( "C:\\Users\\denni\\OneDrive\\Desktop\\model_weights.npz")
graphing_framework.plot_accuracy(model.accuracies)

test_x = x_test
test_y = y_test
print(model.accuracy(test_x, test_y))
y_pred = cp.asnumpy(model.predict(test_x))
y_true = cp.asnumpy(test_y)
graphing_framework.plot_confusion_matrix(num_classes=10, y_true=y_true, y_pred=y_pred)


