from practical_models import deep_neural_net_classes as dnnc
from datasets.DataLoaders import MnistDigits
from graphing import graphing_framework
import numpy as np

np.random.seed(42)
corrects = 0
num_inputs = 784
num_outputs = 10
num_classes = 10
lr = 0.04
epochs = 5
batch_size = 20


#change these paths accordingly
input_path = '../input'
training_images_filepath = 'C:\\Users\\denni\\PycharmProjects\\ml-from-scratch\\datasets\\Mnist\\train-images-idx3-ubyte\\train-images-idx3-ubyte'
training_labels_filepath = 'C:\\Users\\denni\\PycharmProjects\\ml-from-scratch\\datasets\\Mnist\\train-labels-idx1-ubyte\\train-labels-idx1-ubyte'
test_images_filepath = 'C:\\Users\\denni\\PycharmProjects\\ml-from-scratch\\datasets\\Mnist\\t10k-images-idx3-ubyte\\t10k-images-idx3-ubyte'
test_labels_filepath = 'C:\\Users\\denni\\PycharmProjects\\ml-from-scratch\\datasets\\Mnist\\t10k-labels-idx1-ubyte\\t10k-labels-idx1-ubyte'


def argmax_one_hot(prediction):
    one_hot = np.zeros_like(prediction)
    one_hot[np.arange(len(prediction)), np.argmax(prediction, axis=1)] = 1.0
    return one_hot


mnist = MnistDigits.MnistDataloader(training_images_filepath, training_labels_filepath,
                        test_images_filepath, test_labels_filepath)


(x_train, y_train), (x_test, y_test) = mnist.load_data()


x_train = x_train.reshape(x_train.shape[0], -1).astype(np.float32) / 255.0
x_test = x_test.reshape(x_test.shape[0], -1).astype(np.float32) / 255.0

y_train = np.eye(num_classes)[y_train]
y_test = np.eye(num_classes)[y_test]


#create a nueral network model
hidden_layer = dnnc.HiddenLayer(num_inputs=num_inputs, num_layers=140, activation_function="swish")
hidden_layer2 = dnnc.HiddenLayer(num_inputs=140, num_layers=112, activation_function="swish")
hidden_layer3 = dnnc.HiddenLayer(num_inputs=112, num_layers=84, activation_function="swish")
output_layer = dnnc.OutputLayer(num_inputs=84, num_outputs=num_outputs, num_layers=56, activation_function="swish")


model = dnnc.NeuralNetwork(
    [
    hidden_layer,
    hidden_layer2,
    hidden_layer3
    ],
    output_layer=output_layer)

model.train(x_train, y_train, epochs=epochs, lr=lr, batch_size=batch_size, shuffle=True, drop_last=False, seed=42)
model.save_model( "C:\\Users\\denni\\OneDrive\\Desktop\\model_weights.npz")


test_x = x_test
test_y = y_test
print(model.accuracy(test_x, test_y))

y_pred = model.predict(test_x)
y_true = test_y
graphing_framework.plot_confusion_matrix(num_classes=10, y_true=y_true, y_pred=y_pred)

