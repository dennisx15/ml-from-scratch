"""
=====================================================
 GPU-Accelerated Neural Network for MNIST Classification
=====================================================

This script trains a deep neural network on the MNIST dataset using
CuPy (NumPy on GPU) for acceleration.

Author: Dennis Alacahanli
Notes:
 - Adjust file paths as needed for dataset location.
 - Model weights are saved for reuse after training.
"""


from datasets.DataLoaders import mnist_digits # import Mnist digits data loader
import cupy as cp   # GPU-accelerated NumPy
import numpy as np   #for displaying results
from ml_frameworks import deep_neural_net_framework_gpu as dnnfg # import GPU accelerated deep neural net framework
from graphing import graphing_framework # import graphing framework
from data_processors.image_processors import grayscale_processor

# ----------------- Hyperparameters ----------------- #
np.random.seed(42)        # Set random seed for reproducibility
num_inputs = 784          # 28x28 images flattened
num_outputs = 10          # 10 classes (digits 0-9)
lr = 0.3                 # Learning rate
epochs = 80               # Number of training epochs
batch_size = 1000         # Mini-batch size for GPU training

# ----------------- File Paths ----------------- #
# Adjust these paths according to your local setup
input_path = '../input'
training_images_filepath = 'C:\\Users\\denni\\PycharmProjects\\ml-from-scratch\\datasets\\Mnist\\train-images-idx3-ubyte\\train-images-idx3-ubyte'
training_labels_filepath = 'C:\\Users\\denni\\PycharmProjects\\ml-from-scratch\\datasets\\Mnist\\train-labels-idx1-ubyte\\train-labels-idx1-ubyte'
test_images_filepath = 'C:\\Users\\denni\\PycharmProjects\\ml-from-scratch\\datasets\\Mnist\\t10k-images-idx3-ubyte\\t10k-images-idx3-ubyte'
test_labels_filepath = 'C:\\Users\\denni\\PycharmProjects\\ml-from-scratch\\datasets\\Mnist\\t10k-labels-idx1-ubyte\\t10k-labels-idx1-ubyte'

# ----------------- Load MNIST Dataset ----------------- #
mnist = mnist_digits.MnistDataloader(training_images_filepath, training_labels_filepath,
                                    test_images_filepath, test_labels_filepath)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

for i in range(len(x_train)):
    if np.random.choice([False, True]):
        x_train[i] = grayscale_processor.uncenter_image(x_train[i], top=2, bottom=2, left=2, right=2)
    if np.random.choice([False, False]):
        x_train[i] = grayscale_processor.noise_image(x_train[i], np.random.randint(20, 40), np.random.randint(50, 255))

# ----------------- Transfer data to GPU ----------------- #
x_train = cp.asarray(x_train)
y_train = cp.asarray(y_train)
x_test = cp.asarray(x_test)
y_test = cp.asarray(y_test)



# ----------------- Preprocess data ----------------- #
# Flatten images to vectors, convert to float32, normalize to [0,1]
x_train = x_train.reshape(x_train.shape[0], -1).astype(cp.float32) / 255.0
x_test = x_test.reshape(x_test.shape[0], -1).astype(cp.float32) / 255.0

# Convert integer labels to one-hot encoding for training
y_train = cp.eye(num_outputs)[y_train]
y_test = cp.eye(num_outputs)[y_test]

# ----------------- Build Neural Network Model ----------------- #
hidden_layer = dnnfg.HiddenLayer(num_inputs=num_inputs, num_layers=140, activation_function="swish")
hidden_layer2 = dnnfg.HiddenLayer(num_inputs=140, num_layers=112, activation_function="swish")
hidden_layer3 = dnnfg.HiddenLayer(num_inputs=112, num_layers=84, activation_function="swish")
output_layer = dnnfg.OutputLayer(num_inputs=84, num_outputs=num_outputs, num_layers=56, activation_function="softplus")

# Stack hidden layers and output layer into a full model
model = dnnfg.NeuralNetwork(
    [
        hidden_layer,
        hidden_layer2,
        hidden_layer3
    ],
    output_layer=output_layer,
    lr=lr,
    optimizer="MomentumSGD"
)

# ----------------- Train the Model ----------------- #
model.train(x_train, y_train, epochs=epochs, batch_size=batch_size,
            shuffle=True, drop_last=False, seed=42)

# ----------------- Save Trained Model ----------------- #
model.save_model("C:\\Users\\denni\\OneDrive\\Desktop\\model_weights.npz")

# ----------------- Plot Training Accuracy ----------------- #
graphing_framework.plot_accuracy(model.accuracies)

# ----------------- Evaluate on Test Set ----------------- #
test_x = x_test
test_y = y_test
print(model.accuracy(test_x, test_y))  # Print final test accuracy

# ----------------- Predictions & Confusion Matrix ----------------- #
y_pred = cp.asnumpy(model.predict(test_x))  # Move predictions to CPU for plotting
y_true = cp.asnumpy(test_y)                 # Move true labels to CPU
test_x = cp.asnumpy(test_x)
graphing_framework.plot_confusion_matrix(num_classes=num_outputs, y_true=y_true, y_pred=y_pred)

