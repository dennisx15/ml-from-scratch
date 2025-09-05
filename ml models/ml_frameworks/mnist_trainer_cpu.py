from ml_frameworks import deep_neural_net_framework_cpu as dnnfc
from datasets.DataLoaders import mnist_digits
from graphing import graphing_framework
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Hyperparameters
num_inputs = 784        # 28x28 pixels per image, flattened
num_outputs = 10        # 10 digit classes (0–9)
lr = 0.04               # Learning rate
epochs = 30             # Training epochs
batch_size = 20         # Mini-batch size

# --- File paths (change these accordingly to your setup) ---
input_path = '../input'
training_images_filepath = 'C:\\Users\\denni\\PycharmProjects\\ml-from-scratch\\datasets\\Mnist\\train-images-idx3-ubyte\\train-images-idx3-ubyte'
training_labels_filepath = 'C:\\Users\\denni\\PycharmProjects\\ml-from-scratch\\datasets\\Mnist\\train-labels-idx1-ubyte\\train-labels-idx1-ubyte'
test_images_filepath = 'C:\\Users\\denni\\PycharmProjects\\ml-from-scratch\\datasets\\Mnist\\t10k-images-idx3-ubyte\\t10k-images-idx3-ubyte'
test_labels_filepath = 'C:\\Users\\denni\\PycharmProjects\\ml-from-scratch\\datasets\\Mnist\\t10k-labels-idx1-ubyte\\t10k-labels-idx1-ubyte'


# --- Load MNIST dataset ---
mnist = mnist_digits.MnistDataloader(training_images_filepath, training_labels_filepath,
                                    test_images_filepath, test_labels_filepath)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# --- Preprocess data ---
# Flatten images into 784-dim vectors, convert to float32, normalize [0, 1]
x_train = x_train.reshape(x_train.shape[0], -1).astype(np.float32) / 255.0
x_test = x_test.reshape(x_test.shape[0], -1).astype(np.float32) / 255.0

# Convert integer labels to one-hot vectors
y_train = np.eye(num_outputs)[y_train]
y_test = np.eye(num_outputs)[y_test]

# --- Build neural network model ---
hidden_layer = dnnfc.HiddenLayer(num_inputs=num_inputs, num_layers=140, activation_function="swish")
hidden_layer2 = dnnfc.HiddenLayer(num_inputs=140, num_layers=112, activation_function="swish")
hidden_layer3 = dnnfc.HiddenLayer(num_inputs=112, num_layers=84, activation_function="swish")
output_layer = dnnfc.OutputLayer(num_inputs=84, num_outputs=num_outputs, num_layers=56, activation_function="swish")

# Stack hidden layers and output layer into a full model
model = dnnfc.NeuralNetwork(
    [
        hidden_layer,
        hidden_layer2,
        hidden_layer3
    ],
    output_layer=output_layer
)

# --- Train model ---
model.train(x_train, y_train, epochs=epochs, lr=lr, batch_size=batch_size, shuffle=True, drop_last=False, seed=42)

# Save trained model parameters to file
model.save_model("C:\\Users\\denni\\OneDrive\\Desktop\\model_weights.npz")

# --- Evaluate model on test set ---
test_x = x_test
test_y = y_test
print(model.accuracy(test_x, test_y))  # Print test accuracy

# --- Predictions and visualization ---
y_pred = model.predict(test_x)
y_true = test_y

# Plot confusion matrix of predictions vs. true labels
graphing_framework.plot_confusion_matrix(num_classes=10, y_true=y_true, y_pred=y_pred)

