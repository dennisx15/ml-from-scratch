"""
=====================================================
 MNIST Digit Prediction using CPU Neural Network
=====================================================

This script loads a trained CPU-based deep neural network model 
and predicts the class of a single grayscale image.

Features:
 - Loads an image, converts it to 28x28 grayscale
 - Normalizes pixel values to [0,1]
 - Flattens the image to match input size of the network (784)
 - Loads pre-trained model weights from a .npz file
 - Performs forward pass through the network
 - Prints the predicted digit

Note:
 - GPU version not implemented since inference on a single image 
   is very fast on CPU.

Author: Dennis Alacahanli
Purpose: Testing the project results on real life data
"""

from ml_frameworks import deep_neural_net_framework_cpu as dnnfc
import numpy as np
from PIL import Image

# ----------------- Load and Preprocess Image ----------------- #
# Load the image and convert to grayscale
img = Image.open("C:\\Users\\denni\\OneDrive\\Desktop\\three.png").convert("L")  

# Resize the image to 28x28 pixels (MNIST standard)
img = img.resize((28, 28))

# Convert the image to a NumPy array of type float32
img_array = np.array(img, dtype=np.float32)

# Normalize pixel values to [0, 1]
img_array /= 255.0

# Flatten the 2D image into a 1D vector (784)
img_vector = img_array.flatten()

# Reshape to a 2D array with one sample for the network input
img_vector = img_vector.reshape(1, -1)

# ----------------- Build Neural Network ----------------- #
hidden_layer = dnnfc.HiddenLayer(num_inputs=784, num_layers=140, activation_function="swish")
hidden_layer2 = dnnfc.HiddenLayer(num_inputs=140, num_layers=112, activation_function="swish")
hidden_layer3 = dnnfc.HiddenLayer(num_inputs=112, num_layers=84, activation_function="swish")
output_layer = dnnfc.OutputLayer(num_inputs=84, num_outputs=10, num_layers=56, activation_function="swish")

# Stack hidden layers and output layer into a full model
model = dnnfc.NeuralNetwork([
    hidden_layer,
    hidden_layer2,
    hidden_layer3
], output_layer=output_layer)

# ----------------- Load Pre-Trained Weights ----------------- #
model.load_model("C:\\Users\\denni\\OneDrive\\Desktop\\model_weights.npz", printing=True)

# ----------------- Perform Prediction ----------------- #
predicted_class = np.argmax(model.predict(img_vector[0]))
print(f'I think this number is {predicted_class}')

