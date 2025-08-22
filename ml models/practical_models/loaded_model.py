from practical_models import deep_neural_net_classes as dnnc
import numpy as np
from PIL import Image

# Load the image
img = Image.open("C:\\Users\\denni\\OneDrive\\Desktop\\one.png").convert("L")  # Convert to grayscale

# Resize to 28x28
img = img.resize((28, 28))

# Convert to numpy array
img_array = np.array(img, dtype=np.float32)


img_array /= 255.0


img_vector = img_array.flatten()

img_vector = img_vector.reshape(1, -1)



hidden_layer = dnnc.HiddenLayer(num_inputs=784, num_layers=140, activation_function="swish")
hidden_layer2 = dnnc.HiddenLayer(num_inputs=140, num_layers=112, activation_function="swish")
hidden_layer3 = dnnc.HiddenLayer(num_inputs=112, num_layers=84, activation_function="swish")
output_layer = dnnc.OutputLayer(num_inputs=84, num_outputs=10, num_layers=56, activation_function="swish")

model = dnnc.NeuralNetwork([
    hidden_layer,
    hidden_layer2,
    hidden_layer3
], output_layer=output_layer)

model.load_model("C:\\Users\\denni\\OneDrive\\Desktop\\model_weights.npz")

print(f'I think this number is {np.argmax(model.predict(img_vector[0]))}')