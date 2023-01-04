import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]
outputs = np.maximum(0, inputs)
# print(outputs)

class Layer_Dense:
    
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights and biases
        MAGNITUDES = 0.01
        self.weights = MAGNITUDES * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        

    # Forward pass
    def forward(self, inputs):
        # Calculate output values from inputs, weights and biases
        self.outputs = np.dot(inputs, self.weights) + self.biases


# ReLU activation
class Activation_ReLU:
    
    # Forward pass
    def forward(self, inputs):
        # Calculate output values from input
        self.output = np.maximum(0, inputs)
        
if __name__ == '__main__':
    # Create dataset
    X, y = spiral_data(samples=100, classes=3)
    # Create Dense layer with 2 input features and 3 output values
    dense1 = Layer_Dense(2, 3)
    # Create ReLU activation (to be used with Dense layer):
    activation1 = Activation_ReLU()
    # Make a forward pass of our training data through this layer
    dense1.forward(X)
    # Forward pass through activation func.
    # Takes in output from previous layer
    activation1.forward(dense1.outputs)
    print(activation1.output[:5])