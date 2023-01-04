import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

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


if __name__ == '__main__':
    ''' Test 1 Not implement self.output'''
    # n_inputs = 2
    # n_neurons = 4
    # layer = Layer_Dense(n_inputs, n_neurons)
    # print(layer.weights, layer.biases)
    
    # inputs = np.array([
    #     [1, 2],
    #     [2, 3],
    #           ])
    # outputs = layer.forward(inputs)
    # print(f'input shape: {np.array(inputs).shape}')
    # print(f'weights shape: {layer.weights.shape}')
    # print(f'biases shape: {layer.biases.shape}')
    # print(outputs)
    
    ''' Test 2 '''
    # Create dataset
    X, y = spiral_data(samples=100, classes=3)
    # Create Dense layer with 2 input features and 3 output values
    dense1 = Layer_Dense(2, 3)
    # Perform a forward pass of our training data through this layer
    dense1.forward(X)
    print(dense1.outputs[:5])

    