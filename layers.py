import numpy as np

MAGNITUDE = 0.01

class Layer_Dense:
    
    def __init__(self, n_inputs, n_neurons):
        self.weights = MAGNITUDE * np.random.randn(n_inputs, n_neurons)
        self. biases = np.zeros((1, n_neurons))
    
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases