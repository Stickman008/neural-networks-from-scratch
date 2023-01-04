import numpy as np

# Softmax activation
class Activation_Softmax:
    # Forward pass
    def forward(self, inputs):
        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        
        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
        

if __name__ == '__main__':
    ''' Test 1 '''
    # softmax = Activation_Softmax()
    # inputs =  [4.8, 1.21, 2.385]
    
    # softmax.forward(inputs)
    # print(softmax.output, np.sum(softmax.output))
    
    ''' Test 2 '''
    # layer_outputs = np.array([
    #     [4.8, 1.21, 2.385],
    #     [8.9, -1.81, 0.2],
    #     [1.41, 1.051, 0.026]
    #     ])
    # print('Sum without axis')
    # print(np.sum(layer_outputs))
    # print('This will be identical to the above since default is None:')
    # print(np.sum(layer_outputs, axis=None))
    
    # print('Sum axis 1, but keep the same dimensions as input:')
    # print(np.sum(layer_outputs, axis=1, keepdims=True))
    
    ''' Test 3 '''
    # softmax = Activation_Softmax()
    # softmax.forward([[0.5, 1, 1.5]])
    # print(softmax.output)