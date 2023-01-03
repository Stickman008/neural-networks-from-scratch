import numpy as np

# a = [1, 2, 3]
# b = [2, 3, 4]
# a = np.array([a])
# b = np.array([b]).T

# print(a.shape, b.shape)
# print(np.dot(a, b))

inputs = [
    [1.0, 2.0, 3.0, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]
] # (batch, features) (3, 4)
weights = [
    [0.2, 0.8, -0.5, 1.0],
    [0.5, -0.91, -0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87]
] # (n_neuron, features) (3, 4)
biases = [2.0, 3.0, 0.5]

output = np.dot(np.array(inputs), np.array(weights).T) + biases
print(output) # (batch, n_neuron)