import math
import numpy as np

# An example output from the output layer of the neural network
softmax_output = [0.7, 0.1, 0.2]
# Ground truth
target_output = [1, 0, 0]

loss = -(
    math.log(softmax_output[0])*target_output[0] +
    math.log(softmax_output[1])*target_output[1] +
    math.log(softmax_output[2])*target_output[2]
    )

print(loss)

loss_2 = -math.log(
            np.sum(
                np.array(softmax_output)*np.array(target_output)
                )
            )
print(loss_2)