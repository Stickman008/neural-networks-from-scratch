import numpy as np

softmax_outputs = [
    [0.7, 0.1, 0.2],
    [0.1, 0.5, 0.4],
    [0.02, 0.9, 0.08]
    ]
softmax_outputs = np.array(softmax_outputs)
class_targets = [0, 1, 1] # dog, cat, cat

print(softmax_outputs[range(softmax_outputs.shape[0]), class_targets])
print(-np.log(softmax_outputs[range(softmax_outputs.shape[0]), class_targets]))
loss = np.mean(-np.log(softmax_outputs[range(softmax_outputs.shape[0]), class_targets]))
print(loss)