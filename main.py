import numpy as np
import nnfs
from nnfs.datasets import spiral_data

from layers import Layer_Dense
from activations import Activation_ReLU, Activation_Softmax
from loss import CategoricalCrossentropy

nnfs.init()