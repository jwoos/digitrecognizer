from typing import Callable

from layers import activation

import numpy as np


class FC:
    def __init__(
        self,
        input_size: int,
        size: int,
        biases: np.ndarray=None,
        weights: np.ndarray=None,
        activation: Callable[[np.ndarray], np.ndarray]=activation.relu,
    ):
        self.input_size = input_size
        self.size = size
        self.biases = biases
        self.activation = activation
        if weights is None:
            self.weights = np.random.randn(input_size, size)
        else:
            self.weights = weights

    def operate(self, data: np.ndarray) -> np.ndarray:
        if len(data.shape) != 2 and data.shape[0] != 1:
            raise Exception('Expected a 2 dimensional flat matrix')

        if self.biases:
            return self.activation(np.dot(data, self.weights) + self.biases)
        else:
            return self.activation(np.dot(data, self.weights))


class Output(FC):
    pass
