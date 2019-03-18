from typing import Callable, Tuple

from layers import activation, base

import numpy as np


class FC(base.BaseLayer):
    def initialize(self, input_shape: Tuple[int]) -> None:
        super().initialize(input_shape)

        self.weights = self.initialize_weights(self.input_shape[0], self.output_shape[0])
        self.biases = self.initialize_biases(self.output_shape[0])

    def forward(self, data: np.ndarray) -> np.ndarray:
        output = np.dot(data, self.weights) + self.biases

        return output

    def backward(self):
        raise NotImplementedError()

    def infer_output_shape(self, input_shape: Tuple[int]) -> Tuple[int]:
        return (self.units,)
