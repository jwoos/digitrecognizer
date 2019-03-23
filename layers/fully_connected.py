from typing import Callable, Tuple

from layers import base

import numpy as np


class FC(base.BaseLayer):
    def initialize(self, input_shape: Tuple[int]) -> None:
        super().initialize(input_shape)

        self.weights = self.initialize_weights(self.input_shape[0], self.output_shape[0])
        self.biases = self.initialize_biases(self.output_shape[0])

    def forward(self, data: np.ndarray) -> np.ndarray:
        output = np.dot(data, self.weights) + self.biases

        return self.activation(output)

    def backward(self, data: np.ndarray, output: np.ndarray, delta: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        error = delta.dot(self.weights.T) * self.activation(output, derivative=True)
        weight_gradient = data.T.dot(error)
        bias_gradient = np.mean(error, axis=0)

        return error, weight_gradient, bias_gradient

    def infer_output_shape(self, input_shape: Tuple[int]) -> Tuple[int]:
        return (self.units,)
