from typing import Callable, Tuple

from layers import base

import numpy as np


class FC(base.BaseLayer):
    def initialize(self, input_shape: Tuple[int, int]) -> None:
        super().initialize(input_shape)

        self.weights = self.initialize_weights(self.input_shape[1], self.output_shape[1])
        self.biases = self.initialize_biases(self.output_shape[1])

    def forward(self, data: np.ndarray) -> np.ndarray:
        output = np.dot(data, self.weights) + self.biases

        return self.activation(output)

    def backward(self, data: np.ndarray, output: np.ndarray, delta: np.ndarray, previous_weight: np.ndarray=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        error = np.dot(delta, previous_weight) * self.activation(output, derivative=True)
        weight_gradient = np.dot(data.T, error)
        bias_gradient = np.sum(error, axis=0, keepdims=True)

        return error, weight_gradient, bias_gradient

    def infer_output_shape(self, input_shape: Tuple[int, int]) -> Tuple[int, int]:
        return (self.input_shape[0], self.units)


class Output(FC):
    def backward(self, data: np.ndarray, output: np.ndarray, delta: np.ndarray, previous_weight: np.ndarray=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        error = delta * self.activation(output, derivative=True)
        weight_gradient = np.dot(data.T, error)
        bias_gradient = np.sum(error, axis=0, keepdims=True)

        return error, weight_gradient, bias_gradient
