from typing import Callable, Tuple

from layers import base
import utils

import numpy as np


class Flatten(base.BaseLayer):
    def __init__(self):
        super().__init__(units=1)

    def initialize(self, input_shape: Tuple[int, int, int]):
        super().initialize(input_shape)

        self.weights = None
        self.biases = None

    def forward(self, data: np.ndarray) -> np.ndarray:
        return data.flatten()

    def backward(self, data: np.ndarray, output: np.ndarray, delta: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError()

    def infer_output_shape(self, input_shape: Tuple[int, int, int]) -> Tuple[int]:
        return (np.product(input_shape),)
