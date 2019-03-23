from typing import Callable, Tuple

from layers import base
import utils

import numpy as np


class Flatten(base.BaseLayer):
    def __init__(self):
        super().__init__(
            units=1,
            initialize_weights=utils.zeros,
            use_biases=False,
            initialize_biases=utils.zeros,
        )

    def initialize(self, input_shape: Tuple[int, int, int]):
        super().initialize(input_shape)

    def forward(self, data: np.ndarray) -> np.ndarray:
        return data.flatten()

    def backward(self, data: np.ndarray, output: np.ndarray, delta: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError()

    def infer_output_shape(self, input_shape: Tuple[int, int, int]) -> Tuple[int]:
        return (np.product(input_shape),)
