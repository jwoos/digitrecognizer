from enum import Enum, auto

import numpy as np


class ActivationType(Enum):
    RELU = auto()
    SIGMOID = auto()


class Activation:
    @staticmethod
    def relu(data: float) -> float:
        return max(0, data)

    def relu_array(data: np.ndarray) -> np.ndarray:
        data[data < 0] = 0
        return data

    @staticmethod
    def sigmoid(data: float) -> float:
        pass

    @staticmethod
    def sigmoid_array(data: np.ndarray) -> np.ndarray:
        pass
