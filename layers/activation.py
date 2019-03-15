from enum import Enum, auto

import numpy as np


class ActivationType(Enum):
    RELU = auto()
    SIGMOID = auto()
    SOFTMAX = auto()


class Activation:
    @staticmethod
    def relu(data: np.ndarray, derivative=False) -> np.ndarray:
        if not derivative:
            data[data < 0] = 0
            return data

        else:
            raise NotImplementedError()

    @staticmethod
    def sigmoid(data: np.ndarray, derivative=False) -> np.ndarray:
        if not derivative:
            return 1 / (1 + np.exp(-data))

        else:
            raise NotImplementedError()

    @staticmethod
    def softmax(data: float, derivative=False) -> float:
        if not derivative:
            return np.exp(data) / sum(np.exp(data))

        else:
            raise NotImplementedError()
