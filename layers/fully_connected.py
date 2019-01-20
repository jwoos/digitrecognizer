import numpy as np

from layers.activation import ActivationType, Activation


class FC:
    def __init__(
        self,
        weights: np.ndarray,
        biases: float,
        activation: ActivationType=ActivationType.RELU,
    ):
        self.weights = weights
        self.biases = biases
        if activation == ActivationType.RELU:
            self.activation = Activation.relu
        elif activation == ActivationType.SIGMOID:
            self.activation = Activation.sigmoid
        else:
            raise Exception('Invalid activation function')

    def operate(self, data: np.ndarray) -> np.ndarray:
        return self.activation(np.dot(data, self.weights))
