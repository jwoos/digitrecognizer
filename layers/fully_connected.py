import numpy as np

from layers.activation import ActivationType, Activation


class FC:
    def __init__(
        self,
        input_size: int,
        size: int,
        biases: np.ndarray=None,
        weights: np.ndarray=None,
        activation: ActivationType=ActivationType.RELU,
    ):
        self.input_size = input_size
        self.size = size
        self.biases = biases
        if weights is None:
            self.weights = np.random.randn(input_size, size)
        else:
            self.weights = weights

        if activation == ActivationType.RELU:
            self.activation = Activation.relu
        elif activation == ActivationType.SIGMOID:
            self.activation = Activation.sigmoid
        else:
            raise Exception('Invalid activation function')

    def operate(self, data: np.ndarray) -> np.ndarray:
        if len(data.shape) != 2 and data.shape[0] != 1:
            raise Exception('Expected a 2 dimensional flat matrix')

        if self.biases:
            return self.activation(np.dot(data, self.weights) + self.biases)
        else:
            return self.activation(np.dot(data, self.weights))


class Output(FC):
    pass
