'''
CNN Layer - supports 2D and 3D inputs
'''
import abc
import math

import numpy as np

from layers.activation import ActivationType, Activation


class BaseConvolution(abc.ABC):
    def __init__(
        self,
        filters: np.ndarray,
        biases: np.ndarray,
        size: int,
        stride: int,
        padding: int,
        activation: ActivationType=ActivationType.RELU,
    ):
        # array of filters
        self.filters = filters
        # biases per filter
        self.biases = biases
        # filter is a square - (self.size, self.size)
        self.size = size
        # how many pixels to move over
        self.stride = stride
        # zero pad the input to keep the original size
        self.padding = padding
        # activation function
        self.activation = activation

        if activation == ActivationType.RELU:
            self.activation = Activation.relu
        elif activation == ActivationType.SIGMOID:
            self.activation = Activation.sigmoid
        else:
            raise Exception('Invalid activation function')

    @abc.abstractmethod
    def operate(self, data:np.ndarray) -> np.ndarray:
        raise NotImplementedError()


class WindowConvolution(BaseConvolution):
    def operate(self, data: np.ndarray) -> np.ndarray:
        if len(data.shape) != 3:
            raise Exception('Expected a 3 dimensional matrix')

        rows, columns, _ = data.shape

        out_row_count = math.ceil((rows - self.size + 2 * self.padding) / self.stride + 1)
        out_column_count = math.ceil((columns - self.size + 2 * self.padding) / self.stride + 1)
        out_depth_count = self.filters.shape[0]

        output = np.zeros((out_row_count, out_column_count, out_depth_count))

        row_offset = 0
        column_offset = 0

        # for each filter
        for f, _filter in enumerate(self.filters):
            bias = self.biases[f]

            # for each row
            for i in range(out_row_count):
                # for each column
                column_offset = 0

                for j in range(out_column_count):
                    total = 0

                    for k in range(data.shape[2]):
                        # for each layer
                        padded_layer = np.pad(data[:,:,k], self.padding, 'constant')
                        window = padded_layer[row_offset:row_offset+self.size,column_offset:column_offset+self.size]
                        total += np.sum(window * _filter[:,:,k])

                    output[i,j,f] = self.activation(total + bias)
                    column_offset += self.stride

                row_offset += self.stride

        return output


# Matrix Math convolution using Toeplitz matrices
class MMConvolution(BaseConvolution):
    @abc.abstractmethod
    def operate(self, data: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
