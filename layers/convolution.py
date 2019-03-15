'''
CNN Layer - supports 2D and 3D inputs
'''
import abc
import math
from typing import Callable

import numpy as np

from layers import activation


class BaseConvolution(abc.ABC):
    def __init__(
        self,
        size: int,
        count: int,
        stride: int,
        padding: int,
        filters: np.ndarray=None,
        biases: np.ndarray=None,
        activation: Callable[[np.ndarray], np.ndarray]=activation.relu,
    ):
        # filter is a square - (self.size, self.size)
        self.size = size
        # how many pixels to move over
        self.stride = stride
        # zero pad the input to keep the original size
        self.padding = padding
        # activation function
        self.activation = activation
        # biases per filter
        self.biases = biases

        # array of filters
        if filters is None:
            self.filters = np.random.randn(self.size, self.size, self.count)
        else:
            self.filters = filters

    @abc.abstractmethod
    def operate(self, data:np.ndarray) -> np.ndarray:
        raise NotImplementedError()


class WindowConvolution(BaseConvolution):
    def operate(self, data: np.ndarray) -> np.ndarray:
        if len(data.shape) != 3:
            raise Exception('Expected a 3 dimensional matrix')

        rows, columns, channels = data.shape

        out_row_count = math.ceil((rows - self.size + 2 * self.padding) / self.stride + 1)
        out_column_count = math.ceil((columns - self.size + 2 * self.padding) / self.stride + 1)
        out_depth_count = self.filters.shape[0]

        output = np.zeros((out_row_count, out_column_count, out_depth_count))

        row_offset = 0
        column_offset = 0

        # for each filter
        for f, _filter in enumerate(self.filters):
            if self.biases:
                bias = self.biases[f]

            # for each row
            for i in range(out_row_count):
                # for each column
                column_offset = 0

                for j in range(out_column_count):
                    total = 0

                    for k in range(channels):
                        # for each layer
                        padded_layer = np.pad(data[:,:,k], self.padding, 'constant')
                        window = padded_layer[row_offset:row_offset+self.size,column_offset:column_offset+self.size]
                        total += np.sum(window * _filter[:,:,k])

                    output[i,j,f] = total + bias
                    column_offset += self.stride

                row_offset += self.stride

        return self.activation(output)


# Matrix Math convolution using Toeplitz matrices
class MMConvolution(BaseConvolution):
    @abc.abstractmethod
    def operate(self, data: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
