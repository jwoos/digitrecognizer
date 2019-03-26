'''
CNN Layer - supports 2D and 3D inputs
'''
import abc
import math
from typing import Callable, Tuple, Union

from layers import base

import numpy as np


class BaseConvolution(base.BaseLayer):
    def __init__(
        self,
        units: int,
        size: int,
        stride: int,
        padding: int,
        **kwargs,
    ):
        super().__init__(units, **kwargs)

        # how big the kernel is
        self.size = size
        # how many pixels to move over
        self.stride = stride
        # zero pad the input to keep the original size
        self.padding = padding

    def initialize(self, input_shape: Tuple[int, int, int]) -> None:
        super().initialize(input_shape)

        self.weights = self.initialize_weights(self.units, self.size, self.size, input_shape[2])
        self.biases = self.initialize_biases(self.units)

    def infer_output_shape(self, input_shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
        rows, columns, channels = input_shape

        return (
            math.ceil((rows - self.size + 2 * self.padding) / self.stride + 1),
            math.ceil((columns - self.size + 2 * self.padding) / self.stride + 1),
            self.units,
        )


class WindowConvolution(BaseConvolution):
    def forward(self, data: np.ndarray) -> np.ndarray:
        if len(data.shape) != 3:
            raise Exception('Expected a 3 dimensional matrix')

        out_row_count, out_column_count, out_depth_count = self.output_shape
        output = np.zeros((out_row_count, out_column_count, out_depth_count))

        padded_data = np.pad(
            data,
            pad_width=((self.padding, self.padding), (self.padding, self.padding), (0, 0)),
            mode='constant',
            constant_values=0,
        )

        # for each filter
        for f, _filter in enumerate(self.weights):
            # for each row
            for i in range(out_row_count):
                row_offset = i * self.stride

                # for each column
                for j in range(out_column_count):
                    column_offset = j * self.stride

                    output[i,j,f] = np.sum(padded_data[row_offset:row_offset+self.size,column_offset:column_offset+self.size,:] * _filter) + self.biases[f]

        return self.activation(output)

    def backward(self, data: np.ndarray, output: np.ndarray, delta: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        out_row_count, out_column_count, out_depth_count = self.input_shape
        error = np.zeros((out_row_count, out_column_count, out_depth_count))

        # for each filter
        for f, _filter, in enumerate(self.weights):
            # for each row
            for i in range(out_row_count):
                row_offset = i * self.stride

                # for each column
                for j in range(out_column_count):
                    column_offset = j * self.stride

                    error[row_offset:row_offset+self.size,column_offset:column_offset+self.size,:] += _filter * delta[i,j,:]

        weight_gradient = data.T.dot(error)
        bias_gradient = np.mean(error, axis=0)

        return error, weight_gradient, bias_gradient


# Matrix Math convolution using Toeplitz matrices
class MMConvolution(BaseConvolution):
    def forward(self, data: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def backward(self):
        raise NotImplementedError()
