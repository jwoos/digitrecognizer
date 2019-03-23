from typing import Callable, Tuple
import math

from layers import base
import utils

import numpy as np


class Pool(base.BaseLayer):
    def __init__(self, size: int, stride: int, operation: Callable[..., float]=np.max, **kwargs):
        super().__init__(units=1)

        # pooling window is a square - (self.size, self.size)
        self.size = size
        # how many pixels to move over
        self.stride = stride
        # which pooling operation should be done
        self.operation = operation

    def initialize(self, input_shape: Tuple[int, int, int]):
        super().initialize(input_shape)

        self.weights = None
        self.biases = None

    def forward(self, data: np.ndarray) -> np.ndarray:
        if len(data.shape) != 3:
            raise Exception('Expected a 3 dimensional matrix')

        rows, columns, depth = data.shape

        out_row_count = math.ceil((rows - self.size) / self.stride + 1)
        out_column_count = math.ceil((columns - self.size) / self.stride + 1)
        out_depth_count = depth

        output = np.zeros((out_row_count, out_column_count, out_depth_count))

        row_offset = 0
        column_offset = 0

        for i in range(out_row_count):
            column_offset = 0

            for j in range(out_column_count):
                for k in range(out_depth_count):
                    window = data[row_offset:row_offset+self.size,column_offset:column_offset+self.size, k]
                    output[i,j,k] = self.operation(window)

                column_offset += self.stride

            row_offset += self.stride

        return output

    def backward(self, data: np.ndarray, output: np.ndarray, delta: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError()

    def infer_output_shape(self, input_shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
        rows, columns, channels = input_shape.shape()
        return ((rows - self.size) / self.stride + 1, (columns - self.size) / self.stride + 1, channels)
