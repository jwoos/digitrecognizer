'''
CNN functions - these only support 2D matrices
'''

from enum import Enum, auto

import numpy as np


class Convolution:
    def __init__(
        self,
        filters: np.ndarray,
        size: int,
        stride: int,
        padding: int,
    ):
        # array of filters
        self.filters = filters
        # filter is a square - (self.size, self.size)
        self.size = size
        # how many pixels to move over
        self.stride = stride
        # 0 pad the input to keep the original size
        self.padding = padding

    def operate(self, data: np.ndarray) -> np.ndarray:
        row_offset = 0
        column_offset = 0

        rows, columns = data.shape()

        out_row_count = (rows - self.size + 2 * self.padding) / self.stride + 1
        out_column_count = (columns - self.size + 2 * self.padding) / self.stride + 1

        # pad the data
        data = np.pad(data, self.padding, 'constant')

        output = np.zeros((out_row_count, out_column_count))

        for i in range(out_row_count):
            for j in range(out_column_count):
                window = data[row_offset:self.size,column_offset:self.size]
                output[i][j] = operation(window)

                column_offset += self.stride

            row_offset += self.stride

        return output


class PoolOperation(Enum):
    MAX = auto()
    AVERAGE = auto()
    SUM = auto()


class Pool:
    def __init__(self, size: int, stride: int, operation: PoolOperation=PoolOperation.AVERAGE):
        # pooling window is a square - (self.size, self.size)
        self.size = size
        # how many pixels to move over
        self.stride = stride
        # which pooling operation should be done
        self.operation = operation

    def operate(self, data: np.ndarray) -> np.ndarray:
        if self.operation == PoolOperation.AVERAGE:
            operation = np.mean

        elif self.operation == PoolOperation.SUM:
            operation = np.sum

        elif self.operation == PoolOperation.MAX:
            operation = np.amax

        else:
            raise Exception('Invalid operation type')

        row_offset = 0
        column_offset = 0

        rows, columns = data.shape()

        out_row_count = (rows - self.size) / self.stride + 1
        out_column_count = (columns - self.size) / self.stride + 1

        output = np.zeros((out_row_count, out_column_count))

        for i in range(out_row_count):
            for j in range(out_column_count):
                window = data[row_offset:self.size,column_offset:self.size]
                output[i][j] = operation(window)

                column_offset += self.stride

            row_offset += self.stride

        return output
