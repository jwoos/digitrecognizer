'''
CNN Layers - supports 2D and 3D inputs
'''

from enum import Enum, auto
import math

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


class Convolution:
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

    def operate(self, data: np.ndarray) -> np.ndarray:
        if self.activation == ActivationType.RELU:
            activation = Activation.relu
        elif self.activation == ActivationType.SIGMOID:
            activation = Activation.sigmoid
        else:
            raise Exception('Invalid activation function')

        if len(data.shape) == 2:
            rows, columns = data.shape
        else:
            rows, columns, _ = data.shape

        out_row_count = math.ceil((rows - self.size + 2 * self.padding) / self.stride + 1)
        out_column_count = math.ceil((columns - self.size + 2 * self.padding) / self.stride + 1)
        out_depth_count = self.filters.shape[0]

        output = np.zeros((out_row_count, out_column_count, out_depth_count))

        row_offset = 0
        column_offset = 0

        if len(data.shape) == 2:
            # 2 dimensional data

            # pad the data
            data = np.pad(data, self.padding, 'constant')

            # for each filter
            for k in range(out_depth_count):
                # for each row
                for i in range(out_row_count):
                    # for each column
                    column_offset = 0
                    for j in range(out_column_count):
                        window = data[row_offset:row_offset+self.size,column_offset:column_offset+self.size]
                        output[i,j,k] = activation(np.sum(window * self.filters[0]) + self.biases[0])

                        column_offset += self.stride

                    row_offset += self.stride
        else:
            # 3 dimensional data

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

                        output[i,j,f] = total + bias
                        # output[i,j,f] = activation(total)
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

        if len(data.shape) == 2:
            rows, columns = data.shape
            depth = 1
        else:
            rows, columns, depth = data.shape

        out_row_count = math.ceil((rows - self.size) / self.stride + 1)
        out_column_count = math.ceil((columns - self.size) / self.stride + 1)
        out_depth_count = depth

        if len(data.shape) == 2:
            output = np.zeros((out_row_count, out_column_count))
        else:
            output = np.zeros((out_row_count, out_column_count, out_depth_count))

        row_offset = 0
        column_offset = 0

        # Only difference between 2D and 3D pooling is that 3D
        # is pooled per depth layer
        if len(data.shape) == 2:
            # 2 dimensional data

            for i in range(out_row_count):
                column_offset = 0

                for j in range(out_column_count):
                    window = data[row_offset:row_offset+self.size,column_offset:column_offset+self.size]
                    output[i,j] = operation(window)

                    column_offset += self.stride

                row_offset += self.stride
        else:
            # 3 dimensional data

            for i in range(out_row_count):
                column_offset = 0

                for j in range(out_column_count):
                    for k in range(out_depth_count):
                        window = data[row_offset:row_offset+self.size,column_offset:column_offset+self.size, k]
                        output[i,j,k] = operation(window)

                    column_offset += self.stride

                row_offset += self.stride

        return output


class FC:
    def __init__(self):
        pass
