from enum import Enum, auto
import math

import numpy as np


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
