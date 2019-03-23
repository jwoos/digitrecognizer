from typing import Union

import numpy as np


def mean_squared_error(data: np.ndarray, expected: np.ndarray, derivative=False) -> Union[float, np.ndarray]:
    if not derivative:
        return np.sum((data - expected) ** 2) / len(data)
    else:
        raise NotImplementedError()


def cross_entropy(data: np.ndarray, expected: np.ndarray, derivative=False) -> Union[float, np.ndarray]:
    if not derivative:
        return np.sum(-np.log(data[range(len(expected)), expected])) / len(expected)
    else:
        data[range(len(expected)), expected] -= 1
        data /= len(expected)
        return data

