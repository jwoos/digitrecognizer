import numpy as np


def relu(data: np.ndarray, derivative=False) -> np.ndarray:
    if not derivative:
        data[data < 0] = 0
        return data

    else:
        data[data <= 0] = 0
        data[data > 0] = 1
        return data

def sigmoid(data: np.ndarray, derivative=False) -> np.ndarray:
    if not derivative:
        return 1 / (1 + np.exp(-data))

    else:
        return data * (1 - data)

def softmax(data: float, derivative=False) -> np.ndarray:
    if not derivative:
        exponentiated = np.exp(data)
        return exponentiated / np.sum(exponentiated, axis=1, keepdims=True)

    else:
        raise NotImplementedError()
