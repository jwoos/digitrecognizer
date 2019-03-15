import numpy as np


def relu(data: np.ndarray, derivative=False) -> np.ndarray:
    if not derivative:
        data[data < 0] = 0
        return data

    else:
        raise NotImplementedError()

def sigmoid(data: np.ndarray, derivative=False) -> np.ndarray:
    if not derivative:
        return 1 / (1 + np.exp(-data))

    else:
        raise NotImplementedError()

def softmax(data: float, derivative=False) -> np.ndarray:
    if not derivative:
        return np.exp(data) / sum(np.exp(data))

    else:
        raise NotImplementedError()
