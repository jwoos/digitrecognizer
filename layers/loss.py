import numpy as np


def mean_squared_error(data: np.ndarray, expected: np.ndarray) -> float:
    return np.sum((data - expected) ** 2) / len(data)

def cross_entropy(data: np.ndarray, expected: np.ndarray, epsilon=1e-12) -> float:
    data = np.clip(data, epsilon, 1 - epsilon)
    return -np.sum(expected * np.log(data)) / len(data)
