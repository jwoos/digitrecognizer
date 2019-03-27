import abc
from typing import Callable, Tuple

import layers
import utils

import numpy as np


class BaseLayer(abc.ABC):
    def __init__(
        self,
        units: int,
        activation: Callable[[np.ndarray], np.ndarray]=layers.activation.relu,
        initialize_weights: Callable[..., np.ndarray]=np.random.randn,
        use_biases: bool=False,
        initialize_biases: Callable[..., np.ndarray]=np.random.randn,
    ):
        self.units = units
        self.activation = activation
        self.initialize_weights = initialize_weights
        self.use_biases = use_biases
        if use_biases:
            self.initialize_biases = initialize_biases
        else:
            self.initialize_biases = utils.zeros

        self.cache = None

        self.weights = None
        self.biases = None
        self.input_shape = None
        self.output_shape = None

    @abc.abstractmethod
    def initialize(
        self,
        input_shape: Tuple,
    ) -> None:
        self.input_shape = input_shape
        self.output_shape = self.infer_output_shape(input_shape)

    @abc.abstractmethod
    def forward(self, data: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    @abc.abstractmethod
    def backward(self, data: np.ndarray, output: np.ndarray, delta: np.ndarray, previous_weight: np.ndarray=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError()

    @abc.abstractmethod
    def infer_output_shape(self, input_shape: Tuple) -> Tuple:
        raise NotImplementedError()
