from typing import List, Callable, Tuple, Union

import layers

import numpy as np


class Network:
    def __init__(
        self,
        layers: List,
        epochs: int,
        learning_rate: float,
        loss: Callable[[np.ndarray, np.ndarray], Union[float, np.ndarray]]=layers.loss.cross_entropy,
        optimization: Callable[..., np.ndarray]=layers.optimization.stochastic_gradient_descent,
    ):
        self.layers = layers
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.loss = loss
        self.optimization = optimization

    # Infers output shape
    def initialize(self, input_shape: Tuple):
        self.input_shape = input_shape
        shape = input_shape

        for layer in self.layers:
            layer.initialize(shape)
            shape = layer.output_shape

        self.output_shape = shape

    def train(self, data: np.ndarray, labels: np.ndarray):
        for epoch in range(self.epochs):
            output = [None, None]

            for loss, accuracy in self.optimization(self, data, labels):
                output[0] = loss
                output[1] = accuracy

            print(f'[Epoch {epoch}] Loss: {output[0]} | Accuracy: {output[1]}')

    def validate(self, data: np.ndarray, labels: np.ndarray):
        correct = 0

        for image, label in zip(data, labels):
            for layer in self.layers:
                image = layer.forward(image)

            if label[np.argmax(image)]:
                correct += 1

        loss = np.sum(network.loss(outputs[-1], label))
        accuracy = correct / len(data)
        print(f'[Validation] Loss: {loss} | Accuracy: {accuracy}')
