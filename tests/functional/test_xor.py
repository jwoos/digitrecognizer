from unittest import TestCase, mock

import network
import layers

import numpy as np
import pytest


class TestXOR(TestCase):
    def test_works(self):
        model = network.Network(
            layers=[
                layers.fully_connected.FC(units=10, activation=layers.activation.sigmoid),
                layers.fully_connected.Output(units=2, activation=layers.activation.sigmoid),
            ],
            epochs=1,
            learning_rate=0.5,
            loss=layers.loss.mean_squared_error,
            optimization=layers.optimization.stochastic_gradient_descent,
        )

        model.initialize((1, 2))

        data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        labels = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])

        model.train(data=data, labels=labels)
