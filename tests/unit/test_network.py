from unittest import TestCase, mock

import layers
from network import Network

import numpy as np
import pytest


class TestNetworkInitialization(TestCase):
    def test_constructor(self):
        args = {
            'layers': [],
            'epochs': 0,
            'learning_rate': 0.01,
            'loss': layers.loss.cross_entropy,
            'optimization': layers.optimization.stochastic_gradient_descent,
        }

        network = Network(**args)

        for k, v in args.items():
            self.assertEqual(v, getattr(network, k))

    def test_initialization(self):
        network = Network(
            layers=[
                layers.convolution.WindowConvolution(
                    units=3,
                    size=3,
                    stride=1,
                    padding=1,
                ),
                layers.fully_connected.FC(
                    units=10
                ),
            ],
            epochs=0,
            learning_rate=0,
            loss=layers.loss.cross_entropy,
            optimization=layers.optimization.stochastic_gradient_descent,
        )

        network.initialize((28, 28, 3))

        self.assertEqual(network.input_shape, (28, 28, 3))
        self.assertEqual(network.layers[0].input_shape, (28, 28, 3))
        self.assertEqual(network.layers[0].output_shape, (28, 28, 3))
        self.assertEqual(network.layers[1].input_shape, (28, 28, 3))
        self.assertEqual(network.layers[1].output_shape, (10,))
        self.assertEqual(network.output_shape, (10,))
