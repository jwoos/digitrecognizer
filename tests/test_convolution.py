from unittest import TestCase, mock

import layers

import numpy as np
import pytest


class TestConvolutionInitialization(TestCase):
    def test_constructor(self):
        args = {
            'units': 3,
            'size': 3,
            'stride': 1,
            'padding': 1,
        }
        conv = layers.convolution.WindowConvolution(**args)

        for k, v in args.items():
            self.assertEqual(v, getattr(conv, k))

        not_none_attrs = (
            'units',
            'initialize_weights',
            'initialize_biases',
        )
        for x in not_none_attrs:
            self.assertIsNotNone(getattr(conv, x))

        none_attrs = (
            'weights',
            'biases',
            'input_shape',
            'output_shape',
        )
        for x in none_attrs:
            self.assertIsNone(getattr(conv, x))

    def test_initialization(self):
        input_shape = (28, 28, 3)
        output_shape = (28, 28, 10)

        conv = layers.convolution.WindowConvolution(
            units=10,
            size=3,
            stride=1,
            padding=1,
            use_biases=False,
        )
        conv.initialize(input_shape=input_shape)

        self.assertEqual(conv.input_shape, input_shape)
        self.assertIsNotNone(conv.output_shape, output_shape)
        self.assertTrue(np.array_equal(conv.biases, np.zeros(10)))

    def test_initialization_with_bias(self):
        input_shape = (28, 28, 3)
        output_shape = (28, 28, 10)

        conv = layers.convolution.WindowConvolution(
            units=10,
            size=3,
            stride=1,
            padding=1,
            use_biases=True,
        )
        conv.initialize(input_shape=input_shape)

        self.assertEqual(conv.input_shape, input_shape)
        self.assertIsNotNone(conv.output_shape, output_shape)
        self.assertFalse(np.array_equal(conv.biases, np.zeros(10)))


class TestConvolutionOperation(TestCase):
    def test_3d(self):
        filter1 = np.zeros((3, 3, 3))
        filter1[:,:,0] = np.array([
            [0, 1, 1],
            [0, 0, 0],
            [-1, -1, 0],
        ])
        filter1[:,:,1] = np.array([
            [0, 0, -1],
            [-1, -1, 0],
            [1, 1, 1],
        ])
        filter1[:,:,2] = np.array([
            [-1, -1, 0],
            [1, 0, 1],
            [1, 0, 0],
        ])
        conv = layers.convolution.WindowConvolution(
            filters=np.array([
                filter1
            ]),
            biases=[1],
            size=3,
            stride=2,
            padding=1,
            activation=layers.activation.relu,
        )

        data = np.zeros((5, 5, 3))
        data[:,:,0] = np.array([
            [0, 0, 1, 2, 0],
            [2, 0, 2, 0, 1],
            [2, 2, 2, 0, 0],
            [1, 2, 1, 0, 0],
            [1, 2, 0, 2, 2],
        ])
        data[:,:,1] = np.array([
            [0, 1, 0, 2, 1],
            [2, 2, 0, 1, 0],
            [0, 1, 1, 0, 0],
            [2, 2, 0, 1, 1],
            [1, 0, 2, 0, 0],
        ])
        data[:,:,2] = np.array([
            [1, 1, 2, 2, 1],
            [0, 0, 1, 2, 0],
            [2, 1, 0, 2, 1],
            [2, 0, 0, 0, 0],
            [2, 1, 2, 0, 1],
        ])

        conv.operate(data)


class TestPoolOperation(TestCase):
    def test_2d(self):
        pool = layers.pool.Pool(
            size=2,
            stride=2,
            operation=layers.pool.PoolOperation.MAX,
        )

        pool.operate(np.array([
            [1, 1, 2, 4],
            [5, 6, 7, 8],
            [3, 2, 1, 0],
            [1, 2, 3, 4],
        ]))

    def test_3d(self):
        pool = layers.pool.Pool(
            size=2,
            stride=2,
            operation=layers.pool.PoolOperation.MAX,
        )

        data = np.zeros((4, 4, 3))
        data[:,:,0] = np.array([
            [1, 1, 2, 4],
            [5, 6, 7, 8],
            [3, 2, 1, 0],
            [1, 2, 3, 4],
        ])
        data[:,:,1] = np.array([
            [1, 1, 2, 4],
            [5, 6, 7, 8],
            [3, 2, 1, 0],
            [1, 2, 3, 4],
        ])
        data[:,:,2] = np.array([
            [1, 1, 2, 4],
            [5, 6, 7, 8],
            [3, 2, 1, 0],
            [1, 2, 3, 4],
        ])

        pool.operate(data)
