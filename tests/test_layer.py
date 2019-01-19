from unittest import TestCase, mock

import numpy as np
import pytest

from .. import layer


class TestConvolutionInitialization(TestCase):
    def test_works(self):
        args = {
            'filters': np.zeros((1, 3, 3)),
            'biases': np.zeros((3,)),
            'size': 3,
            'stride': 1,
            'padding': 1,
            'activation': mock.Mock(),
        }
        conv = layer.Convolution(**args)

        for k, v in args.items():
            if k == 'filters' or k == 'biases':
                self.assertTrue(np.array_equal(v, getattr(conv, k)))
            else:
                self.assertEqual(v, getattr(conv, k))

class TestConvolutionOperation(TestCase):
    def test_2d(self):
        conv = layer.Convolution(
            filters=np.array([
                [
                    [0, 1, 1],
                    [0, 1, 0],
                    [-1, -1, 1],
                ],
            ]),
            biases=[0],
            size=3,
            stride=2,
            padding=1,
            activation=layer.ActivationType.RELU,
        )

        conv.operate(np.array([
            [0, 0, 1, 2, 0],
            [2, 0, 2, 0, 1],
            [2, 2, 2, 0, 0],
            [1, 2, 1, 0, 0],
            [1, 2, 0, 2, 2],
        ]))

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
        conv = layer.Convolution(
            filters=np.array([
                filter1
            ]),
            biases=[1],
            size=3,
            stride=2,
            padding=1,
            activation=layer.ActivationType.RELU,
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
        pool = layer.Pool(
            size=2,
            stride=2,
            operation=layer.PoolOperation.MAX,
        )

        pool.operate(np.array([
            [1, 1, 2, 4],
            [5, 6, 7, 8],
            [3, 2, 1, 0],
            [1, 2, 3, 4],
        ]))

    def test_3d(self):
        pool = layer.Pool(
            size=2,
            stride=2,
            operation=layer.PoolOperation.MAX,
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
