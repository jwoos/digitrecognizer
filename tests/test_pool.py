from unittest import TestCase, mock

import layers

import numpy as np
import pytest


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

        conv.operate(data)
