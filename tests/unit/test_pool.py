from unittest import TestCase, mock

import layers

import numpy as np
import pytest


class TestPoolForward(TestCase):
    def test_max(self):
        pool = layers.pool.Pool(
            size=2,
            stride=2,
            operation=np.max,
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

        result = pool.forward(data)
        expected = np.zeros((2, 2, 3))
        expected[:,:,0] = np.array([
            [6, 8],
            [3, 4],
        ])
        expected[:,:,1] = np.array([
            [6, 8],
            [3, 4],
        ])
        expected[:,:,2] = np.array([
            [6, 8],
            [3, 4],
        ])
