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
