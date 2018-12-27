from unittest import TestCase, mock

import numpy as np
import pytest

from .. import layer


class TestConvolutionInitialization(TestCase):
    def test_works(self):
        conv = layer.Convolution(
            np.zeros((1, 3, 3)),
            np.zeros((3,)),
            3,
            1,
            1,
            mock.Mock()
        )
