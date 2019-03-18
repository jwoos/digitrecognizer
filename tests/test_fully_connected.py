from unittest import TestCase, mock

import layers

import numpy as np
import pytest


class TestFullyConnectedInitialization(TestCase):
    def test_constructor(self):
        units = 10
        fc = layers.fully_connected.FC(units)

        self.assertEqual(fc.units, units)

    def test_initialization(self):
        input_shape = (100,)
        output_shape = (10,)
        fc = layers.fully_connected.FC(output_shape[0])
        fc.initialize(input_shape)

        self.assertEqual(fc.input_shape, input_shape)
        self.assertEqual(fc.output_shape, output_shape)
        self.assertTrue(np.array_equal(fc.biases, np.zeros(output_shape)))

    def test_initialization_with_bias(self):
        input_shape = (100,)
        output_shape = (10,)
        fc = layers.fully_connected.FC(output_shape[0], use_biases=True)
        fc.initialize(input_shape)

        self.assertEqual(fc.input_shape, input_shape)
        self.assertEqual(fc.output_shape, output_shape)
        self.assertFalse(np.array_equal(fc.biases, np.zeros(output_shape)))


class TestFullyConnectedForward(TestCase):
    def test_same_shape(self):
        fc = layers.fully_connected.FC(10)
        fc.initialize((10,))

        data = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        fc.weights = np.zeros((10, 10))
        fc.weights[:,0] = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

        result = fc.forward(data)

        expected = np.array([10, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        self.assertTrue(np.array_equal(result, expected))

    def test_small_big(self):
        fc = layers.fully_connected.FC(10)
        fc.initialize((1,))

        data = np.array([1])
        fc.weights = np.zeros((1, 10))
        fc.weights[0,:] = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

        result = fc.forward(data)

        expected = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

        self.assertTrue(np.array_equal(result, expected))

    def test_big_small(self):
        fc = layers.fully_connected.FC(1)
        fc.initialize((10,))

        data = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        fc.weights = np.zeros((10, 1))
        fc.weights[:,0] = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

        result = fc.forward(data)

        expected = np.array([10])

        self.assertTrue(np.array_equal(result, expected))
