#! /usr/bin/env python3

import layers
from file import read_data

import numpy as np


if __name__ == '__main__':
    data = read_data('TRAINING')

    image_count, image_width, image_height, image_channel = data.images.shape

    network = [
        layer.convolution.WindowConvolution(
            filters=np.random.randn((3, 3, 3)),
            biases=[0],
            size=image_width,
            stride=1,
            padding=1,
            activation=layers.activation.ActivationType.RELU,
        ),
        layer.convolution.WindowConvolution(
            filters=np.random.randn((3, 3, 3)),
            biases=[0],
            size=image_width,
            stride=1,
            padding=1,
            activation=layers.activation.ActivationType.RELU,
        ),
        layer.pool.Pool(
            size=2,
            stride=2,
            operation=layers.pool.PoolOperation.MAX,
        ),
        layer.fully_connected.FC(),
        layer.fully_connected.FC(),
        layer.fully_connected.Output(),
    ]
