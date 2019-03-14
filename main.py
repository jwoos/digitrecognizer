#! /usr/bin/env python3

import layers
from file import read_data

import numpy as np


def main():
    data = read_data('TRAINING')
    network = [
        layer.convolution.WindowConvolution(),
        layer.convolution.WindowConvolution(),
        layer.pool.Pool(),
        layer.convolution.WindowConvolution(),
        layer.convolution.WindowConvolution(),
        layer.pool.Pool(),
        layer.fully_connected.FC(),
        layer.fully_connected.FC(),
        layer.fully_connected.Output(),
    ]


if __name__ == '__main__':
    main()
