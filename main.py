#! /usr/bin/env python3

import sys

import layers
from file import Data, read_data

import keras
import numpy as np


def framework(training_data, validation_data):
    training_data = Data(
        images=training_data.images,
        labels=keras.utils.to_categorical(training_data.labels),
    )
    validation_data = Data(
        images=validation_data.images,
        labels=keras.utils.to_categorical(validation_data.labels),
    )

    network = keras.models.Sequential([
        keras.layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='valid',
            data_format='channels_last',
            activation='relu',
            input_shape=(28, 28, 1)
        ),
        keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='valid',
            data_format='channels_last',
            activation='relu',
        ),
        keras.layers.MaxPooling2D(
            pool_size=(2, 2),
            strides=(2, 2),
        ),
        keras.layers.Conv2D(
            filters=16,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='valid',
            data_format='channels_last',
            activation='relu',
        ),
        keras.layers.Conv2D(
            filters=8,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='valid',
            data_format='channels_last',
            activation='relu',
        ),
        keras.layers.MaxPooling2D(
            pool_size=(2, 2),
            strides=(2, 2),
        ),
        keras.layers.Flatten(),
        keras.layers.Dense(100, activation='relu'),
        keras.layers.Dense(50, activation='relu'),
        keras.layers.Dense(10, activation='softmax'),
    ])

    network.compile(
        optimizer='sgd',
        loss='mean_squared_error',
        metrics=['accuracy'],
    )

    network.fit(
        training_data.images,
        training_data.labels,
        validation_data=validation_data,
        epochs=1,
    )


def custom(training_data, validation_data):
    image_count, image_width, image_height, image_channel = training_data.images.shape

    network = [
        layer.convolution.WindowConvolution(
            filters=np.random.randn(3, 3, 3),
            biases=[0],
            size=image_width,
            stride=1,
            padding=1,
            activation=layers.activation.ActivationType.RELU,
        ),
        layer.convolution.WindowConvolution(
            filters=np.random.randn(3, 3, 3),
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


if __name__ == '__main__':
    training_data = read_data('TRAINING')
    validation_data = read_data('VALIDATION')

    if len(sys.argv) > 2:
        custom(training_data, validation_data)
    else:
        framework(training_data, validation_data)
