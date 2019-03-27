#!/usr/bin/env python3

import keras
import numpy as np


def xor():
    network = keras.models.Sequential([
        keras.layers.Dense(
            10,
            activation='sigmoid',
            use_bias=False,
            input_shape=(2,),
        ),
        keras.layers.Dense(
            2,
            activation='sigmoid',
            use_bias=False
        ),
    ])
    network.compile(
        optimizer=keras.optimizers.SGD(lr=0.5),
        loss='mean_squared_error',
        metrics=['accuracy'],
    )

    layer0 = network.layers[0]
    layer0.set_weights([np.array([
        [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45],
        [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
    ])])
    layer1 = network.layers[1]
    layer1.set_weights([np.array([
        [0.0, 0.05],
        [0.1, 0.15],
        [0.2, 0.25],
        [0.3, 0.35],
        [0.4, 0.45],
        [0.5, 0.55],
        [0.6, 0.65],
        [0.7, 0.75],
        [0.8, 0.85],
        [0.9, 0.95],
    ])])

    print_weights = keras.callbacks.LambdaCallback(on_epoch_end=lambda batch, logs: print(layer0.get_weights()[0], layer1.get_weights()[0]))

    network.fit(
        np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
        np.array([[1, 0], [0, 1], [0, 1], [1, 0]]),
        batch_size=1,
        epochs=1,
        callbacks=[print_weights]
    )


if __name__ == '__main__':
    xor()
