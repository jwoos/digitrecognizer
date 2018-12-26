'''
Module to read in files from http://yann.lecun.com/exdb/mnist/.
The data format is described at the bottom of the page.
'''

import numpy as np


def read_labels(path: str) -> np.ndarray:
    with open(path, 'rb') as f:
        data = f.read()

    magic_numer = data[:4]
    item_count = data[4:8]

    # remove header information
    data = data[8:]

    return np.array(list(data))


def read_images(path: str) -> np.ndarray:
    with open(path, 'rb') as f:
        data = f.read()

    magic_numer = data[:4]
    item_count = int.from_bytes(data[4:8], byteorder='big', signed=False)
    row_count = int.from_bytes(data[8:12], byteorder='big', signed=False)
    column_count = int.from_bytes(data[12:16], byteorder='big', signed=False)

    # remove header information
    data = data[16:]

    images = np.zeros((item_count, row_count, column_count))

    for i in range(item_count):
        image = images[i]
        base = i * row_count * column_count

        for row in range(row_count):
            offset = base + (column_count * row)
            image[row][:column_count] = list(data[offset:offset+column_count])

    return images
