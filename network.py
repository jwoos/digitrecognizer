from typing import List


class Network:
    def __init__(
        self,
        layers: List,
        epochs: int,
        learning_rate: float,
    ):
        self.layers = layers
        self. epochs = epochs
        self.learning_rate = learning_rate

    def run(self):
        pass
