import numpy as np


class Genome:
    # List of the values of the delays between each gate representing the structure of a PUF
    values = []

    def __init__(self, length: int):
        for i in range(0, length):
            self.values = np.random.normal(loc=0.0, scale=1.0, size=length)

    def set_values(self, values):
        self.values = values

    def set_value_at_index(self, index, value):
        self.values[index] = value

    def __repr__(self):
        return f"{self.values}"
