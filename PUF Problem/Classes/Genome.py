import numpy as np


class Genome:
    # List of the values of the delays between each gate representing the structure of a PUF
    values = []

    def __init__(self, length: int):
        # TODO Check if correct
        # Random gaussian distribution of values between -1 and 1 for each gate of the genome representing a PUF
        self.values = np.random.normal(loc=0.0, scale=1.0, size=length)

    def set_values(self, values):
        self.values = values

    def set_value_at_index(self, index, value):
        self.values[index] = value

    def __repr__(self):
        return f"{self.values}"
