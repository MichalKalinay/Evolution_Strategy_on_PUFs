import random


class Genome:
    # List of the values of the delays between each gate representing the structure of a PUF
    values = [float]

    def __init__(self, length: int):
        self.values = []
        for i in range(0, length):
            self.values.append(random.uniform(0, 1))

    def set_values(self, values):
        self.values = values

    def set_value_at_index(self, index, value):
        self.values[index] = value

    def __repr__(self):
        return f"{self.values}"
