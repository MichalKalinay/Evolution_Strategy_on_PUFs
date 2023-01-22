import random
from typing import List


class Genome:
    # List of the values of the delays between each gate representing the structure of a PUF
    values = List[float]

    def __init__(self, length: int):
        self.values = []
        for i in range(0, length):
            self.values.append(random.uniform(0, 1))

    def __repr__(self):
        return f"{self.values}"
