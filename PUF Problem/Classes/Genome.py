import random
from typing import List, Tuple


class Genome:
    # List of the values of the delays between each gate representing the structure of a PUF
    # Subject to be changed
    values = List[Tuple[float]]

    def __init__(self, length: int):
        self.values = []
        for i in range(0, length):
            gate = (random.uniform(0, 1), random.uniform(0, 1))
            self.values.append(gate)

    def __repr__(self):
        return f"{self.values}"

