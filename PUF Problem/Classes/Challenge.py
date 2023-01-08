from random import choices
from typing import List


class Challenge:
    # List of 1 and -1 representing which path to choose at each gate
    values = List[int]

    def __init__(self, length: int):
        self.values = choices([-1, 1], k=length)

    def __repr__(self):
        return f"{self.values}"

