from typing import List

import numpy as np
from numpy import ndarray

from Genome import Genome


def fitness(genome: Genome, crps: List[np.array]) -> int:
    correct_responses = 0

    for i, crp in enumerate(crps):
        response = int(np.sign(np.array(crp[0]).dot(genome.values)))
        if response == crp[1]:
            correct_responses += 1

    return correct_responses
