from random import randint
from typing import Tuple

import numpy as np

from Genome import Genome


# Crossover between 2 genomes to generate 2 new genomes based on them for the next generation
def single_point_crossover(parent1: Genome, parent2: Genome) -> Tuple[Genome, Genome]:
    # For this use-case the genomes have to be the same length
    if len(parent1.values) != len(parent2.values):
        raise ValueError("Parents must be the same size for a crossover")

    # Since both parents are the same size it doesn't matter which one we take the length from
    length = len(parent1.values)
    # The genomes should have a length of at least 2 for a crossover to work
    if length < 2:
        return parent1, parent2

    # Perform a crossover between the parents; Slicing them randomly in 2 parts and merging them
    slice = randint(1, length - 1)

    # Results
    selection1 = Genome(0)
    selection2 = Genome(0)
    selection1.set_values(np.concatenate((parent1.values[0:slice], parent2.values[slice:]), axis=None))
    selection2.set_values(np.concatenate((parent2.values[0:slice], parent1.values[slice:]), axis=None))

    return selection1, selection2
