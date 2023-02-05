from random import randrange, random

import numpy as np

from Genome import Genome


# Mutate one genome by giving it @mutations many new values each with a @probability
def mutation_function(genome: Genome, mutations: int = 1, probability: float = 0.5) -> Genome:
    for _ in range(mutations):
        # Choose a random index
        index = randrange(len(genome.values))
        # Generate a new value for the float at the index
        if random() <= probability:
            genome.set_value_at_index(index, np.random.normal(loc=0.0, scale=1.0, size=1))

    # Mutated genome
    return genome
