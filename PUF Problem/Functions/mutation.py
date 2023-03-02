from random import randrange, random

import numpy as np

from Genome import Genome


# Mutate one genome by giving it @mutations many new values each with a @probability
def new_value(genome: Genome, mutations: int = 1, probability: float = 0.5) -> Genome:
    for _ in range(mutations):
        # Choose a random index
        index = randrange(len(genome.values))
        # Generate a new value for the float at the index
        if random() <= probability:
            genome.set_value_at_index(index, np.random.normal(loc=0.0, scale=1.0, size=1))

    # Mutated genome
    return genome


def deviate_constant(genome: Genome, mutations: int = 1, probability: float = 0.5) -> Genome:
    for _ in range(mutations):
        # Choose a random index
        index = randrange(len(genome.values))
        # Deviation from the current value
        deviation = 0.5
        # Add / subtract the deviation from the float at the index
        if random() <= probability:
            if random() > 0.5:
                genome.set_value_at_index(index, genome.values[index] + deviation)
            else:
                genome.set_value_at_index(index, genome.values[index] - deviation)

    # Mutated genome
    return genome


def deviate_gaussian(genome: Genome, mutations: int = 1, probability: float = 0.5) -> Genome:
    for _ in range(mutations):
        # Choose a random index
        index = randrange(len(genome.values))
        # Deviation from the current value (between -0.5 and 0.5)
        deviation = np.random.normal(loc=0.0, scale=0.5, size=1)
        # Add / subtract the deviation from the float at the index
        if random() <= probability:
            genome.set_value_at_index(index, genome.values[index] + deviation)

    # Mutated genome
    return genome


def deviate_percentage(genome: Genome, mutations: int = 1, probability: float = 0.5) -> Genome:
    for _ in range(mutations):
        # Choose a random index
        index = randrange(len(genome.values))
        # Deviation from the current value (percentage)
        deviation = 0.5
        # Multiply / divide the float with the deviation at the index
        if random() <= probability:
            if random() > 0.5:
                genome.set_value_at_index(index, genome.values[index] * deviation)
            else:
                genome.set_value_at_index(index, genome.values[index] / deviation)

    # Mutated genome
    return genome
