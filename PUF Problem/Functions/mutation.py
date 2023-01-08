from random import randrange, random

from Genome import Genome


def mutation_function(genome: Genome, mutations: int = 1, probability: float = 0.5) -> Genome:
    for _ in range(mutations):
        # Choose a random index
        index = randrange(len(genome.values))
        # Generate a new value for the float at the index
        if random() <= probability:
            genome.values[index] = (random.uniform(0, 1), random.uniform(0, 1))

    # Mutated genome
    return genome
