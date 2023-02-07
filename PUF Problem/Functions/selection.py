from random import choices
from typing import Callable, List

import numpy as np

from Genome import Genome
from Population import Population

FitnessFunc = Callable[[Genome, List[np.array]], int]


# Select the fittest 2 Genomes from a given Population to be the parents for the next generation
def select_pair(population: Population, fitness_func: FitnessFunc,
                crps: List[np.array]) -> Population:
    result = Population(0, 0)
    # Solutions with a higher fitness value will be more likely to be chosen
    result.add_genomes(choices(
        population=population.genomes,
        # Weight of the choice is the fitness
        weights=[fitness_func(genome, crps) for genome in population.genomes],
        # Select 2 parents
        k=2
    ))
    return result
