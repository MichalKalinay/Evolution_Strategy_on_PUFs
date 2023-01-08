from random import choices

from fitness import fitness
from Population import Population


# Select the fittest 2 Genomes from a given Population to be the parents for the next generation
def select_pair(population: Population, fitness_func: fitness) -> Population:
    result = Population(0, 0)
    # Solutions with a higher fitness value will be more likely to be chosen
    result.genomes.append(choices(
        population=population.genomes,
        # Weight of the choice is the fitness
        weights=[fitness_func(genome) for genome in population.genomes],
        # Select 2 parents
        k=2
    ))
    return result
