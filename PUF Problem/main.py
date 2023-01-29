from typing import Callable, Tuple

import pypuf.io
import pypuf.simulation
from numpy import ndarray

from Genome import Genome
from Population import Population
from Functions import crossover, fitness, mutation, selection

import time

number_of_challenges = 250000
puf_length = 16
# Amount of parallel PUFs
k = 1

# Functions as parameters:
# Fitness function has to take a genome and give a value of a solution
FitnessFunc = Callable[[Genome, ndarray, ndarray], int]
# Takes population and fitness function and gives parents for the next generations population
SelectionFunc = Callable[[Population, FitnessFunc, ndarray, ndarray], Tuple[Genome, Genome]]
# Manipulating the parents and children for the next generation
CrossoverFunc = Callable[[Genome, Genome], Tuple[Genome, Genome]]
MutationFunc = Callable[[Genome], Genome]

puf = pypuf.simulation.XORArbiterPUF(n=puf_length, k=k, seed=1)

challenges = pypuf.io.random_inputs(n=puf_length, N=number_of_challenges, seed=1)
responses = puf.eval(challenges)


# noinspection DuplicatedCode
def run_evolution(
        fitness_func=FitnessFunc,
        selection_func=SelectionFunc,
        crossover_func=CrossoverFunc,
        mutation_func=MutationFunc,

        population_size=50,
        genome_length=puf_length,

        # Limits the function to return when this limit of the best value is reached
        fitness_limit=int,
        # Limits the amount of generations this function will run in case fitness limit is not reached
        generation_limit=100
) -> Tuple[Population, int]:
    population = Population(population_size, genome_length)

    for i in range(generation_limit):
        # Sort the population by fitness
        population.genomes = sorted(
            population.genomes,
            key=lambda genome: fitness_func(genome, challenges, responses),
            # Best fitness is at the start
            reverse=True
        )

        # Is the fitness limit reached?
        if fitness_func(population.genomes[0]) >= fitness_limit:
            break

        # Scuffed solution due to my implementation of the data
        next_generation = Population(0, genome_length)
        # Top 2 solutions are kept for next generation as is (elitism)
        next_generation.add_genomes(population.genomes[0:2])

        # Go through half the population since there's 2 parents. One pair is already kept as elites
        for j in range(int(len(population.genomes) / 2) - 1):
            parents = selection_func(population, fitness_func, challenges, responses)
            # Generate offsprings by crossing the parents
            offspring1, offspring2 = crossover_func(parents.genomes[0], parents.genomes[1])
            # Mutate both offsprings
            offspring1 = mutation_func(offspring1)
            offspring2 = mutation_func(offspring2)
            # Include the offsprings in the next generation
            next_generation.add_genomes([offspring1, offspring2])

        # Set the population for the next generation
        population = next_generation

    # Sort final population in case generation limit was reached
    population.genomes = sorted(
        population.genomes,
        key=lambda genome: fitness_func(genome, challenges, responses),
        reverse=True
    )

    # Return population and index to distinguish termination by fitness limit or generation limit
    return population, i
