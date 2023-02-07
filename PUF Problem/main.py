import random
from typing import Callable, Tuple, List

import numpy as np
import pypuf.io
import pypuf.simulation

from Genome import Genome
from Population import Population
from Functions import crossover, fitness, mutation, selection

import time

number_of_challenges = 10000
puf_length = 16
# Amount of parallel PUFs
k = 1

# Functions as parameters:
# Fitness function has to take a genome and give a value of a solution
FitnessFunc = Callable[[Genome, List[np.array]], int]
# Takes population and fitness function and gives parents for the next generations population
SelectionFunc = Callable[[Population, FitnessFunc, List[np.array]], Tuple[Genome, Genome]]
# Manipulating the parents and children for the next generation
CrossoverFunc = Callable[[Genome, Genome], Tuple[Genome, Genome]]
MutationFunc = Callable[[Genome, int], Genome]

puf = pypuf.simulation.XORArbiterPUF(n=puf_length, k=k, seed=1)

challenges_pre_trans = pypuf.io.random_inputs(n=puf_length, N=number_of_challenges, seed=1)
responses = puf.eval(challenges_pre_trans)

# Transformation of challenges like in the Lin paper
challenges_transformed = np.fliplr(challenges_pre_trans)
challenges_transformed = np.cumprod(challenges_transformed, axis=1)
challenges_transformed = np.fliplr(challenges_transformed)
ones = np.ones((len(challenges_pre_trans), 1))
challenges = np.hstack((challenges_transformed, ones))

# Zipping of crps so later a batch for each generation can be randomly selected
zip_challenge_response = list(zip(challenges, responses))


# noinspection DuplicatedCode
def run_evolution(
        fitness_func=FitnessFunc,
        selection_func=SelectionFunc,
        crossover_func=CrossoverFunc,
        mutation_func=MutationFunc,

        population_size=50,
        genome_length=puf_length,

        # Limits the function to return when this accuracy is reached
        fitness_limit=0.9,
        # Limits the amount of generations this function will run in case fitness limit is not reached
        generation_limit=50,
        crps_per_generation=2000
) -> Tuple[Population, int]:
    start_time = time.time()
    population = Population(population_size, genome_length+1)

    for i in range(generation_limit):
        generation_start_time = time.time()
        print(f"Running generation {i} ... ", end="")

        # Generate challenge-response-pairs to be used for this generation
        crps_generation = random.choices(zip_challenge_response, k=crps_per_generation)

        # Sort the population by fitness
        population.genomes = sorted(
            population.genomes,
            key=lambda genome: fitness_func(genome, crps_generation),
            # Best fitness is at the start
            reverse=True
        )

        # Is the fitness limit reached?
        best_fit = fitness_func(population.genomes[0], crps_generation) / crps_per_generation
        print(f"(Best fitness so far: {best_fit})")
        if best_fit >= fitness_limit:
            early_break_time = time.time()
            print(f"... done! Processing time of generation {i}: {early_break_time - generation_start_time} sec")
            break

        # Scuffed solution due to my implementation of the data
        next_generation = Population(0, 0)
        # Top 2 solutions are kept for next generation as is (elitism)
        next_generation.add_genomes(population.genomes[0:2])

        # Go through half the population since there's 2 parents. One pair is already kept as elites
        for j in range(int(len(population.genomes) / 2) - 1):
            parents = selection_func(population, fitness_func, crps_generation)
            # Generate offsprings by crossing the parents
            offspring1, offspring2 = crossover_func(parents.genomes[0], parents.genomes[1])
            # Higher accuracy -> fewer mutations
            mutations_amount = int((genome_length-(genome_length*best_fit))/2)
            # Mutate offsprings
            offspring1 = mutation_func(offspring1, mutations_amount)
            offspring2 = mutation_func(offspring2, mutations_amount)
            # Include the offsprings in the next generation
            next_generation.add_genomes([offspring1, offspring2])

        # Set the population for the next generation
        population = next_generation

        generation_end_time = time.time()
        print(f"Processing time of generation {i}: {generation_end_time - generation_start_time} sec")

    # Sort final population in case generation limit was reached
    population.genomes = sorted(
        population.genomes,
        key=lambda genome: fitness_func(genome, zip_challenge_response),
        # Best fitness is at the start
        reverse=True
    )

    end_time = time.time()

    # Return population and index to distinguish termination by fitness limit or generation limit
    print(f"... done! Processing time of: {end_time - start_time} sec")
    return population, i


# Run The Evolution
population, generations = run_evolution(
    fitness_func=fitness.fitness,
    selection_func=selection.select_pair,
    crossover_func=crossover.single_point_crossover,
    mutation_func=mutation.mutation_function,
)
print(f"Number of generations: {generations+1}")
print(f"Best solution: {population.genomes[0]}")
print(f"And its accuracy: {fitness.fitness(population.genomes[0], zip_challenge_response) / number_of_challenges}")
