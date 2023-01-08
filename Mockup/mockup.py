# This will be the solving the knapsack problem using ML Evolution Strategy (a genetic algorithm)

# This is based off of a tutorial from Kie Codes: https://www.youtube.com/watch?v=nhT56blfRpE

from collections import namedtuple
from functools import partial
from random import choices, randint, randrange, random
from typing import List, Callable, Tuple

# Genome is a list of either 1 or 0 representing wether an item is included in the solution or not
Genome = List[int]
Population = List[Genome]

# Functions as parameters:
# Fitness function has to take a genome and give a value of a solution
FitnessFunc = Callable[[Genome], int]
PopulateFunc = Callable[[], Population]
# Takes population and fitness function and gives parents for the next generations population
SelectionFunc = Callable[[Population, FitnessFunc], Tuple[Genome, Genome]]
# Manipulating the parents and children for the next generation
CrossoverFunc = Callable[[Genome, Genome], Tuple[Genome, Genome]]
MutationFunc = Callable[[Genome], Genome]

# An Item represents any kind of object that could be used to fill for the knapsack problem
Item = namedtuple("Item", ["name", "weight", "value"])

# Hardcoded values for the sake of simplicity of the mockup (randomly generated)
items = [
    Item("Laptop", 2200, 500),
    Item("Headphones", 160, 150),
    Item("Coffee Mug", 350, 60),
    Item("Notepad", 333, 40),
    Item("Water Bottle", 192, 30),
    Item("Mints", 25, 5),
    Item("Socks", 38, 10),
    Item("Tissues", 80, 15),
    Item("Phone", 200, 500),
    Item("Baseball Cap", 70, 100),
]


# Run one evolution to solve this specific knapsack problem
def runEvolution(
        populate_func=PopulateFunc,
        fitness_func=FitnessFunc,
        selection_func=SelectionFunc,
        crossover_func=CrossoverFunc,
        mutation_func=MutationFunc,
        # Limits the function to return when this limit of the best value is reached
        fitness_limit=int,
        # Limits the amount of generations this function will run in case fitness limit is not reached
        generation_limit=100
) -> Tuple[Population, int]:
    population = populate_func()

    for i in range(generation_limit):
        # Sort the population by fitness
        population = sorted(
            population,
            key=lambda genome: fitness_func(genome),
            # Best fitness is at the start
            reverse=True
        )

        # Is the fitness limit reached?
        if fitness_func(population[0]) >= fitness_limit:
            break

        # Top 2 solutions are kept for next generation as is (elitism)
        next_generation = population[0:2]

        # Go through half the population since there's 2 parents. One pair is already kept as elites
        for j in range(int(len(population) / 2) - 1):
            parents = selection_func(population, fitness_func)
            # Generate offsprings by crossing the parents
            offspring1, offspring2 = crossover_func(parents[0], parents[1])
            # Mutate both offsprings
            offspring1 = mutation_func(offspring1)
            offspring2 = mutation_func(offspring2)
            # Include the offsprings in the next generation
            next_generation += [offspring1, offspring2]

        # Set the population for the next generation
        population = next_generation

    # Sort final population in case generation limit was reached
    population = sorted(
        population,
        key=lambda genome: fitness_func(genome),
        reverse=True
    )

    # Return population and index to distinguish termination by fitness limit or generation limit
    return population, i


# Generation functions:
def generateGenome(length: int) -> Genome:
    # List of randomly generated 1 and 0
    return choices([0, 1], k=length)


def generatePopulation(size: int, genome_length: int) -> Population:
    return [generateGenome(genome_length) for _ in range(size)]


# Calculate a solution for the problem
def fitness(genome: Genome, items: [Item], weight_limit: int) -> int:
    # Since a genome represents the inclusion for all items their size must be the same
    if len(genome) != len(items):
        raise ValueError("Genome and items must be the same size")

    # The current weight and value of the items we are taking into consideration for this solution
    weight = 0
    value = 0

    for i, item in enumerate(items):
        if genome[i] == 1:
            weight += item.weight
            value += item.value
            # Invalid if the solution exceeds the limit
            if weight > weight_limit:
                return 0

    # Return the value of the solution. The higher the value the better the guess.
    return value


# Select a pair of solutions to be the parents for the next generation
def selectPair(population: Population, fitness_func: FitnessFunc) -> Population:
    # Solutions with a higher fitness value will be more likely to be chosen
    return choices(
        population=population,
        # Weight of the choice is the fitness
        weights=[fitness_func(genome) for genome in population],
        # Select 2 parents
        k=2
    )


# Crossover between 2 genomes to generate 2 new genomes based on them for the next generation
def singlePointCrossover(parent1: Genome, parent2: Genome) -> Tuple[Genome, Genome]:
    # For this use-case the genomes have to be the same length
    if len(parent1) != len(parent2):
        raise ValueError("Parents must be the same size for a crossover")

    # Since both parents are the same size it doesn't matter which one we take the length from
    length = len(parent1)
    # The genomes should have a length of at least 2 for a crossover to work
    if length < 2:
        return parent1, parent2

    # Perform a crossover between the parents; Slicing them randomly in 2 parts and merging them
    slice = randint(1, length - 1)
    return parent1[0:slice] + parent2[slice:], parent2[0:slice] + parent1[slice:]


# Randomly mutate a genome for the next generation
def mutation(genome: Genome, mutations: int = 1, probability: float = 0.5) -> Genome:
    for _ in range(mutations):
        # Choose a random index
        index = randrange(len(genome))
        # Flip the "bit" of the representation of the inclusion of an item in the genome
        genome[index] = genome[index] if random() > probability else abs(genome[index] - 1)

    # Mutated genome
    return genome


# Run The Evolution
population, generations = runEvolution(
    # In this example the "populate" and "fitness" functions differ from runEvolutions params -> partial
    populate_func=partial(
        generatePopulation, size=10, genome_length=len(items)
    ),
    fitness_func=partial(
        # Arbitrary value for weight limit
        fitness, items=items, weight_limit=3000
    ),
    selection_func=selectPair,
    crossover_func=singlePointCrossover,
    mutation_func=mutation,
    # Arbitrary value for fitness limit
    fitness_limit=1310,
    generation_limit=100
)


def sumOfBest(genome: Genome, items: [Item]) -> [int]:
    result = [0, 0]
    for i, item in enumerate(items):
        if genome[i] == 1:
            result[0] += item.value
            result[1] += item.weight

    return result


# Function for a nicer print to console
def genomeToItems(genome: Genome, items: [Item]) -> [Item]:
    result = []
    for i, item in enumerate(items):
        if genome[i] == 1:
            result += [item.name]

    return result


print(f"Number of generations: {generations}")
print(f"Best solution: {genomeToItems(population[0], items)}")
print(f"And its value: {sumOfBest(population[0], items)[0]} and weight of: {sumOfBest(population[0], items)[1]}")
