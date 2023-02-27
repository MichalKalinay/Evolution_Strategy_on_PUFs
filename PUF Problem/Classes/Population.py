from typing import List

from Genome import Genome


class Population:
    genomes = []

    def __init__(self, size: int, genome_length: int):
        self.genomes = []
        # Edge case for when generating next generation for the first time as an empty Population
        if size == 0 or genome_length == 0:
            return

        for i in range(0, size):
            genome = Genome(genome_length)
            self.genomes.append(genome)

    def add_genomes(self, genomes_to_add: List[Genome]):
        self.genomes += genomes_to_add

    def __repr__(self):
        return f"{self.genomes}"
