from typing import List

from Genome import Genome


class Population:
    genomes = List[Genome]

    def __init__(self, size: int, genome_length: int):
        self.genomes = []
        for i in range(0, size):
            genome = Genome(genome_length)
            self.genomes.append(genome)

    def __repr__(self):
        return f"{self.genomes}"
