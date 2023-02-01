import numpy as np
from numpy import ndarray

from Genome import Genome


def fitness(genome: Genome, challenges: ndarray, responses: ndarray) -> int:
    correct_responses = 0
    genome_responses = evaluate(genome, challenges)
    if len(genome_responses) != len(responses):
        raise ValueError("Calculated responses are not the same size as provided responses!")

    for i in range(len(responses)):
        if genome_responses[i] == responses[i]:
            correct_responses += 1

    return correct_responses


def evaluate(genome: Genome, challenges: ndarray):
    responses = []
    for challenge in challenges:
        responses.append(int(np.sign(challenge.dot(genome.values))))
    return responses
