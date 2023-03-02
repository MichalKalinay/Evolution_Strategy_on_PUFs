from typing import List

import numpy as np

from Genome import Genome


def fitness(genome: Genome, crps: List[np.array]) -> int:
    correct_responses = 0

    for i, crp in enumerate(crps):
        # Separate challenges and true responses
        challenge = crp[0]
        response = crp[1]
        # Calculate responses of genome using a dot product over its values and challenges. Take the sign of that.
        genome_response = int(np.sign(np.array(challenge).dot(genome.values)))
        if genome_response == response:
            correct_responses += 1

    # Returns amount of correct responses from the genome
    return correct_responses
