from pypuf.simulation import XORArbiterPUF
from pypuf.io import random_inputs


puf = XORArbiterPUF(n=16, k=1, seed=1)
challenges = random_inputs(n=16, N=10, seed=1)

print(puf.eval(challenges))
