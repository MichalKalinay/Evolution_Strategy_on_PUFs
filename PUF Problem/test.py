# Testing different attacks on (XOR Arbiter) PUFs before tackling it with Evolution Strategy

import pypuf.io
import pypuf.simulation

import pypuf.attack

import pypuf.metrics

puf = pypuf.simulation.XORArbiterPUF(n=16, k=1, seed=1)
crps = pypuf.io.ChallengeResponseSet.from_simulation(puf, N=500000, seed=1)

attack = pypuf.attack.MLPAttack2021(crps, seed=3, net=[2 ** 4, 2 ** 5, 2 ** 4],
                                    epochs=30, lr=.001, bs=1000, early_stop=.08)

attack.fit()
model = attack.model

# This is a bad version of evaluating the success of the attack. Using a data and test set is better
print(pypuf.metrics.similarity(puf, model, seed=1))
