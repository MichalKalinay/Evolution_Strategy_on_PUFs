# Testing different attacks on (XOR Arbiter) PUFs before tackling it with Evolution Strategy

import pypuf.io
import pypuf.simulation
import pypuf.attack
import pypuf.metrics

import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

import time


Pypuf_attack = True
Sklearn_attack = True

number_of_challenges = 250000
puf_length = 64

print("Generating PUF, Challenges and Responses... ", end="")
puf = pypuf.simulation.XORArbiterPUF(n=puf_length, k=5, seed=1)
crps = pypuf.io.ChallengeResponseSet.from_simulation(puf, N=number_of_challenges, seed=1)

challenges_manual = pypuf.io.random_inputs(n=puf_length, N=number_of_challenges, seed=1)
responses_manual = puf.eval(challenges_manual)
print("done!")

# pypuf MLP attack
if Pypuf_attack:
    slice = int(len(crps) * 0.7)
    Train_data = crps[0:slice]
    Test_data = crps[slice:]

    start_time = time.time()
    attack = pypuf.attack.MLPAttack2021(Train_data, seed=3, net=[2 ** 4, 2 ** 5, 2 ** 4],
                                        epochs=100, lr=.001, bs=1000, early_stop=.08)
    attack.fit()
    model = attack.model
    end_time = time.time()

    # Evaluate attack model on test data (and transform it into a numpy array for later comparison)
    Test_results = np.array(model.eval(Test_data.challenges).flatten())

    # This is a bad version of evaluating the success of the attack. Using a data and test set is better
    # NOTE: In this version this is evaluated only using Train_data
    pypuf_score_similarity = pypuf.metrics.similarity(puf, model, seed=1)

    # Evaluation using a training and test set
    pypuf_score = 1 - np.mean(Test_results != np.array(Test_data.information.flatten()))

    # Timing of attack and fit
    pypuf_time = end_time - start_time

# sklearn MLP attack
if Sklearn_attack:
    # Probably transformed data by pypuf lib?
    # Challenges, Responses = crps.challenges, crps.information.flatten()

    # This should be raw, not previously transformed data
    Challenges, Responses = challenges_manual, responses_manual
    Ch_train, Ch_test, Res_train, Res_test = train_test_split(Challenges, Responses, stratify=Responses, random_state=1)
    start_time = time.time()
    clf = MLPClassifier(random_state=1, max_iter=100, early_stopping=True, verbose=True)
    clf.fit(Ch_train, Res_train)
    end_time = time.time()
    clf.predict_proba(Ch_test[:1])
    clf.predict(Ch_test[:5, :])

    sklearn_score = clf.score(Ch_test, Res_test)

    # Timing of attack and fit
    sklearn_time = end_time - start_time


if Pypuf_attack:
    print("Score of pypuf MLP Attack (data / test set): " + str(pypuf_score))
    # Since data previously split into training and test set, this only applies to the training set
    print("Score of pypuf MLP Attack (similarity; train set only): " + str(pypuf_score_similarity))
    print("And processing time of: " + str(pypuf_time))
if Sklearn_attack:
    print("Score of sklearn MLP Attack (data / test set): " + str(sklearn_score))
    print("And processing time of: " + str(sklearn_time))
