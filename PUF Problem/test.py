# Testing different attacks on (XOR Arbiter) PUFs before tackling it with Evolution Strategy

import pypuf.io
import pypuf.simulation
import pypuf.attack
import pypuf.metrics

import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


Pypuf_attack = True
Sklearn_attack = True

number_of_challenges = 250000

puf = pypuf.simulation.XORArbiterPUF(n=16, k=1, seed=1)
crps = pypuf.io.ChallengeResponseSet.from_simulation(puf, N=number_of_challenges, seed=1)

challenges_manual = pypuf.io.random_inputs(n=16, N=number_of_challenges, seed=1)
responses_manual = puf.eval(challenges_manual)

# pypuf MLP attack
if Pypuf_attack:
    slice = int(len(crps) * 0.7)
    Train_data = crps[0:slice]
    Test_data = crps[slice:]

    attack = pypuf.attack.MLPAttack2021(Train_data, seed=3, net=[2 ** 4, 2 ** 5, 2 ** 4],
                                        epochs=30, lr=.001, bs=1000, early_stop=.08)
    attack.fit()
    model = attack.model

    # Evaluate attack model on test data (and transform it into a numpy array for later comparison)
    Test_results = np.array(model.eval(Test_data.challenges).flatten())

    # This is a bad version of evaluating the success of the attack. Using a data and test set is better
    # NOTE: In this version this is evaluated only using Train_data
    pypuf_score_similarity = pypuf.metrics.similarity(puf, model, seed=1)

    # Evaluation using a data and a test set
    pypuf_score = 1 - np.mean(Test_results != np.array(Test_data.information.flatten()))

# sklearn MLP attack
if Sklearn_attack:
    # Probably transformed data by pypuf lib?
    # X, y = crps.challenges, crps.information.flatten()

    # This should be raw, not previously transformed data
    X, y = challenges_manual, responses_manual
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=1)
    clf = MLPClassifier(random_state=1, max_iter=100, early_stopping=True, verbose=True)
    clf.fit(X_train, y_train)
    clf.predict_proba(X_test[:1])
    clf.predict(X_test[:5, :])

    sklearn_score = clf.score(X_test, y_test)


if Pypuf_attack:
    print("Score of pypuf MLP Attack (data / test set): " + str(pypuf_score))
    # Since data and test set previously split this only applies to the training set
    print("Score of pypuf MLP Attack (similarity; train set only): " + str(pypuf_score_similarity))
if Sklearn_attack:
    print("Score of sklearn MLP Attack (data / test set): " + str(sklearn_score))