""" To plot a summary of cross validation. """
from .common import getSetup
import numpy as np
from ..LineageTree import LineageTree
from ..crossval import hide_observation, crossval
from ..BaumWelch import calculate_stationary
from .figure18 import state0, state1, state2, state3, state4

desired_num_states = np.arange(1, 8)

Etwo = [state0, state1]
Ethree = [state0, state1, state2]
Efour = [state0, state1, state2, state3]
Efive = [state0, state1, state2, state3, state4]
Es = [Etwo, Ethree, Efour, Efive]


def makeFigure():
    """
    Makes figure 19.
    """
    ax, f = getSetup((16, 4), (1, 4))

    output = []
    for e in Es:
        T = np.eye(len(e)) + 0.1
        T = T / np.sum(T, axis=1)[:, np.newaxis]
        pi = calculate_stationary(T)
        complete_population = [
            [LineageTree.rand_init(pi, T, e, 7, censored_condition=3, desired_experiment_time=200) for _ in range(100)] for _ in range(4)
        ]

        output.append(output_LL(complete_population))

    for i in range(4):
        ax[i].plot(desired_num_states, output[i])
        ax[i].set_title(str(i + 2) + " state population")
        ax[i].set_ylabel("Log-likelihood")
        ax[i].set_xlabel("Number of States")

    return f


def output_LL(complete_population):
    # create training data by hiding 20% of cells in each lineage
    train_population = [hide_observation(complete_pop, 0.25) for complete_pop in complete_population]
    # Copy out data to full set
    dataFull = []
    for _ in desired_num_states:
        dataFull.append(train_population)

    return crossval(dataFull, desired_num_states)
