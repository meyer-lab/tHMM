""" To plot a summary of cross validation. """
from .common import getSetup
import numpy as np
import itertools as it
from ..LineageTree import LineageTree
from ..states.StateDistributionGaPhs import StateDistribution
from ..crossval import hide_observation, crossval

desired_num_states = np.arange(1, 7)
Sone = StateDistribution(0.99, 0.9, 100, 0.1, 50, 0.2)
Stwo = StateDistribution(0.9, 0.7, 100, 0.2, 50, 0.4)
Sthree = StateDistribution(0.85, 0.9, 100, 0.3, 50, 0.6)
Sfour = StateDistribution(0.8, 0.9, 100, 0.4, 50, 0.8)
Sfive = StateDistribution(0.75, 0.85, 100, 0.5, 50, 1.0)

Etwo = [Sone, Stwo]
Ethree = [Sone, Stwo, Sthree]
Efour = [Sone, Stwo, Sthree, Sfour]
Efive = [Sone, Stwo, Sthree, Sfour, Sfive]
Es = [Etwo, Ethree, Efour, Efive]

def makeFigure():
    """
    Makes figure 19.
    """
    ax, f = getSetup((16, 4), (1, 4))

    output = []
    for e in Es:
        pi = np.ones(len(e)) / len(e)
        T = (np.eye(len(e)) + 0.1)
        T = T / np.sum(T, axis=1)[:, np.newaxis]
        complete_population = [[LineageTree.init_from_parameters(pi, T, e, 7, censored_condition=3, desired_experiment_time=200) for _ in range(100)] for _ in range(4)]

        output.append(output_LL(complete_population))

    for i in range(4):
        ax[i].plot(desired_num_states, output[i])
        ax[i].set_title(str(i+2) + " state model")
        ax[i].set_ylim(bottom=0, top=max(list(it.chain(*output))) + 2)
        ax[i].set_ylabel("Likelihood")
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
