"""
File: figure5.py
Purpose: Generates figure 5.

AIC.
"""
import numpy as np
from matplotlib.ticker import MaxNLocator

from .figureCommon import getSetup
from ..Analyze import getAIC, run_Analyze_over
from ..LineageTree import LineageTree
from ..StateDistribution import StateDistribution


def makeFigure():
    """
    Makes figure 11.
    """
    ax, f = getSetup((7, 3), (1, 3))

    figure_maker(ax[0], *AIC_increased_cells1())
    figure_maker(ax[1], *AIC_increased_cells2())
    figure_maker(ax[2], *AIC_increased_cells3())

    return f


def run_AIC(Trate, E, num_to_evaluate=10):
    # Normalize the transition matrix
    T = Trate + np.eye(len(E))
    T = T / np.sum(T, axis=0)[np.newaxis, :]

    # pi: the initial probability vector
    # make an even starting p
    pi = np.ones(T.shape[0]) / T.shape[0]

    # States to evaluate with the model
    desired_num_states = [1, 2, 3]

    list_of_populations = []
    for idx in range(num_to_evaluate):
        # Creating an unpruned and pruned lineage
        list_of_populations.append([LineageTree(pi, T, E, (2**8) - 1, 1E9, prune_condition='fate', prune_boolean=False)])

    AIC_holder = np.empty((len(desired_num_states), num_to_evaluate))
    for ii, num_states_to_evaluate in enumerate(desired_num_states):
        # Analyze the lineages in the list of populations
        output = run_Analyze_over(list_of_populations, num_states_to_evaluate)
        # Collecting the results of analyzing the lineages
        for idx, (tHMMobj, _, LL) in enumerate(output):
            AIC_holder[ii, idx] = getAIC(tHMMobj, LL)[0]

    return desired_num_states, AIC_holder


def AIC_increased_cells1():
    """
    Calculates accuracy and parameter estimation by increasing the number of cells in a lineage for a one-state model.
    """

    # bern, gamma_a, gamma_scale
    E = [StateDistribution(0.99, 20, 5),
         StateDistribution(0.99, 20, 5)]

    return run_AIC(0.01, E)


def AIC_increased_cells2():
    """
    Calculates accuracy and parameter estimation by increasing the number of cells in a lineage for a two-state model.
    """

    # bern, gamma_a, gamma_scale
    E = [StateDistribution(0.99, 20, 5),
         StateDistribution(0.88, 10, 1)]

    return run_AIC(0.01, E)


def AIC_increased_cells3():
    """
    Calculates accuracy and parameter estimation by increasing the number of cells in a lineage for a three-state model.
    """

    # E: states are defined as StateDistribution objects
    # bern, gamma_a, gamma_scale
    E = [StateDistribution(0.7, 5.0, 1.0),
         StateDistribution(0.85, 10.0, 2.0),
         StateDistribution(0.99, 15.0, 3.0)]

    return run_AIC(0.01, E)


def figure_maker(ax, desired_num_states, AIC_holder):
    """
    Makes figure 11.
    """
    AIC_holder = AIC_holder - np.min(AIC_holder, axis=0)[np.newaxis, :]
    ax.set_xlabel('Number of States')
    ax.plot(desired_num_states, AIC_holder, 'k', alpha=0.5)
    ax.set_ylabel('Normalized AIC')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title('State Assignment AIC')
