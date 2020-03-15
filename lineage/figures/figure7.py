"""
File: figure7.py
Purpose: Generates figure 7.

AIC.
"""
import numpy as np
from matplotlib.ticker import MaxNLocator

from .figureCommon import getSetup
from ..Analyze import getAIC, run_Analyze_over, LLFunc
from ..LineageTree import LineageTree
from ..StateDistribution import StateDistribution


def makeFigure():
    """
    Makes figure 7.
    """
    ax, f = getSetup((7, 3), (1, 3))

    # bern, gamma_a, gamma_scale
    Sone = StateDistribution(0.99, 20, 5)
    Stwo = StateDistribution(0.88, 10, 1)
    Eone = [Sone, Sone]
    Etwo = [Sone, Stwo]
    Ethree = [Sone, Stwo, StateDistribution(0.40, 30, 1)]

    figure_maker(ax[0], run_AIC(0.02, Eone))
    figure_maker(ax[1], run_AIC(0.02, Etwo))
    figure_maker(ax[2], run_AIC(0.02, Ethree))

    return f


# States to evaluate with the model
desired_num_states = np.arange(1, 6)


def run_AIC(Trate, E, num_to_evaluate=10):
    # Normalize the transition matrix
    T = Trate + np.eye(len(E))
    T = T / np.sum(T, axis=0)[np.newaxis, :]

    # pi: the initial probability vector
    # make an even starting p
    pi = np.ones(T.shape[0]) / T.shape[0]

    list_of_populations = []
    for idx in range(num_to_evaluate):
        # Creating an unpruned and pruned lineage
        list_of_populations.append([LineageTree(pi, T, E, (2**8) - 1)])

    AIC_holder = np.empty((len(desired_num_states), num_to_evaluate))
    for ii, num_states_to_evaluate in enumerate(desired_num_states):
        # Analyze the lineages in the list of populations
        output = run_Analyze_over(list_of_populations, num_states_to_evaluate)
        # Collecting the results of analyzing the lineages
        for idx, (tHMMobj,pred_states_by_lineage,_) in enumerate(output):
            # Get the likelihood of states
            LLtemp = LLFunc(T, pi, tHMMobj, pred_states_by_lineage)
            LL = np.sum(LLtemp)
            AIC_holder[ii, idx] = getAIC(tHMMobj, LL)[0]

    return AIC_holder


def figure_maker(ax, AIC_holder):
    """
    Makes figure 11.
    """
    AIC_holder = AIC_holder - np.min(AIC_holder, axis=0)[np.newaxis, :]
    ax.set_xlabel('Number of States')
    ax.plot(desired_num_states, AIC_holder, 'k', alpha=0.5)
    ax.set_ylabel('Normalized AIC')
    ax.set_ylim(0.0, 50.0)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title('State Assignment AIC')
