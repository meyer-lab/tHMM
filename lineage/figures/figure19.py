""" To plot a summary of cross validation. """
import numpy as np
import scipy.stats as sp
import itertools as it
from ..LineageTree import LineageTree
from .common import getSetup
from ..Analyze import Analyze_list, Results
from ..tHMM import tHMM, fit_list
from ..BaumWelch import calculate_stationary
from ..states.StateDistributionGamma import StateDistribution

T = np.array([[0.9, 0.1], [0.1, 0.9]], dtype=float)

# pi: the initial probability vector
pi = calculate_stationary(T)

# bern, gamma_a, gamma_scale
state0 = StateDistribution(0.99, 100, 0.1)
state1 = StateDistribution(0.75, 80, 0.5)
E = [state0, state1]

def makeFigure():
    """
    Makes figure 19.
    """
    ax, f = getSetup((4, 4), (1, 1))

    ls, nls = all_cv()
    percs = np.arange(1, 8)
    ax[0].bar(percs - 0.1, list(it.chain(*ls)), width=0.2, label="pred state")
    ax[0].bar(percs + 0.1, list(it.chain(*nls)), width=0.2, label="not pred state")

    return f

def all_cv():
    """ find out the likelihoods for various masking percentages."""
    percs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    ls, nls = [], []
    for perc in percs:
        l, nl = cv(perc)
        ls.append(l)
        nls.append(nl)

    return ls, nls

def cv(percentage):
    """Simplest case of cross validation."""

    # create a lineage
    complete_lineage = LineageTree.init_from_parameters(pi, T, E, 31)
    true_states_by_lineage = [cell.state for cell in complete_lineage.output_lineage]

    # hide some percentage of observations
    lineage, hide_index = hide_observation(complete_lineage, percentage)

    # fit the not-hidden data from the lineages to the model
    tHMMobj_list, LL = Analyze_list([[lineage]], 2)

    # assign the predicted states to each cell
    states_list = [tHMMobj.predict() for tHMMobj in tHMMobj_list]

    for idx, tHMMobj in enumerate(tHMMobj_list):
        for lin_indx, lin in enumerate(tHMMobj.X):
            for cell_indx, cell in enumerate(lin.output_lineage):
                cell.state = states_list[idx][lin_indx][cell_indx]

    results = Results(tHMMobj_list[0], LL)
    print("confusion_mat", results["confusion_matrix"])
    print("accuracy", results["state_similarity"])

    # print the ttotal likelihood of the observations to the assigned compared to the other state
    lls, not_lls = calc_likelihood(tHMMobj_list[0], complete_lineage, hide_index)
    print("Likelihoods ", lls, "Not Likelihoods", not_lls)
    return lls, not_lls


def hide_observation(lineage, percentage):
    """This assumes we have cell lifetime and bernoulli as observations.
    We mark a random number of cells' lifetime as negative, to be removed from fitting."""

    num_cells = len(lineage.output_lineage)
    hide_index = sp.multinomial.rvs(n=int(percentage * num_cells), p=[1 / num_cells]*num_cells, size=1)[0]

    for ix, cell in enumerate(lineage.output_lineage):
        if hide_index[ix] == 1: # means we hide the cell lifetime

            if len(cell.obs) == 3: # in case of phase non-specific observations
                cell.obs[1] = -1
            else: # in the case of phase-specific observation, only do this for G1 lifetime
                cell.obs[2] = -1

    return lineage, hide_index


def calc_likelihood(tHMMobj, complete_lineage, hidden_index):
    """ Calculates the likelihood of cell's observation to the state it is assigned to, and sums over all for each lineage. """

    # find the list of hidden observations with the same order find their corresponding estimated state
    hidden_obs = []
    hidden_states = []
    for ix, cell in enumerate(complete_lineage.output_lineage):
        if hidden_index[ix] == 1:
            hidden_obs.append(cell.obs)
            hidden_states.append(tHMMobj.X[0].output_lineage[ix].state)

    # calculate the likelihood of the observation, to the state it is assigned to
    Ls = 0
    not_Ls = 0
    for ix, obs in enumerate(hidden_obs):
        Ls += tHMMobj.estimate.E[hidden_states[ix]].pdf(np.array(obs)[np.newaxis, :])
        not_Ls += tHMMobj.estimate.E[1 - hidden_states[ix]].pdf(np.array(obs)[np.newaxis, :])

    return Ls, not_Ls
