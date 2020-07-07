""" This file contains functions for plotting different phenotypes in the manuscript. """

import pandas as pd
import seaborn as sns
from .figureCommon import (
    getSetup,
    subplotLabel,
    pi,
    T,
    E2
)
from ..LineageTree import LineageTree
from ..Analyze import Analyze


def makeFigure():
    """
    Makes fig 2.
    """

    # Get list of axis objects
    ax, f = getSetup((5.0, 7.5), (3, 2))

    figureMaker2(ax, *forHistObs([LineageTree.init_from_parameters(pi, T, E2, desired_num_cells=2**8 - 1)]))

    subplotLabel(ax)

    return f


def forHistObs(X):
    """ To plot the histogram of the observations regardless of their state.

    :param X: list of lineages in the population.
    :type X: list
    """
    # regardless of states
    # two state model
    obsBernoulliG1 = []
    obsBernoulliG2 = []
    obsG1 = []
    obsG2 = []

    _, pred_states_by_lineage, _ = Analyze(X, 2)
    # state 1 observations
    obsBernoulliG1S1 = []
    obsBernoulliG2S1 = []
    obsG1S1 = []
    obsG2S1 = []
    # state 2 observations
    obsBernoulliG1S2 = []
    obsBernoulliG2S2 = []
    obsG1S2 = []
    obsG2S2 = []

    for indx, lineage in enumerate(X):
        for cell_ind, cell in enumerate(lineage.full_lineage):
            obsBernoulliG1.append(cell.obs[0])
            obsBernoulliG2.append(cell.obs[1])
            obsG1.append(cell.obs[2])
            obsG2.append(cell.obs[3])

            if pred_states_by_lineage[indx][cell_ind] == 0:  # if the cell is in state 1
                obsBernoulliG1S1.append(cell.obs[0])
                obsBernoulliG2S1.append(cell.obs[1])
                obsG1S1.append(cell.obs[2])
                obsG2S1.append(cell.obs[3])
            else:  # if the cell is in state 2
                obsBernoulliG1S2.append(cell.obs[0])
                obsBernoulliG2S2.append(cell.obs[1])
                obsG1S2.append(cell.obs[2])
                obsG2S2.append(cell.obs[3])

    list_bern_g1 = [obsBernoulliG1, obsBernoulliG1S1, obsBernoulliG1S2]
    list_bern_g2 = [obsBernoulliG2, obsBernoulliG2S1, obsBernoulliG2S2]

    totalObsG1 = pd.DataFrame(columns=['G1 phase duration [hr]', 'state'])
    totalObsG1['G1 phase duration [hr]'] = obsG1 + obsG1S1 + obsG1S2
    totalObsG1['state'] = ['total'] * len(obsG1) + ['state 1'] * len(obsG1S1) + ['state 2'] * len(obsG1S2)
    totalObsG2 = pd.DataFrame(columns=['G2 phase duration [hr]', 'state'])
    totalObsG2['G2 phase duration [hr]'] = obsG2 + obsG2S1 + obsG2S2
    totalObsG2['state'] = ['total'] * len(obsG2) + ['state 1'] * len(obsG2S1) + ['state 2'] * len(obsG2S2)

    return totalObsG1, totalObsG2, list_bern_g1, list_bern_g2


def figureMaker2(ax, totalObsG1, totalObsG2, list_bern_g1, list_bern_g2):
    """
    Makes the common 6 panel figures displaying parameter estimation across lineages
    of various types and sizes.
    """
    # cartoon to show different shapes --> similar shapes
    i = 0
    ax[i].axis('off')
    i += 1
    ax[i].axis('off')
    i += 1
    ax[i].set_xlabel("death or division")
    ax[i].set_ylim(bottom=0.5, top=1.1)
    ax[i].set_ylabel("division prob.")
    ax[i].set_title(r"Bernoulli obs. G1")
    ax[i].grid(linestyle="--")
    g = sns.barplot(data=list_bern_g1, ax=ax[i], palette="deep")
    g.text(-0.25, 0.65, "total", rotation=30)
    g.text(0.7, 0.65, "state 1", rotation=30)
    g.text(1.7, 0.65, "state 2", rotation=30)
    ax[i].tick_params(axis="both", which="major", grid_alpha=0.25)

    i += 1
    ax[i].set_xlabel("death or division")
    ax[i].set_ylim(bottom=0.5, top=1.1)
    ax[i].set_ylabel("division prob.")
    ax[i].set_title(r"Bernoulli obs. G2")
    ax[i].grid(linestyle="--")
    f = sns.barplot(data=list_bern_g2, ax=ax[i], palette="deep")
    f.text(-0.25, 0.55, "total", rotation=30)
    f.text(0.7, 0.55, "state 1", rotation=30)
    f.text(1.7, 0.55, "state 2", rotation=30)
    ax[i].tick_params(axis="both", which="major", grid_alpha=0.25)

    i += 1
    sns.violinplot(x="G1 phase duration [hr]", y="state", data=totalObsG1, ax=ax[i], palette="deep", scale="count", inner="quartile")
    ax[i].set_ylabel(r"PDF")
    ax[i].set_title(r"G1 phase")
    ax[i].grid(linestyle="--")
    ax[i].tick_params(axis="both", which="major", grid_alpha=0.25)

    i += 1
    sns.violinplot(x="G2 phase duration [hr]", y="state", data=totalObsG2, ax=ax[i], palette="deep", scale="count", inner="quartile")
    ax[i].set_ylabel(r"PDF")
    ax[i].set_title(r"G2 phase")
    ax[i].grid(linestyle="--")
    ax[i].tick_params(axis="both", which="major", grid_alpha=0.25)
