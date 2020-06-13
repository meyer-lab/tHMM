""" This file contains functions for plotting different phenotypes in the manuscript. """

import numpy as np
import math
import pandas as pd
import seaborn as sns
import itertools
from .figureCommon import (
    getSetup,
    subplotLabel,
    commonAnalyze,
    pi,
    T,
    E2,
    min_desired_num_cells,
    min_num_lineages,
    max_num_lineages,
    num_data_points,
)
from ..LineageTree import LineageTree
from ..Analyze import Analyze

lineage1 = LineageTree(pi, T, E2, desired_num_cells=1023)
X = [lineage1]

def makeFigure():
    """
    Makes fig 2.
    """

    # Get list of axis objects
    ax, f = getSetup((7.5, 5.0), (2, 3))

    figureMaker2(ax, *forHistObs(X))

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

    tHMMobj, pred_states_by_lineage, LL = Analyze(X, 2)
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

            if pred_states_by_lineage[indx][cell_ind] == 0: # if the cell is in state 1
                obsBernoulliG1S1.append(cell.obs[0])
                obsBernoulliG2S1.append(cell.obs[1])
                obsG1S1.append(cell.obs[2])
                obsG2S1.append(cell.obs[3])
            else: # if the cell is in state 2
                obsBernoulliG1S2.append(cell.obs[0])
                obsBernoulliG2S2.append(cell.obs[1])
                obsG1S2.append(cell.obs[2])
                obsG2S2.append(cell.obs[3])

    list_bern_g1 = [obsBernoulliG1, obsBernoulliG1S1, obsBernoulliG1S2]
    list_bern_g2 = [obsBernoulliG2, obsBernoulliG2S1, obsBernoulliG2S2]
    list_obs_g1 = [obsG1, obsG1S1, obsG1S2]
    list_obs_g2 = [obsG2, obsG2S1, obsG2S2]

    totalbernG1 = pd.DataFrame(columns = ['values', 'state'])
    totalbernG1['values'] = obsBernoulliG1 + obsBernoulliG1S1 + obsBernoulliG1S2
    totalbernG1['state'] = ['total'] * len(obsBernoulliG1) + ['state 1'] * len(obsBernoulliG1S1) + ['state 2'] * len(obsBernoulliG1S2)
    totalbernG2 = pd.DataFrame(columns = ['values', 'state'])
    totalbernG2['values'] = obsBernoulliG2 + obsBernoulliG2S1 + obsBernoulliG2S2
    totalbernG2['state'] = ['total'] * len(obsBernoulliG2) + ['state 1'] * len(obsBernoulliG2S1) + ['state 2'] * len(obsBernoulliG2S2)
    
    totalObsG1 = pd.DataFrame(columns = ['values', 'state'])
    totalObsG1['values'] = obsG1 + obsG1S1 + obsG1S2
    totalObsG1['state'] = ['total'] * len(obsG1) + ['state 1'] * len(obsG1S1) + ['state 2'] * len(obsG1S2)
    totalObsG2 = pd.DataFrame(columns = ['values', 'state'])
    totalObsG2['values'] = obsG2 + obsG2S1 + obsG2S2
    totalObsG2['state'] = ['total'] * len(obsG2) + ['state 1'] * len(obsG2S1) + ['state 2'] * len(obsG2S2)

    return totalbernG1, totalbernG2, totalObsG1, totalObsG2, list_obs_g1, list_obs_g2, list_bern_g1, list_bern_g2

def figureMaker2(ax, totalbernG1, totalbernG2, totalObsG1, totalObsG2, list_obs_g1, list_obs_g2, list_bern_g1, list_bern_g2):
    """
    Makes the common 6 panel figures displaying parameter estimation across lineages
    of various types and sizes.
    """
    i = 0
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
    ax[i].set_xlabel("G1 phase duration")
    w = 10
    n2 = math.ceil((np.max(list_obs_g1[1]) - np.min(list_obs_g1[1]))/w)
    n3 = math.ceil((np.max(list_obs_g1[2]) - np.min(list_obs_g1[2]))/w)
    ax[i].hist(list_obs_g1[1], density=True, label="in state 1", alpha=0.6, color="sienna", bins=n2)
    ax[i].hist(list_obs_g1[2], density=True, label="in state 2", alpha=0.6, color="seagreen", bins=n3)
    ax[i].set_ylabel(r"PDF")
    ax[i].set_title(r"G1 phase [hr]")
    ax[i].grid(linestyle="--")
    sns.kdeplot(list_obs_g1[0], ax=ax[i], label="total")
    ax[i].tick_params(axis="both", which="major", grid_alpha=0.25)
    ax[i].legend()

    i += 1
    # violin
    ax[i].set_xlabel("G1 phase duration")
    sns.violinplot(x="values", y="state", data=totalObsG1, ax=ax[i], palette="deep", scale="count", inner="quartile")
    ax[i].set_ylabel(r"PDF")
    ax[i].set_title(r"G1 phase [hr]")
    ax[i].grid(linestyle="--")
    ax[i].tick_params(axis="both", which="major", grid_alpha=0.25)

    i += 1
    ax[i].set_xlabel("G2 phase duration")
    w = 10
    n2 = math.ceil((np.max(list_obs_g2[1]) - np.min(list_obs_g2[1]))/w)
    n3 = math.ceil((np.max(list_obs_g2[2]) - np.min(list_obs_g2[2]))/w)
    ax[i].hist(list_obs_g2[1], density=True, label="in state 1", alpha=0.6, color="sienna", bins=n2)
    ax[i].hist(list_obs_g2[2], density=True, label="in state 2", alpha=0.6, color="seagreen", bins=n3)
    ax[i].set_ylabel(r"PDF")
    ax[i].set_title(r"G2 phase [hr]")
    ax[i].grid(linestyle="--")
    sns.kdeplot(list_obs_g2[0], ax=ax[i], label="total")
    ax[i].tick_params(axis="both", which="major", grid_alpha=0.25)
    ax[i].legend()

    i += 1
    # violin
    ax[i].set_xlabel("G2 phase duration")
    sns.violinplot(x="values", y="state", data=totalObsG2, ax=ax[i], palette="deep", scale="count", inner="quartile")
    ax[i].set_ylabel(r"PDF")
    ax[i].set_title(r"G2 phase [hr]")
    ax[i].grid(linestyle="--")
    ax[i].tick_params(axis="both", which="major", grid_alpha=0.25)