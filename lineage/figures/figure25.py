""" This file contains figures related to how big the experment needs to be. """
import math
import numpy as np
import pandas as pd
import seaborn as sns
from .figureCommon import (
    getSetup,
    subplotLabel,
    commonAnalyze,
    pi,
    E2,
    T,
    min_desired_num_cells,
    min_num_lineages,
    max_num_lineages,
    lineage_good_to_analyze,
    num_data_points,
)
from ..LineageTree import LineageTree
from ..Analyze import Analyze


def makeFigure():
    """
    Makes fig 3.
    """

    # Get list of axis objects
    ax, f = getSetup((4.5, 9.0), (4, 1))

    figureMaker2(ax, *accuracy())

    subplotLabel(ax)

    return f

def accuracy():
    """
    Calculates accuracy and parameter estimation
    over an increasing number of lineages in a population for
    a uncensored two-state model.
    We increase the desired number of cells in a lineage by
    the experiment time.
    """

    # Creating a list of populations to analyze over
    num_lineages = np.linspace(min_num_lineages, max_num_lineages, num_data_points, dtype=int)
    list_of_populations = []
    list_of_fpi = []
    list_of_fT = []
    list_of_fE = []
    for num in num_lineages:
        population = []

        for _ in range(num):

            good2go = False
            while not good2go:
                tmp_lineage = LineageTree(pi, T, E2, min_desired_num_cells)
                good2go = lineage_good_to_analyze(tmp_lineage)

            population.append(tmp_lineage)

        # Adding populations into a holder for analysing
        list_of_populations.append(population)
        list_of_fpi.append(pi)
        list_of_fT.append(T)
        list_of_fE.append(E2)

    total_cellnum, all_states, paramEst, accuracy_after_switching, transition_matrix_norm, pi_vector_norm, paramTrues = commonAnalyze(list_of_populations)

    dataframe = pd.DataFrame(columns=['cell number', 'state acc.', 'T norm', r'$\pi$ norm'])
    dataframe['cell number'] = total_cellnum
    dataframe['state acc.'] = accuracy_after_switching
    dataframe['T norm'] = transition_matrix_norm
    dataframe[r'$\pi$ norm'] = pi_vector_norm
    for indx, num in enumerate(dataframe['cell number']):
        if num >= 0 and num <= 200:
            dataframe['cell number'][indx] = 100
        elif num > 200 and num < 400:
            dataframe['cell number'][indx] = 300
        elif num > 400 and num < 600:
            dataframe['cell number'][indx] = 500
        elif num > 600 and num < 800:
            dataframe['cell number'][indx] = 700
        elif num > 800:
            dataframe['cell number'][indx] = 900

    dataParams = pd.DataFrame(columns=['cell number', 'state', 'Bern. G1 p', 'Bern. G2 p', 'shape G1', 'scale G1', 'shape G2', 'scale G2'])
    dataParams['cell number'] = dataframe['cell number'].append(dataframe['cell number'], ignore_index=True)
    dataParams['state'] = ['state 1'] * paramEst[:, 0, 0].shape[0] + ['state 2'] * paramEst[:, 1, 0].shape[0]
    dataParams['Bern. G1 p'] = np.concatenate((paramEst[:, 0, 0], paramEst[:, 1, 0]), axis=0)
    dataParams['Bern. G2 p'] = np.concatenate((paramEst[:, 0, 1], paramEst[:, 1, 1]), axis=0)
    dataParams['shape G1'] = np.concatenate((paramEst[:, 0, 2], paramEst[:, 1, 2]), axis=0)
    dataParams['scale G1'] = np.concatenate((paramEst[:, 0, 3], paramEst[:, 1, 3]), axis=0)
    dataParams['shape G2'] = np.concatenate((paramEst[:, 0, 4], paramEst[:, 1, 4]), axis=0)
    dataParams['scale G2'] = np.concatenate((paramEst[:, 0, 5], paramEst[:, 1, 5]), axis=0)

            
    return dataframe, paramTrues, dataParams

def figureMaker2(ax, dataframe, paramTrues, dataParams):
    """
    """
    i = 0
    sns.violinplot(x="cell number", y="state acc.", data=dataframe, ax=ax[i], palette="deep", scale="count", inner="quartile")
    ax[i].set_ylabel("accuracy")
    ax[i].set_title("state assignemnt accuracy")
    ax[i].grid(linestyle="--")
    ax[i].tick_params(axis="both", which="major", grid_alpha=0.25)

    i += 1
    sns.violinplot(x="cell number", y="T norm", data=dataframe, ax=ax[i], palette="deep", inner="quartile", aplha=0.7)
    sns.violinplot(x="cell number", y=r'$\pi$ norm', data=dataframe, ax=ax[i], palette="deep", scale="count", inner="quartile", alpha=0.7)
    ax[i].set_ylim(bottom=0, top=1.02)
    ax[i].set_ylabel("transition probability matrix")
    ax[i].set_title(r"$||T-T_{est}||_{F}$")
    ax[i].grid(linestyle="--")
    ax[i].tick_params(axis="both", which="major", grid_alpha=0.25)

    i += 1
    sns.stripplot(x="cell number", y='Bern. G1 p', hue='state', data=dataParams, ax=ax[i], marker='o')
    sns.stripplot(x="cell number", y='Bern. G2 p', hue='state', data=dataParams, ax=ax[i], marker='^')
    ax[i].grid(linestyle="--")
    ax[i].tick_params(axis="both", which="major", grid_alpha=0.25)

    i += 1
    sns.stripplot(x="cell number", y='shape G1', hue='state', data=dataParams, ax=ax[i])
    sns.stripplot(x="cell number", y='scale G1', hue='state', data=dataParams, ax=ax[i])
    sns.stripplot(x="cell number", y='shape G2', hue='state', data=dataParams, ax=ax[i])
    sns.stripplot(x="cell number", y='scale G2', hue='state', data=dataParams, ax=ax[i])
    ax[i].grid(linestyle="--")
    ax[i].tick_params(axis="both", which="major", grid_alpha=0.25)
