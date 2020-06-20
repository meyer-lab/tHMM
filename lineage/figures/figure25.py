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
    max_desired_num_cells,
    min_num_lineages,
    max_num_lineages,
    lineage_good_to_analyze,
    num_data_points,
    max_experiment_time
)
from ..LineageTree import LineageTree
from ..Analyze import Analyze


def makeFigure():
    """
    Makes fig 3.
    """

    # Get list of axis objects
    ax, f = getSetup((5.0, 11.5), (5,1))

    figureMaker2(ax, E2, *accuracy())

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
                tmp_lineage = LineageTree(pi, T, E2, max_desired_num_cells)
                good2go = lineage_good_to_analyze(tmp_lineage)

            population.append(tmp_lineage)

        # Adding populations into a holder for analysing
        list_of_populations.append(population)
        list_of_fpi.append(pi)
        list_of_fT.append(T)
        list_of_fE.append(E2)

    total_cellnum, all_states, paramEst, accuracy_after_switching, transition_matrix_norm, pi_vector_norm, paramTrues = commonAnalyze(list_of_populations)

    dataframe = pd.DataFrame(columns=['cell number', 'state acc.'])
    dataframe['cell number'] = total_cellnum
    dataframe['state acc.'] = accuracy_after_switching
    maxx = np.max(total_cellnum)
    for indx, num in enumerate(dataframe['cell number']):
        if num >= 0 and num <= maxx/5:
            dataframe['cell number'][indx] = int(maxx/10)
        elif num > maxx/5 and num <= maxx*(2/5):
            dataframe['cell number'][indx] = maxx*3/10
        elif num > maxx*(2/5) and num <= maxx*(3/5):
            dataframe['cell number'][indx] = maxx/2
        elif num > maxx*(3/5) and num <= maxx*(4/5):
            dataframe['cell number'][indx] = maxx*(7/10)
        elif num > maxx*(4/5) and num <= maxx:
            dataframe['cell number'][indx] = maxx*(9/10)

    dataParams = pd.DataFrame(columns=['cell number', 'state', 'Bern. G1 p', 'Bern. G1 true', 'Bern. G2 p', 'Bern. G2 true', 'shape G1', 'shape G1 true', 'scale G1', 'scale G1 true', 'shape G2', 'shape G2 true', 'scale G2', 'scale G2 true', 'T and pi', 'hue'])
    dataParams['cell number'] = dataframe['cell number'].append(dataframe['cell number'], ignore_index=True)
    dataParams['state'] = ['S1'] * paramEst[:, 0, 0].shape[0] + ['S2'] * paramEst[:, 1, 0].shape[0]
    dataParams['Bern. G1 p'] = np.concatenate((paramEst[:, 0, 0], paramEst[:, 1, 0]), axis=0)
    dataParams['Bern. G1 true'] = np.concatenate((paramTrues[:, 0, 0], paramTrues[:, 1, 0]), axis=0)
    dataParams['Bern. G2 p'] = np.concatenate((paramEst[:, 0, 1], paramEst[:, 1, 1]), axis=0)
    dataParams['Bern. G2 true'] = np.concatenate((paramTrues[:, 0, 1], paramTrues[:, 1, 1]), axis=0)
    dataParams['shape G1'] = np.concatenate((paramEst[:, 0, 2], paramEst[:, 1, 2]), axis=0)
    dataParams['shape G1 true'] = np.concatenate((paramTrues[:, 0, 2], paramTrues[:, 1, 2]), axis=0)
    dataParams['scale G1'] = np.concatenate((paramEst[:, 0, 3], paramEst[:, 1, 3]), axis=0)
    dataParams['scale G1 true'] = np.concatenate((paramTrues[:, 0, 3], paramTrues[:, 1, 3]), axis=0)
    dataParams['shape G2'] = np.concatenate((paramEst[:, 0, 4], paramEst[:, 1, 4]), axis=0)
    dataParams['shape G2 true'] = np.concatenate((paramTrues[:, 0, 4], paramTrues[:, 1, 4]), axis=0)
    dataParams['scale G2'] = np.concatenate((paramEst[:, 0, 5], paramEst[:, 1, 5]), axis=0)
    dataParams['scale G2 true'] = np.concatenate((paramTrues[:, 0, 5], paramTrues[:, 1, 5]), axis=0)
    dataParams['T and pi'] = np.concatenate((transition_matrix_norm, pi_vector_norm), axis=0)
    dataParams['hue'] = ['T'] * len(transition_matrix_norm) + ['pi'] * len(pi_vector_norm)

    return total_cellnum, dataframe, dataParams, paramTrues

def figureMaker2(ax, E, total_cellnum, dataframe, dataParams, paramTrues):
    """
    """
    i = 0
    sns.boxplot(x="cell number", y="state acc.", data=dataframe, ax=ax[i], palette="deep")
    ax[i].set_ylabel("accuracy")
    ax[i].set_title("state assignemnt accuracy")
    ax[i].set_ylabel("accuracy (%)")
    ax[i].grid(linestyle="--")
    ax[i].set_ylim(bottom=50.0, top=100.02)
    ax[i].tick_params(axis="both", which="major", grid_alpha=0.25)

    i += 1
    sns.stripplot(x="cell number", y='T and pi', hue='hue', dodge=False, jitter=True, data=dataParams, ax=ax[i], palette="deep", marker='o', linewidth=0.5, edgecolor="white", alpha=0.6)
    ax[i].set_ylim(bottom=-0.05, top=1.02)
    ax[i].set_ylabel("dif. from true value")
    ax[i].grid(linestyle="--")
    ax[i].tick_params(axis="both", which="major", grid_alpha=0.25)

    i += 1
    sns.stripplot(x="cell number", y='Bern. G1 p', hue='state', data=dataParams, dodge=False, jitter=True, ax=ax[i], marker='o', linewidth=0.5, edgecolor="white", palette=sns.xkcd_palette(['blue', 'green']))
    for tick, text in zip(ax[i].get_xticks(), ax[i].get_xticklabels()):
        # plot horizontal lines across the column, centered on the tick
        ax[i].plot([tick-0.5, tick+0.5], [paramTrues[:, 0, 0][0], paramTrues[:, 0, 0][0]], color='blue')
        ax[i].plot([tick-0.5, tick+0.5], [paramTrues[:, 1, 0][0], paramTrues[:, 1, 0][0]], color='green')
        ax[i].plot([tick-0.5, tick+0.5], [paramTrues[:, 0, 1][0], paramTrues[:, 0, 1][0]], color='orange')
        ax[i].plot([tick-0.5, tick+0.5], [paramTrues[:, 1, 1][0], paramTrues[:, 1, 1][0]], color='red')
    sns.stripplot(x="cell number", y='Bern. G2 p', hue='state', data=dataParams, dodge=False, jitter=True, ax=ax[i], marker='^', linewidth=0.5, edgecolor="white", palette=sns.xkcd_palette(['orange', 'red']))
    ax[i].grid(linestyle="--")
    ax[i].set_ylabel("bernoulli parameters")
    ax[i].tick_params(axis="both", which="major", grid_alpha=0.25)

    i += 1
    sns.stripplot(x="cell number", y='shape G1', hue='state', jitter=True, dodge=False, data=dataParams, ax=ax[i], marker='o', linewidth=0.5, edgecolor="white", palette=sns.xkcd_palette(['blue', 'green']))
    for tick, text in zip(ax[i].get_xticks(), ax[i].get_xticklabels()):
        # plot horizontal lines across the column, centered on the tick
        ax[i].plot([tick-0.5, tick+0.5], [paramTrues[:, 0, 2][0], paramTrues[:, 0, 2][0]], color='blue')
        ax[i].plot([tick-0.5, tick+0.5], [paramTrues[:, 1, 2][0], paramTrues[:, 1, 2][0]], color='green')
        ax[i].plot([tick-0.5, tick+0.5], [paramTrues[:, 0, 4][0], paramTrues[:, 0, 4][0]], color='orange')
        ax[i].plot([tick-0.5, tick+0.5], [paramTrues[:, 1, 4][0], paramTrues[:, 1, 4][0]], color='red')
    sns.stripplot(x="cell number", y='shape G2', hue='state', data=dataParams, dodge=False, jitter=True, ax=ax[i], marker='^', linewidth=0.5, edgecolor="white", palette=sns.xkcd_palette(['orange', 'red']))
    ax[i].grid(linestyle="--")
    ax[i].set_ylabel("shape parameter")
    ax[i].tick_params(axis="both", which="major", grid_alpha=0.25)

    i += 1
    sns.stripplot(x="cell number", y='scale G1', hue='state', data=dataParams, dodge=False, jitter=True, ax=ax[i], marker='o', linewidth=0.5, edgecolor="white", palette=sns.xkcd_palette(['blue', 'green']))
    for tick, text in zip(ax[i].get_xticks(), ax[i].get_xticklabels()):
        # plot horizontal lines across the column, centered on the tick
        ax[i].plot([tick-0.5, tick+0.5], [paramTrues[:, 0, 3][0], paramTrues[:, 0, 3][0]], color='blue')
        ax[i].plot([tick-0.5, tick+0.5], [paramTrues[:, 1, 3][0], paramTrues[:, 1, 3][0]], color='green')
        ax[i].plot([tick-0.5, tick+0.5], [paramTrues[:, 0, 5][0], paramTrues[:, 0, 5][0]], color='orange')
        ax[i].plot([tick-0.5, tick+0.5], [paramTrues[:, 1, 5][0], paramTrues[:, 1, 5][0]], color='red')
    sns.stripplot(x="cell number", y='scale G2', hue='state', data=dataParams, dodge=False, jitter=True, ax=ax[i], marker='^', linewidth=0.5, edgecolor="white", palette=sns.xkcd_palette(['orange', 'red']))
    ax[i].grid(linestyle="--")
    ax[i].set_ylabel("scale parameter")
    ax[i].tick_params(axis="both", which="major", grid_alpha=0.25)
