""" This file contains figures related to how big the experment needs to be. """
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
    num_data_points
)
from ..LineageTree import LineageTree


def makeFigure():
    """
    Makes fig 3A.
    """

    # Get list of axis objects
    ax, f = getSetup((7.5, 5), (2, 3))
    number_of_columns = 5
    figureMaker2(ax, *accuracy(number_of_columns))

    subplotLabel(ax)

    return f


def accuracy(number_of_columns):
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
                tmp_lineage = LineageTree.init_from_parameters(pi, T, E2, min_desired_num_cells)
                good2go = lineage_good_to_analyze(tmp_lineage)

            population.append(tmp_lineage)

        # Adding populations into a holder for analysing
        list_of_populations.append(population)
        list_of_fpi.append(pi)
        list_of_fT.append(T)
        list_of_fE.append(E2)

    cell_number_x, paramEst, accuracy_after_switching, transition_matrix_norm, pi_vector_norm, paramTrues = commonAnalyze(list_of_populations)

    accuracy_df = pd.DataFrame(columns=["Approximate Cell Number", 'State Assignment Accuracy'])
    accuracy_df['Approximate Cell Number'] = cell_number_x
    accuracy_df['State Assignment Accuracy'] = accuracy_after_switching
    maxx = np.max(cell_number_x)
    cell_number_columns = [maxx*(i+1)/number_of_columns for i in range(number_of_columns)]
    assert len(cell_number_columns) == number_of_columns
    for indx, num in enumerate(cell_number_x):
        bottom_range = 0
        for i in range(number_of_columns):
            if bottom_range < num <= cell_number_columns[i]:
                accuracy_df['Approximate Cell Number'][indx] = maxx*((2*i+1)/(2*number_of_columns))
            bottom_range += cell_number_columns[i]

    data_df = pd.DataFrame(columns=["Approximate Cell Number", "State", 'Bern. G1 p', 'Bern. G2 p', 'shape G1', 'scale G1', 'shape G2', 'scale G2', 'T and pi', 'Parameter'])
    data_df["Approximate Cell Number"] = accuracy_df["Approximate Cell Number"]
    data_df['state'] = ["State 1"] * paramEst[:, 0, 0].shape[0] + ["State 2"] * paramEst[:, 1, 0].shape[0]
    data_df['Bern. G1 p'] = np.concatenate((paramEst[:, 0, 0], paramEst[:, 1, 0]), axis=0)
    data_df['Bern. G2 p'] = np.concatenate((paramEst[:, 0, 1], paramEst[:, 1, 1]), axis=0)
    data_df['shape G1'] = np.concatenate((paramEst[:, 0, 2], paramEst[:, 1, 2]), axis=0)
    data_df['scale G1'] = np.concatenate((paramEst[:, 0, 3], paramEst[:, 1, 3]), axis=0)
    data_df['shape G2'] = np.concatenate((paramEst[:, 0, 4], paramEst[:, 1, 4]), axis=0)
    data_df['scale G2'] = np.concatenate((paramEst[:, 0, 5], paramEst[:, 1, 5]), axis=0)
    data_df['T and pi'] = np.concatenate((transition_matrix_norm, pi_vector_norm), axis=0)
    data_df['hue'] = ['T'] * len(transition_matrix_norm) + ['pi'] * len(pi_vector_norm)

    return accuracy_df, data_df, paramTrues


def figureMaker2(ax, accuracy_df, data_df, paramTrues):
    """
    This makes figure 3A.
    """
    i = 0
    ax[i].axis('off')
    

    i += 1
    sns.boxplot(x="cell number", y="state acc.", data=accuracy_df, ax=ax[i])
    ax[i].set_title("State Assignment Accuracy")
    ax[i].set_ylabel("Accuracy (%)")
    ax[i].set_ylim(bottom=50.0, top=105.0)

    # T and pi matrix distance to their true value
    i += 1
    sns.stripplot(x="cell number", y='T and pi', hue='hue', dodge=False, jitter=True, data=data_df, ax=ax[i], marker='o', linewidth=0.5, edgecolor="white", alpha=0.6)
    ax[i].set_ylim(bottom=-0.2, top=1.02)
    ax[i].set_ylabel("dif. from true value")

    i += 1
    # Bernoulli parameter estimation
    sns.stripplot(x="cell number", y='Bern. G1 p', hue='state', data=data_df, dodge=False,
                  jitter=True, ax=ax[i], marker='o', linewidth=0.5, edgecolor="white",
                  palette=sns.xkcd_palette(['blue', 'green']))
    for tick, _ in zip(ax[i].get_xticks(), ax[i].get_xticklabels()):
        # plot horizontal lines across the column, centered on the tick
        ax[i].plot([tick - 0.5, tick + 0.5], [paramTrues[:, 0, 0][0], paramTrues[:, 0, 0][0]],
                   color='blue', alpha=0.6)
        ax[i].plot([tick - 0.5, tick + 0.5], [paramTrues[:, 1, 0][0], paramTrues[:, 1, 0][0]], color='green',
                   alpha=0.6)
        ax[i].plot([tick - 0.5, tick + 0.5], [paramTrues[:, 0, 1][0], paramTrues[:, 0, 1][0]],
                   color='orange', alpha=0.6)
        ax[i].plot([tick - 0.5, tick + 0.5], [paramTrues[:, 1, 1][0], paramTrues[:, 1, 1][0]], color='red',
                   alpha=0.6)
    sns.stripplot(x="cell number", y='Bern. G2 p', hue='state', data=data_df, dodge=False,
                  jitter=True, ax=ax[i], marker='^', linewidth=0.5, edgecolor="white",
                  palette=sns.xkcd_palette(['orange', 'red']))
    ax[i].set_ylim(bottom=0.6, top=1.2)
    ax[i].set_ylabel("bernoulli parameters")
    ax[i].text(5.0, 1.0, str(repr('o') + " G1 \n" + str(repr('^')) + " G2"))
    ax[i].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    i += 1
    sns.stripplot(x="cell number", y='shape G1', hue='state', jitter=True, dodge=False, data=data_df,
                  ax=ax[i], marker='o', linewidth=0.5, edgecolor="white",
                  palette=sns.xkcd_palette(['blue', 'green']))
    for tick, _ in zip(ax[i].get_xticks(), ax[i].get_xticklabels()):
        # plot horizontal lines across the column, centered on the tick
        ax[i].plot([tick - 0.5, tick + 0.5], [paramTrues[:, 0, 2][0], paramTrues[:, 0, 2][0]], color='blue',
                   alpha=0.6)
        ax[i].plot([tick - 0.5, tick + 0.5], [paramTrues[:, 1, 2][0], paramTrues[:, 1, 2][0]], color='green',
                   alpha=0.6)
        ax[i].plot([tick - 0.5, tick + 0.5], [paramTrues[:, 0, 4][0], paramTrues[:, 0, 4][0]], color='orange',
                   alpha=0.6)
        ax[i].plot([tick - 0.5, tick + 0.5], [paramTrues[:, 1, 4][0], paramTrues[:, 1, 4][0]], color='red',
                   alpha=0.6)
    sns.stripplot(x="cell number", y='shape G2', hue='state', data=data_df, dodge=False, jitter=True,
                  ax=ax[i], marker='^', linewidth=0.5, edgecolor="white", palette=sns.xkcd_palette(['orange', 'red']))
    ax[i].set_ylim(bottom=-0.05, top=15.0)
    ax[i].text(1.2, 2.5, str(repr('o') + " G1 \n" + str(repr('^')) + " G2"))
    ax[i].set_ylabel("shape parameter")
    ax[i].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    i += 1
    sns.stripplot(x="cell number", y='scale G1', hue='state', data=data_df, dodge=False, jitter=True,
                  ax=ax[i], marker='o', linewidth=0.5, edgecolor="white", palette=sns.xkcd_palette(['blue', 'green']))
    for tick, _ in zip(ax[i].get_xticks(), ax[i].get_xticklabels()):
        # plot horizontal lines across the column, centered on the tick
        ax[i].plot([tick - 0.5, tick + 0.5], [paramTrues[:, 0, 3][0], paramTrues[:, 0, 3][0]], color='blue',
                   alpha=0.6)
        ax[i].plot([tick - 0.5, tick + 0.5], [paramTrues[:, 1, 3][0], paramTrues[:, 1, 3][0]], color='green',
                   alpha=0.6)
        ax[i].plot([tick - 0.5, tick + 0.5], [paramTrues[:, 0, 5][0], paramTrues[:, 0, 5][0]], color='orange',
                   alpha=0.6)
        ax[i].plot([tick - 0.5, tick + 0.5], [paramTrues[:, 1, 5][0], paramTrues[:, 1, 5][0]], color='red',
                   alpha=0.6)
    sns.stripplot(x="cell number", y='scale G2', hue='state', data=data_df, dodge=True, jitter=True,
                  ax=ax[i], marker='^', linewidth=0.5, edgecolor="white", palette=sns.xkcd_palette(['orange', 'red']))
    ax[i].set_ylim(bottom=-0.05, top=11.0)
    ax[i].set_ylabel("scale parameter")
    ax[i].text(1.1, 7.5, str(repr('o') + " G1 \n" + str(repr('^')) + " G2"))
    ax[i].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
