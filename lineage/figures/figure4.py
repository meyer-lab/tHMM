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
    num_data_points,
)
from ..LineageTree import LineageTree
import statsmodels


def makeFigure():
    """
    Makes fig S4.
    """

    # Get list of axis objects
    ax, f = getSetup((10, 10), (3, 3))
    figureMaker4(ax, *accuracy())

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
    num_lineages = np.linspace(min_num_lineages, int(0.35 * max_num_lineages), num_data_points, dtype=int)
    experiment_times = np.linspace(1000, int(2.5 * 1000), num_data_points)
    list_of_populations = []
    list_of_fpi = []
    list_of_fT = []
    list_of_fE = []
    for indx, num in enumerate(num_lineages):
        population = []
        for _ in range(num):

            good2go = False
            while not good2go:
                tmp_lineage = LineageTree.init_from_parameters(pi, T, E2, desired_num_cells=min_desired_num_cells, censor_condition=3, desired_experiment_time=experiment_times[indx])
                good2go = lineage_good_to_analyze(tmp_lineage)

            population.append(tmp_lineage)

        # Adding populations into a holder for analysing
        list_of_populations.append(population)
        list_of_fpi.append(pi)
        list_of_fT.append(T)
        list_of_fE.append(E2)

    cell_number_x, paramEst, accuracy_after_switching, transition_matrix_norm, pi_vector_norm, paramTrues = commonAnalyze(list_of_populations)

    accuracy_df = pd.DataFrame(columns=["Cell Number", 'State Assignment Accuracy'])
    accuracy_df['Cell Number'] = cell_number_x
    accuracy_df['State Assignment Accuracy'] = accuracy_after_switching

    param_df = pd.DataFrame(columns=["T", "pi"])
    param_df["Cell Number"] = accuracy_df["Cell Number"].to_list() + accuracy_df["Cell Number"].to_list()
    param_df['Error'] = transition_matrix_norm + pi_vector_norm
    param_df["Parameter"] = len(transition_matrix_norm) * [r"$T$"] + len(pi_vector_norm) * [r"$\pi$"]

    data_df = pd.DataFrame(columns=["Cell Number", "State", 'Bern. G1 p', 'Bern. G2 p', 'shape G1', 'scale G1', 'shape G2', 'scale G2'])
    data_df["Cell Number"] = accuracy_df["Cell Number"].to_list() + accuracy_df["Cell Number"].to_list()
    data_df["State"] = ["State 1"] * paramEst[:, 0, 0].shape[0] + ["State 2"] * paramEst[:, 1, 0].shape[0]
    data_df['Bern. G1 p'] = np.concatenate((paramEst[:, 0, 0], paramEst[:, 1, 0]), axis=0)
    data_df['Bern. G2 p'] = np.concatenate((paramEst[:, 0, 1], paramEst[:, 1, 1]), axis=0)
    data_df['shape G1'] = np.concatenate((paramEst[:, 0, 2], paramEst[:, 1, 2]), axis=0)
    data_df['scale G1'] = np.concatenate((paramEst[:, 0, 3], paramEst[:, 1, 3]), axis=0)
    data_df['shape G2'] = np.concatenate((paramEst[:, 0, 4], paramEst[:, 1, 4]), axis=0)
    data_df['scale G2'] = np.concatenate((paramEst[:, 0, 5], paramEst[:, 1, 5]), axis=0)

    return accuracy_df, param_df, data_df, paramTrues


def figureMaker4(ax, accuracy_df, param_df, data_df, paramTrues):
    """
    This makes figure 4.
    """
    i = 0
    ax[i].axis('off')

    i += 1
    sns.regplot(x="Cell Number", y="State Assignment Accuracy", data=accuracy_df, ax=ax[i], lowess=True)
    ax[i].set_title("State Assignment Accuracy")
    ax[i].set_ylabel("Accuracy [%]")
    ax[i].set_ylim(bottom=25.0, top=101)

    # T and pi matrix distance to their true value
    i += 1
    ax[i] = sns.regplot(x="Cell Number", y="Error", data=param_df.loc["Parameter" == r"$T$"], ax=ax[i], lowess=True)
    ax[i] = sns.regplot(x="Cell Number", y="Error", data=param_df.loc["Parameter" == r"$\pi$"], ax=ax[i], lowess=True)
    ax[i].set_title(r"Error in estimating $T$ & $\pi$")
    ax[i].set_ylabel(r"Error [$||x-\hat{x}||$]")
    ax[i].set_ylim(bottom=0.01, top=1.02)

    i += 1
    # Bernoulli parameter estimation
    ax[i].axhline(y=paramTrues[:, 0, 0][0], ls='--', c='k', alpha=0.5)
    ax[i].axhline(y=paramTrues[:, 1, 0][0], ls='--', c='k', alpha=0.5)
    ax[i] = sns.lmplot(x="Cell Number", y='Bern. G1 p', hue='State', data=data_df, lowess=True).fig.axes[0]
    ax[i].set_title(r"G1 fate parameter estimation ($p$)")
    ax[i].set_ylabel("Bernoulli rate estimate ($p$)")
    ax[i].set_ylim(0.75, 1.01)

    i += 1
    ax[i].axhline(y=paramTrues[:, 0, 2][0], ls='--', c='k', alpha=0.5)
    ax[i].axhline(y=paramTrues[:, 1, 2][0], ls='--', c='k', alpha=0.5)
    ax[i] = sns.lmplot(x="Cell Number", y='shape G1', hue='State', data=data_df, lowess=True).fig.axes[0]
    ax[i].set_title(r"G1 lifetime parameter estimation ($k$, $\theta$)")
    ax[i].set_ylabel("Gamma shape estimate ($k$)")
    ax[i].set_ylim(1, 15)

    i += 1
    ax[i].axhline(y=paramTrues[:, 0, 3][0], ls='--', c='k', alpha=0.5)
    ax[i].axhline(y=paramTrues[:, 1, 3][0], ls='--', c='k', alpha=0.5)
    ax[i] = sns.lmplot(x="Cell Number", y='scale G1', hue='State', data=data_df, lowess=True).fig.axes[0]
    ax[i].set_title(r"G1 lifetime parameter estimation ($k$, $\theta$)")
    ax[i].set_ylabel(r"Gamma scale estimate ($\theta$)")
    ax[i].set_ylim(1, 15)

    i += 1
    ax[i].axhline(y=paramTrues[:, 0, 1][0], ls='--', c='k', alpha=0.5)
    ax[i].axhline(y=paramTrues[:, 1, 1][0], ls='--', c='k', alpha=0.5)
    ax[i] = sns.lmplot(x="Cell Number", y='Bern. G2 p', hue='State', data=data_df, lowess=True).fig.axes[0]
    ax[i].set_title(r"G2 fate parameter estimation ($p$)")
    ax[i].set_ylabel(r"Bernoulli rate estimate ($p$)")
    ax[i].set_ylim(0.75, 1.01)

    i += 1
    ax[i].axhline(y=paramTrues[:, 0, 4][0], ls='--', c='k', alpha=0.5)
    ax[i].axhline(y=paramTrues[:, 1, 4][0], ls='--', c='k', alpha=0.5)
    ax[i] = sns.lmplot(x="Cell Number", y='shape G2', hue='State', data=data_df, lowess=True).fig.axes[0]
    ax[i].set_title(r"G2 lifetime parameter estimation ($k$, $\theta$)")
    ax[i].set_ylabel(r"Gamma shape estimate ($k$)")
    ax[i].set_ylim(0, 10)

    i += 1
    ax[i].axhline(y=paramTrues[:, 0, 5][0], ls='--', c='k', alpha=0.5)
    ax[i].axhline(y=paramTrues[:, 1, 5][0], ls='--', c='k', alpha=0.5)
    ax[i] = sns.lmplot(x="Cell Number", y='scale G2', hue='State', data=data_df, lowess=True).fig.axes[0]
    ax[i].set_title(r"G2 lifetime parameter estimation ($k$, $\theta$)")
    ax[i].set_ylabel(r"Gamma scale estimate ($\theta$)")
    ax[i].set_ylim(0, 10)
