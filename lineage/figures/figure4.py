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
    scatter_kws_list,
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

    param_df = pd.DataFrame(columns=["Cell Number", "T", "pi"])
    param_df["Cell Number"] = accuracy_df["Cell Number"].to_list()
    param_df['T Error'] = transition_matrix_norm
    param_df['pi Error'] = pi_vector_norm

    data_df = pd.DataFrame(columns=["Cell Number", "State", 'Bern. G1 p', 'Bern. G2 p', 'shape G1', 'scale G1', 'shape G2', 'scale G2'])
    data_df["Cell Number"] = accuracy_df["Cell Number"].to_list()
    data_df['Bern. G1 0'] = paramEst[:, 0, 0]
    data_df['Bern. G1 1'] = paramEst[:, 1, 0]
    data_df['Bern. G2 0'] = paramEst[:, 0, 1]
    data_df['Bern. G2 1'] = paramEst[:, 1, 1]
    data_df['shape G1 0'] = paramEst[:, 0, 2]
    data_df['shape G1 1'] = paramEst[:, 1, 2]
    data_df['scale G1 0'] = paramEst[:, 0, 3]
    data_df['scale G1 1'] = paramEst[:, 1, 3]
    data_df['shape G2 0'] = paramEst[:, 0, 4]
    data_df['shape G2 1'] = paramEst[:, 1, 4]
    data_df['scale G2 0'] = paramEst[:, 0, 5]
    data_df['scale G2 1'] = paramEst[:, 1, 5]

    return accuracy_df, param_df, data_df, paramTrues


def figureMaker4(ax, accuracy_df, param_df, data_df, paramTrues):
    """
    This makes figure 4.
    """    
    i = 0
    ax[i].axis('off')

    i += 1
    sns.regplot(x="Cell Number", y="State Assignment Accuracy", data=accuracy_df, ax=ax[i], lowess=True, marker='+', scatter_kws=scatter_kws_list[0])
    ax[i].set_title("State Assignment Accuracy")
    ax[i].set_ylabel("Accuracy [%]")
    ax[i].set_ylim(bottom=25.0, top=101)

    # T and pi matrix distance to their true value
    i += 1
    sns.regplot(x="Cell Number", y="T Error", data=param_df, ax=ax[i], lowess=True, marker='+', scatter_kws=scatter_kws_list[0])
    sns.regplot(x="Cell Number", y="pi Error", data=param_df, ax=ax[i], lowess=True, marker='+', scatter_kws=scatter_kws_list[1])
    ax[i].set_title(r"Error in estimating $T$ & $\pi$")
    ax[i].set_ylabel(r"Error [$||x-\hat{x}||$]")
    ax[i].set_ylim(bottom=0.01, top=1.02)

    i += 1
    # Bernoulli parameter estimation
    ax[i].axhline(y=paramTrues[:, 0, 0][0], ls='--', c='b', alpha=0.75)
    ax[i].axhline(y=paramTrues[:, 1, 0][0], ls='--', c='orange', alpha=0.75)
    sns.regplot(x="Cell Number", y='Bern. G1 0', data=data_df, ax=ax[i], lowess=True, marker='+', scatter_kws=scatter_kws_list[0])
    sns.regplot(x="Cell Number", y='Bern. G1 1', data=data_df, ax=ax[i], lowess=True, marker='+', scatter_kws=scatter_kws_list[1])
    ax[i].set_title(r"G1 fate parameter estimation ($p$)")
    ax[i].set_ylabel("Bernoulli rate estimate ($p$)")
    ax[i].set_ylim(paramTrues[:, 1, 0][0]-0.025, 1.001)

    i += 1
    ax[i].axhline(y=paramTrues[:, 0, 2][0], ls='--', c='b', alpha=0.75)
    ax[i].axhline(y=paramTrues[:, 1, 2][0], ls='--', c='orange', alpha=0.75)
    sns.regplot(x="Cell Number", y='shape G1 0', data=data_df, ax=ax[i], lowess=True, marker='+', scatter_kws=scatter_kws_list[0])
    sns.regplot(x="Cell Number", y='shape G1 1', data=data_df, ax=ax[i], lowess=True, marker='+', scatter_kws=scatter_kws_list[1])
    ax[i].set_title(r"G1 lifetime parameter estimation ($k$, $\theta$)")
    ax[i].set_ylabel("Gamma shape estimate ($k$)")
    ax[i].set_ylim(1, 15)

    i += 1
    ax[i].axhline(y=paramTrues[:, 0, 3][0], ls='--', c='b', alpha=0.75)
    ax[i].axhline(y=paramTrues[:, 1, 3][0], ls='--', c='orange', alpha=0.75)
    sns.regplot(x="Cell Number", y='scale G1 0', data=data_df, ax=ax[i], lowess=True, marker='+', scatter_kws=scatter_kws_list[0])
    sns.regplot(x="Cell Number", y='scale G1 1', data=data_df, ax=ax[i], lowess=True, marker='+', scatter_kws=scatter_kws_list[1])
    ax[i].set_title(r"G1 lifetime parameter estimation ($k$, $\theta$)")
    ax[i].set_ylabel(r"Gamma scale estimate ($\theta$)")
    ax[i].set_ylim(1, 15)

    i += 1
    ax[i].axhline(y=paramTrues[:, 0, 1][0], ls='--', c='b', alpha=0.75)
    ax[i].axhline(y=paramTrues[:, 1, 1][0], ls='--', c='orange', alpha=0.75)
    sns.regplot(x="Cell Number", y='Bern. G2 0', data=data_df, ax=ax[i], lowess=True, marker='+', scatter_kws=scatter_kws_list[0])
    sns.regplot(x="Cell Number", y='Bern. G2 1', data=data_df, ax=ax[i], lowess=True, marker='+', scatter_kws=scatter_kws_list[1])
    ax[i].set_title(r"G2 fate parameter estimation ($p$)")
    ax[i].set_ylabel(r"Bernoulli rate estimate ($p$)")
    ax[i].set_ylim(paramTrues[:, 1, 1][0]-0.025, 1.001)

    i += 1
    ax[i].axhline(y=paramTrues[:, 0, 4][0], ls='--', c='b', alpha=0.75)
    ax[i].axhline(y=paramTrues[:, 1, 4][0], ls='--', c='orange', alpha=0.75)
    sns.regplot(x="Cell Number", y='shape G2 0', data=data_df, ax=ax[i], lowess=True, marker='+', scatter_kws=scatter_kws_list[0])
    sns.regplot(x="Cell Number", y='shape G2 1', data=data_df, ax=ax[i], lowess=True, marker='+', scatter_kws=scatter_kws_list[1])
    ax[i].set_title(r"G2 lifetime parameter estimation ($k$, $\theta$)")
    ax[i].set_ylabel(r"Gamma shape estimate ($k$)")
    ax[i].set_ylim(0, 10)

    i += 1
    ax[i].axhline(y=paramTrues[:, 0, 5][0], ls='--', c='b', alpha=0.75)
    ax[i].axhline(y=paramTrues[:, 1, 5][0], ls='--', c='orange', alpha=0.75)
    sns.regplot(x="Cell Number", y='scale G2 0', data=data_df, ax=ax[i], lowess=True, marker='+', scatter_kws=scatter_kws_list[0])
    sns.regplot(x="Cell Number", y='scale G2 1', data=data_df, ax=ax[i], lowess=True, marker='+', scatter_kws=scatter_kws_list[1])
    ax[i].set_title(r"G2 lifetime parameter estimation ($k$, $\theta$)")
    ax[i].set_ylabel(r"Gamma scale estimate ($\theta$)")
    ax[i].set_ylim(0, 10)
