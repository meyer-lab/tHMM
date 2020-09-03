""" This file contains figures related to how big the experment needs to be. """
import numpy as np
import pandas as pd
import seaborn as sns
from .figureCommon import (
    getSetup,
    subplotLabel,
    pi,
    E2,
    T,
    min_desired_num_cells,
    min_num_lineages,
    max_num_lineages,
    num_data_points,
    scatter_kws_list,
)
from ..Analyze import run_Analyze_over, run_Results_over
from ..LineageTree import LineageTree


def makeFigure():
    """
    Makes fig 5.
    """

    # Get list of axis objects
    ax, f = getSetup((13, 6.66), (2, 4))
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
    num_lineages = np.linspace(3, max_num_lineages, num_data_points, dtype=int)
    experiment_times = np.linspace(1200, int(2.5 * 1000), num_data_points)
    list_of_populations = []
    for indx, num in enumerate(num_lineages):
        population = []
        for _ in range(num):
            tmp_lineage = LineageTree.init_from_parameters(pi, T, E2, desired_num_cells=min_desired_num_cells, censor_condition=3, desired_experiment_time=experiment_times[indx])
            if len(tmp_lineage.output_lineage) < 3:
                pass
            else:
                population.append(tmp_lineage)

        # Adding populations into a holder for analysing
        list_of_populations.append(population)

    output = run_Analyze_over(list_of_populations, 2)

    # Collecting the results of analyzing the lineages
    results_holder = run_Results_over(output)

    dictOut = {}

    for key in results_holder[0].keys():
        dictOut[key] = []

    for results_dict in results_holder:
        for key, val in results_dict.items():
            dictOut[key].append(val)

    paramEst = np.array(dictOut["param_estimates"])
    paramTrues = np.array(dictOut["param_trues"])
    accuracy_df = pd.DataFrame(columns=["Cell Number", 'State Assignment Accuracy'])
    accuracy_df['Cell Number'] = dictOut['total_number_of_cells']
    accuracy_df['State Assignment Accuracy'] = dictOut['balanced_accuracy_score']

    param_df = pd.DataFrame(columns=["Cell Number", "Lineage Number", "T", "pi"])
    param_df["Cell Number"] = accuracy_df["Cell Number"].to_list()
    param_df["Lineage Number"] = num_lineages
    param_df['T Error'] = dictOut['transition_matrix_norm']
    param_df['pi Error'] = dictOut['pi_vector_norm']

    data_df = pd.DataFrame(columns=["Cell Number"])
    data_df["Cell Number"] = accuracy_df["Cell Number"].to_list()
    data_df['Bern. G1 0'] = paramEst[:, 0, 0]
    data_df['Bern. G1 1'] = paramEst[:, 1, 0]
    data_df['Bern. G2 0'] = paramEst[:, 0, 1]
    data_df['Bern. G2 1'] = paramEst[:, 1, 1]
    data_df['wasserstein distance 0'] = dictOut["distribution distance 0"]
    data_df['wasserstein distance 1'] = dictOut["distribution distance 1"]

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
    ax[i].set_title(r"Error in estimating $T$")
    ax[i].set_ylabel(r"Error [$||x-\hat{x}||$]")
    ax[i].set_ylim(bottom=0.01, top=1.02)

    i += 1
    sns.regplot(x="Lineage Number", y="pi Error", data=param_df, ax=ax[i], lowess=True, marker='+', scatter_kws=scatter_kws_list[0])
    ax[i].set_title(r"Error in estimating $\pi$")
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
    ax[i].set_ylim(paramTrues[:, 1, 0][0] - 0.025, 1.001)

    i += 1
    ax[i].axhline(y=paramTrues[:, 0, 1][0], ls='--', c='b', alpha=0.75)
    ax[i].axhline(y=paramTrues[:, 1, 1][0], ls='--', c='orange', alpha=0.75)
    sns.regplot(x="Cell Number", y='Bern. G2 0', data=data_df, ax=ax[i], lowess=True, marker='+', scatter_kws=scatter_kws_list[0])
    sns.regplot(x="Cell Number", y='Bern. G2 1', data=data_df, ax=ax[i], lowess=True, marker='+', scatter_kws=scatter_kws_list[1])
    ax[i].set_title(r"G2 fate parameter estimation ($p$)")
    ax[i].set_ylabel(r"Bernoulli rate estimate ($p$)")
    ax[i].set_ylim(paramTrues[:, 1, 1][0] - 0.025, 1.001)

    i += 1
    sns.regplot(x="Cell Number", y='wasserstein distance 0', data=data_df, ax=ax[i], lowess=True, label="state 1", marker='+', scatter_kws=scatter_kws_list[0])
    sns.regplot(x="Cell Number", y='wasserstein distance 1', data=data_df, ax=ax[i], lowess=True, label="state 2", marker='+', scatter_kws=scatter_kws_list[1])
    ax[i].set_title(r"distance bw true and estm. gamma dists")
    ax[i].set_ylabel(r"wasserstein distance")
    ax[i].set_ylim(0.0, 30.0)
    ax[i].legend()

    i += 1
    ax[i].axis('off')
