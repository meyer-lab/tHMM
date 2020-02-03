"""
File: figure6.py
Purpose: Generates figure 6.

Figure 12 is the KL divergence for different sets of parameters
for a two state model. It plots the KL-divergence against accuracy.
"""
import random
import numpy as np
from scipy.stats import wasserstein_distance
import pandas as pd
import seaborn as sns
from ..StateDistribution import StateDistribution
from ..LineageTree import LineageTree
from ..Analyze import run_Results_over, run_Analyze_over
from .figureCommon import getSetup


def makeFigure():
    """
    Makes figure 12.
    """

    # Get list of axis objects
    ax, f = getSetup((20, 6), (1, 2))
    accuracies, w_divs_to_use, dists = wasserstein()
    figure_maker(ax, accuracies, w_divs_to_use, dists)

    return f


def wasserstein():
    """ Assuming we have 2-state model """

    # pi: the initial probability vector
    pi = np.array([0.5, 0.5], dtype="float")

    # T: transition probability matrix
    T = np.array([[0.66, 0.33],
                  [0.33, 0.66]])

    a0 = np.logspace(2, 5, 5, base=2)

    state_obj0 = StateDistribution(1, 0.99, 4, 0, 3)

    w_divs = []

    dists = pd.DataFrame(columns=["Lifetimes [hr]", "Distributions", "Hues"])
    tmp_lifetimes = []
    tmp_distributions = []
    tmp_hues = []
    list_of_populations_unsort = []
    for idx, a0 in enumerate(a0):
        state_obj1 = StateDistribution(0, 0.99, a0, 0, 3)

        E = [state_obj0, state_obj1]
        lineage = LineageTree(pi, T, E, (2**12) - 1, desired_experiment_time=1000000000, prune_condition='fate', prune_boolean=False)
        while len(lineage.output_lineage) < 16:
            del lineage
            lineage = LineageTree(pi, T, E, (2**12) - 1, desired_experiment_time=1000000000, prune_condition='fate', prune_boolean=False)
        list_of_populations_unsort.append([lineage])

        # First collect all the observations from the entire population across the lineages ordered by state
        obs_by_state_rand_sampled = []
        for state in range(len(E)):
            full_list = [obs[1] for obs in lineage.lineage_stats[state].full_lin_cells_obs]
            obs_by_state_rand_sampled.append(random.sample(full_list, 750))

        w_value = wasserstein_distance(obs_by_state_rand_sampled[0], obs_by_state_rand_sampled[1])
        w_divs.append(w_value)
        tmp_lifetimes.append(obs_by_state_rand_sampled[0] + obs_by_state_rand_sampled[1])
        tmp_distributions.append(["{}".format(round(w_value, 2))] * 750 * 2)
        tmp_hues.append([1] * 750 + [2] * 750)

    # Change the order of lists
    indices = np.argsort(w_divs)

    w_divs_to_use = [w_divs[idx] for idx in indices]

    dists["Lifetimes [hr]"] = sum([tmp_lifetimes[idx] for idx in indices], [])
    dists["Distributions"] = sum([tmp_distributions[idx] for idx in indices], [])
    dists["Hues"] = sum(tmp_hues, [])
    list_of_populations = [list_of_populations_unsort[idx] for idx in indices]

    # Analyzing the lineages in the list of populations (parallelized function)
    output = run_Analyze_over(list_of_populations, 2)

    # Collecting the results of analyzing the lineages
    results_holder = run_Results_over(output)

    accuracies = [results_dict["accuracy_after_switching"] for results_dict in results_holder]

    return accuracies, w_divs_to_use, dists


def figure_maker(ax, accuracies, w_divs, dists):
    """ makes the figure showing
    distributions get farther """

    i = 0
    ax[i].set_xlabel('KL divergence')
    ax[i].set_ylim(0, 110)
    ax[i].set_xlim(0, 1.07 * max(kl_divs))
    ax[i].scatter(w_divs, accuracies, c='k', marker="o", edgecolors='k', alpha=0.25)
    ax[i].set_ylabel(r'Accuracy [\%]')
    ax[i].axhline(y=100, linestyle='--', linewidth=2, color='k', alpha=1)
    ax[i].set_title('Wasserstein divergence')
    ax[i].grid(linestyle='--')

    i += 1
    sns.violinplot(x="Distributions", y="Lifetimes [hr]", inner="quart", palette="muted", split=True, hue="Hues", data=dists, ax=ax[i], order=["{}".format(round(w_value, 2)) for w_value in w_divs])
    sns.despine(left=True, ax=ax[i])
