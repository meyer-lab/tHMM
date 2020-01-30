"""
File: figure6.py
Purpose: Generates figure 6.

Figure 12 is the KL divergence for different sets of parameters
for a two state model. It plots the KL-divergence against accuracy.
"""
import random
import numpy as np
from scipy.stats import entropy
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
    accuracies, kl_divs, dists = KLdivergence()
    figure_maker(ax, accuracies, kl_divs, dists)

    return f


def KLdivergence():
    """ Assuming we have 2-state model """

    # pi: the initial probability vector
    pi = np.array([0.5, 0.5], dtype="float")

    # T: transition probability matrix
    T = np.array([[0.50, 0.50],
                  [0.50, 0.50]])

    a0 = np.linspace(5.0, 20.0, 10)

    state_obj1 = StateDistribution(1, 0.88, 4, 0, 3)

    kl_divs = []

    dists = pd.DataFrame(columns=["Lifetimes [hr]", "Distributions", "Hues"])
    tmp_lifetimes = []
    tmp_distributions = []
    tmp_hues = []
    list_of_populations = []
    for idx, a0 in enumerate(a0):
        state_obj0 = StateDistribution(0, 0.99, a0, 0, 2)

        E = [state_obj0, state_obj1]
        lineage = LineageTree(pi, T, E, (2**12) - 1, desired_experiment_time=600, prune_condition='fate', prune_boolean=False)
        while len(lineage.output_lineage) < 16:
            del lineage
            lineage = LineageTree(pi, T, E, (2**12) - 1, desired_experiment_time=600, prune_condition='fate', prune_boolean=False)
        list_of_populations.append([lineage])

        # First collect all the observations from the entire population across the lineages ordered by state
        obs_by_state_rand_sampled = []
        for state in range(len(E)):
            full_list = [obs for obs in lineage.lineage_stats[state].full_lin_cells_obs]
            obs_by_state_rand_sampled.append(random.sample(full_list, 500))

        # Calculate their PDFs for input to the symmetric KL
        p = [E[0].pdf(y) for y in obs_by_state_rand_sampled[0]]
        q = [E[1].pdf(x) for x in obs_by_state_rand_sampled[1]]

        KL_value = entropy(p, q) + entropy(q, p)
        kl_divs.append(KL_value)
        tmp_lifetimes.append(([b for a, b in obs_by_state_rand_sampled[0]] + [b for a, b in obs_by_state_rand_sampled[1]]))
        tmp_distributions.append(["{}".format(KL_value)] * 500 * 2)
        tmp_hues.append([1] * 500 + [2] * 500)

    dists["Lifetimes [hr]"] = sum(tmp_lifetimes, [])
    dists["Distributions"] = sum(tmp_distributions, [])
    dists["Hues"] = sum(tmp_hues, [])

    # Analyzing the lineages in the list of populations (parallelized function)
    output = run_Analyze_over(list_of_populations, 2)

    # Collecting the results of analyzing the lineages
    results_holder = run_Results_over(output)

    accuracies = [results_dict["accuracy_after_switching"] for results_dict in results_holder]

    return accuracies, kl_divs, dists


def figure_maker(ax, accuracies, kl_divs, dists):
    """ makes the figure showing
    distributions get farther """

    i = 0
    ax[i].set_xlabel('KL divergence')
    ax[i].set_ylim(0, 110)
    ax[i].set_xlim(0, 1.07 * max(KL_gamma))
    ax[i].scatter(kl_divs, accuracies, c='k', marker="o", edgecolors='k', alpha=0.25)
    ax[i].set_ylabel(r'Accuracy [\%]')
    ax[i].axhline(y=100, linestyle='--', linewidth=2, color='k', alpha=1)
    ax[i].set_title('KL divergence for two state model')
    ax[i].grid(linestyle='--')

    i += 1
    sns.violinplot(x="Distributions", y="Lifetimes [hr]", inner="quart", palette="muted", split=True, hue="Hues", data=dists, ax=ax[i])
    sns.despine(left=True, ax=ax[i])
