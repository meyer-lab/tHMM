"""
File: figure6.py
Purpose: Generates figure 6.

Figure 12 is the KL divergence for different sets of parameters
for a two state model. It plots the KL-divergence against accuracy.
"""
import itertools
import random
import numpy as np
from scipy.stats import entropy
import pandas as pd
import seaborn as sns
from ..StateDistribution import StateDistribution
from ..LineageTree import LineageTree
from ..Analyze import get_results, run_Analyze_over
from .figureCommon import getSetup

def makeFigure():
    """
    Makes figure 12.
    """

    # Get list of axis objects
    ax, f = getSetup((20, 6), (1, 2))
    accuracyy, KL_gamma = KLdivergence()
    dists = distributionPlot()
    figure_maker(ax, accuracyy, KL_gamma, dists)

    return f


def KLdivergence():
    """ Assuming we have 2-state model """

    # pi: the initial probability vector
    pi = np.array([0.5, 0.5], dtype="float")

    # T: transition probability matrix
    T = np.array([[0.50, 0.50],
                  [0.50, 0.50]])

    a0 = np.linspace(5.0, 20.0, 10)
    
    kl_divs = []
    acc = []

    dists = pd.DataFrame(columns=['Distributions', 'hues'])
    list_of_populations = []
    for idx, a0 in enumerate(a0):
        state_obj0 = StateDistribution(0, 0.99, a0, 0, 2)
        state_obj1 = StateDistribution(1, 0.88, 4, 0, 3)

        E = [state_obj0, state_obj1]
        lineage = LineageTree(pi, T, E, (2**12) - 1, desired_experiment_time=600, prune_condition='fate', prune_boolean=False)
        while len(lineage.output_lineage) < 16:
            del lineage
            lineage = LineageTree(pi, T, E, (2**12) - 1, desired_experiment_time=600, prune_condition='fate', prune_boolean=False)
        list_of_populations.append([lineage])

        # First collect all the observations from the entire population across the lineages ordered by state
        obs_by_state = []
        for state in range(tHMMobj.numStates):
            full_list = [obs for obs in lineage.lineage_stats[state].full_lin_cells_obs]
            obs_by_state_rand_sampled = random.sample(full_list, 500)
        
        # Calculate their PDFs for input to the symmetric KL
        p = [tHMMobj.X[0].E[0].pdf(y) for y in obs_by_state[0]]
        q = [tHMMobj.X[0].E[1].pdf(x) for x in obs_by_state[1]]

        KL_value = entropy(p,q)+entropy(q,p)
        kl_divs.append(KL_value)
        dists["Distributions"] += ["{}".format(KL_value)]*
        
        
        
        
    return acc, gammaKL_total[0]


def distributionPlot():
    """ Here we plot the distributions used in the KL divergence plot
    to show how far the distributions are for each point.
    For 10 distributions, 500 random variables are generated
    and to be used by the violinplot they are concatenated in
    "lifetime" column of the DataFrame, and to separate them,
    there is another column named "distributions". """
    gamma_loc = 0
    a0 = np.linspace(5.0, 17.0, 10)
    scale0 = 2.0
    a1 = np.linspace(30.0, 17.0, 10)
    scale1 = 3.0
    total = []
    for i in range(10):
        a = list(sp.gamma.rvs(a=a0[9 - i], loc=gamma_loc,
                              scale=scale0,
                              size=500))
        total.append(a)
        b = list(sp.gamma.rvs(a=a1[9 - i], loc=gamma_loc,
                              scale=scale1,
                              size=500))
        total.append(b)

    TOdf = list(itertools.chain.from_iterable(total))

    dists = pd.DataFrame(columns=['distributions', 'hues'])
    dists['distributions'] = 1000 * (['d1']) + 1000 * (['d2']) + \
        1000 * (['d3']) + 1000 * (['d4']) + 1000 * (['d5']) + \
        1000 * (['d6']) + 1000 * (['d7']) + 1000 * (['d8']) + \
        1000 * (['d9']) + 1000 * (['d10'])
    dists['hues'] = 500 * [1] + 500 * [2] + 500 * [1] + \
        500 * [2] + 500 * [1] + 500 * [2] + 500 * [1] + 500 * [2] \
        + 500 * [1] + 500 * [2] + 500 * [1] + 500 * [2] + \
        500 * [1] + 500 * [2] + 500 * [1] + 500 * [2] + 500 * [1] + 500 * [2] \
        + 500 * [1] + 500 * [2]
    dists['lifetime [hr]'] = TOdf
    return dists


def figure_maker(ax, accuracyyy, KL_gamma, dists):
    """ makes the figure showing
    distributions get farther """

    i = 0
    ax[i].set_xlabel('KL divergence')
    ax[i].set_ylim(0, 110)
    ax[i].set_xlim(0, 1.07 * max(KL_gamma))
    ax[i].scatter(KL_gamma, accuracyyy, c='k', marker="o", edgecolors='k', alpha=0.25)
    ax[i].set_ylabel(r'Accuracy [\%]')
    ax[i].axhline(y=100, linestyle='--', linewidth=2, color='k', alpha=1)
    ax[i].set_title('KL divergence for two state model')
    ax[i].grid(linestyle='--')

    i += 1
    sns.violinplot(x="distributions", y="lifetime [hr]", inner="quart", palette="muted", split=True, hue="hues", data=dists, ax=ax[i])
    sns.despine(left=True, ax=ax[i])
