"""
File: figure12.py
Purpose: Generates figure 12.

Figure 12 is the KL divergence for different sets of parameters
for a two state model. It plots the KL-divergence against accuracy.
"""
from ..StateDistribution import StateDistribution
from ..LineageTree import LineageTree
from ..Analyze import accuracy, Analyze
from .figureCommon import getSetup
import numpy as np
import random
import scipy.stats as sp
import pandas as pd
import seaborn as sns
sns.set(style="whitegrid", palette="pastel", color_codes=True)


def makeFigure():
    """
    Makes figure 12.
    """

    # Get list of axis objects
    ax, f = getSetup((20, 6), (1, 2))
    accuracy, KL_gamma = KLdivergence()
    dists = distributionPlot()
    figure_maker(ax, accuracy, KL_gamma, dists)

    return f


def KLdivergence():
    """ Assuming we have 2-state model """

    # pi: the initial probability vector
    pi = np.array([0.5, 0.5], dtype="float")

    # T: transition probability matrix
    T = np.array([[0.50, 0.50],
                  [0.50, 0.50]])

    state0 = 0
    state1 = 1
    gamma_loc = 0
    bern_p0 = 0.99
    bern_p1 = 0.88
    a0 = np.linspace(5.0, 17.0, 10)
    scale0 = 10 * ([2.0])
    a1 = np.linspace(30.0, 17.0, 10)
    scale1 = 10 * ([3.0])

    gammaKL1 = []
    gammaKL_total = []
    acc1 = []
    acc_total = []
    acc = []

    assert len(a0) == len(scale0) == len(a1) == len(scale1), "parameter length problem!"

    for i in range(a0.shape[0]):
        state_obj0 = StateDistribution(state0,
                                       bern_p0,
                                       a0[i],
                                       gamma_loc,
                                       scale0[i])
        state_obj1 = StateDistribution(state1,
                                       bern_p1,
                                       a1[i],
                                       gamma_loc,
                                       scale1[i])

        E = [state_obj0, state_obj1]
        lineage = LineageTree(pi, T, E, (2**12) - 1,
                              desired_experiment_time=600,
                              prune_condition='both',
                              prune_boolean=True)
        while len(lineage.output_lineage) < 16:
            del lineage
            lineage = LineageTree(pi, T, E, (2**12) - 1,
                                  desired_experiment_time=600,
                                  prune_condition='both',
                                  prune_boolean=True)

        _, obs0 = list(zip(*lineage.lineage_stats[0].full_lin_cells_obs))
        _, obs1 = list(zip(*lineage.lineage_stats[1].full_lin_cells_obs))

        p = sp.gamma.pdf(obs0, a=a0[i], loc=gamma_loc, scale=scale0[i])
        q = sp.gamma.pdf(obs1, a=a0[i], loc=gamma_loc, scale=scale1[i])

        size = min(p.shape[0], q.shape[0])
        if size == 0:
            raise ValueError('the number of cells predicted in one of the states is zero!')
        else:
            pprime = random.sample(list(p), size)
            qprime = random.sample(list(q), size)
            # find the KL divergence
        gammaKL1.append(sp.entropy(np.asarray(pprime), np.asarray(qprime)))

        X = [lineage]
        states = [cell.state for cell in lineage.output_lineage]
        num_iter = 5  # for every KL value, it runs the model 5 times
        # to get the accuracy and returns the average accuracy for 5 iterations.
        for j in range(num_iter):
            _, _, all_states, tHMMobj, _, _ = Analyze(X, 2)

            # find the accuracy
            temp = accuracy(tHMMobj, all_states)[0] * 100

            acc1.append(temp)
    gammaKL_total.append(gammaKL1)

    for j in range(10):
        tmp = np.sum(acc1[j:num_iter * (j + 1)]) \
            / len(acc1[j:num_iter * (j + 1)])
        acc.append(tmp)
    return acc, gammaKL_total[0]


def distributionPlot():
    """ Here we plot the distributions used in the KL divergence plot
    to show how far the distributions are for each point.
    For 10 distributions, 500 random variables are generated
    and to be used by the violinplot they are concatenated in
    "lifetime" column of the DataFrame, and to separate them,
    there is another column named "distributions". """
    gamma_loc = 0
    bern_p0 = 0.99
    bern_p1 = 0.88
    a0 = np.linspace(5.0, 17.0, 10)
    scale0 = 10 * ([2.0])
    a1 = np.linspace(30.0, 17.0, 10)
    scale1 = 10 * ([3.0])

    dist1 = np.concatenate((sp.gamma.rvs(a=a0[9], loc=gamma_loc,
                                         scale=scale0[9],
                                         size=500),
                            sp.gamma.rvs(a=a1[9], loc=gamma_loc,
                                         scale=scale1[9],
                                         size=500),
                            sp.gamma.rvs(a=a0[8], loc=gamma_loc,
                                         scale=scale0[8],
                                         size=500),
                            sp.gamma.rvs(a=a1[8],
                                         loc=gamma_loc, scale=scale1[8],
                                         size=500),
                            sp.gamma.rvs(a=a0[7], loc=gamma_loc,
                                         scale=scale0[7], size=500),
                            sp.gamma.rvs(a=a1[7], loc=gamma_loc,
                                         scale=scale1[7], size=500),
                            sp.gamma.rvs(a=a0[6], loc=gamma_loc,
                                         scale=scale0[6], size=500),
                            sp.gamma.rvs(a=a1[6], loc=gamma_loc,
                                         scale=scale1[6], size=500),
                            sp.gamma.rvs(a=a0[5], loc=gamma_loc,
                                         scale=scale0[5], size=500),
                            sp.gamma.rvs(a=a1[5], loc=gamma_loc,
                                         scale=scale1[5], size=500),
                            sp.gamma.rvs(a=a0[4], loc=gamma_loc,
                                         scale=scale0[4], size=500),
                            sp.gamma.rvs(a=a1[4], loc=gamma_loc,
                                         scale=scale1[4], size=500),
                            sp.gamma.rvs(a=a0[3], loc=gamma_loc,
                                         scale=scale0[3], size=500),
                            sp.gamma.rvs(a=a1[3], loc=gamma_loc,
                                         scale=scale1[3], size=500),
                            sp.gamma.rvs(a=a0[2],
                                         loc=gamma_loc, scale=scale0[2],
                                         size=500),
                            sp.gamma.rvs(a=a1[2], loc=gamma_loc,
                                         scale=scale1[2], size=500),
                            sp.gamma.rvs(a=a0[1], loc=gamma_loc,
                                         scale=scale0[1], size=500),
                            sp.gamma.rvs(a=a1[1],
                                         loc=gamma_loc, scale=scale1[1], size=500),
                            sp.gamma.rvs(a=a0[0], loc=gamma_loc, scale=scale0[0],
                                         size=500),
                            sp.gamma.rvs(a=a1[0], loc=gamma_loc,
                                         scale=scale1[0], size=500)))

    dists = pd.DataFrame(columns=['lifetime [hr]', 'distributions', 'hues'])
    dists['lifetime [hr]'] = dist1
    dists['distributions'] = 1000 * (['d1']) + 1000 * (['d2']) + \
        1000 * (['d3']) + 1000 * (['d4']) + 1000 * (['d5']) + \
        1000 * (['d6']) + 1000 * (['d7']) + 1000 * (['d8']) + \
        1000 * (['d9']) + 1000 * (['d10'])
    dists['hues'] = 500 * [1] + 500 * [2] + 500 * [1] + \
        500 * [2] + 500 * [1] + 500 * [2] + 500 * [1] + 500 * [2] \
        + 500 * [1] + 500 * [2] + 500 * [1] + 500 * [2] + \
        500 * [1] + 500 * [2] + 500 * [1] + 500 * [2] + 500 * [1] + 500 * [2] \
        + 500 * [1] + 500 * [2]
    return dists


def figure_maker(ax, accuracy, KL_gamma, dists):

    i = 0
    ax[i].set_xlabel('KL divergence')
    ax[i].set_ylim(0, 110)
    ax[i].set_xlim(0, 1.07 * max(KL_gamma))
    ax[i].scatter(KL_gamma, accuracy, c='k', marker="o", edgecolors='k', alpha=0.25)
    ax[i].set_ylabel(r'Accuracy [\%]')
    ax[i].axhline(y=100, linestyle='--', linewidth=2, color='k', alpha=1)
    ax[i].set_title('KL divergence for two state model')
    ax[i].grid(linestyle='--')

    i += 1

    sns.violinplot(x="distributions", y="lifetime [hr]",
                   inner="quart", palette="muted",
                   split=True, hue="hues", data=dists, ax=ax[i])
    sns.despine(left=True, ax=ax[i])
