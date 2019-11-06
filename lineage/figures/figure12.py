"""
File: figure12.py
Purpose: Generates figure 12.

Figure 12 is the KL divergence for different sets of parameters for a two state model. It plots the KL-divergence against accuracy.
"""
import numpy as np

from .figureCommon import getSetup
from ..Analyze import accuracy, Analyze, kl_divergence
from ..LineageTree import LineageTree
from ..StateDistribution import StateDistribution


def makeFigure():
    """
    Makes figure 12.
    """

    # Get list of axis objects
    ax, f = getSetup((8, 6), (1, 1))
    acc, gammaKL_total = KLdivergence()
    figure_maker(ax, acc, gammaKL_total)

    return f


def KLdivergence():
    """ Assuming we have 2-state model """

    # pi: the initial probability vector
    pi = np.array([0.5, 0.5], dtype="float")

    # T: transition probability matrix
    T = np.array([[0.50, 0.50],
                  [0.50, 0.50]])

    a0 = [5.0, 10.0, 15.0, 12.0]
    scale0 = [2.0, 2.0, 2.0, 3.3]
    a1 = [23.0, 20.0, 17.0, 12.0]
    scale1 = [3.0, 3.0, 3.0, 3.3]

    gammaKL1 = []
    gammaKL_total = []
    acc1 = []
    acc_total = []
    acc = []

    assert len(a0) == len(scale0) == len(a1) == len(scale1), "the length of the parameters are not the same!"

    for i in range(len(a0)):
        state_obj0 = StateDistribution(state0, bern_p0, a0[i], gamma_loc, scale0[i])
        state_obj1 = StateDistribution(state1, bern_p1, a1[i], gamma_loc, scale1[i])

        E = [state_obj0, state_obj1]
        lineage = LineageTree(pi, T, E, (2**12) - 1, desired_experiment_time=1000, prune_condition='both', prune_boolean=True)
        while len(lineage.output_lineage) < 32:
            del lineage
            lineage = LineageTree(pi, T, E, (2**12) - 1, desired_experiment_time=1000, prune_condition='both', prune_boolean=True)

        _, obs0 = list(zip(*lineage.lineage_stats[0].full_lin_cells_obs))
        _, obs1 = list(zip(*lineage.lineage_stats[1].full_lin_cells_obs))

        p = scipy.stats.gamma.pdf(obs0, a=a0[i], loc=gamma_loc, scale=scale0[i])
        q = scipy.stats.gamma.pdf(obs1, a=a0[i], loc=gamma_loc, scale=scale1[i])

        size = min(p.shape[0], q.shape[0])
        if size == 0:
            raise ValueError('the number of cells predicted in one of the states is zero! ')
        else:
            pprime = random.sample(list(p), size)
            qprime = random.sample(list(q), size)
        gammaKL1.append(kl_divergence(np.asarray(pprime), np.asarray(qprime)))

        X = [lineage]
        states = [cell.state for cell in lineage.output_lineage]
        for j in range(10):
            _, _, all_states, tHMMobj, _, _ = Analyze(X, 2)

            # find the accuracy
            temp = accuracy(tHMMobj, all_states)[0] * 100

            acc1.append(temp)
    gammaKL_total.append(gammaKL1)

    for j in range(4):
        tmp = np.sum(acc1[j:10 * (j + 1)]) / len(acc1[j:10 * (j + 1)])
        acc.append(tmp)
    return acc, gammaKL_total


def figure_maker(ax, x, KL_gamma, accuracy)


i = 0
#     ax[i].set_xlim((16, int(np.ceil(4 * max(x)))))
ax[i].set_xlabel('KL divergence')
ax[i].set_ylim(0, 110)
ax[i].scatter(KL_gamma, accuracy, c='k', marker="o", edgecolors='k', alpha=0.25)
ax[i].plot(KL_gamma, accuracy, c='k')
#     ax[i].set_xscale('log', basex=2)
ax[i].set_ylabel(r'Accuracy [\%]')
ax[i].axhline(y=100, linestyle='--', linewidth=2, color='k', alpha=1)
ax[i].set_title('KL divergence for two state model')
ax[i].grid(linestyle='--')
ax[i].tick_params(axis='both', which='major', grid_alpha=0.25)
