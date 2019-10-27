"""
File: figure8.py
Purpose: Generates figure 8. 

Figure 8 is the parameter estimation for a single pruned lineage with heterogeneity (two true states). 
"""
from .figureCommon import  getSetup
from ..Analyze import accuracy, Analyze
from ..LineageTree import LineageTree
from ..StateDistribution import StateDistribution

import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', **{'family': 'sans-serif', 'size': 25})
# for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
plt.rc('text', usetex=True)
plt.rc('xtick', **{'labelsize':'medium'})
plt.rc('ytick', **{'labelsize':'medium'})


def makeFigure():
    """
    Makes figure 8.
    """

    # Get list of axis objects
    ax, f = getSetup((21, 6), (1, 3))
    x, bern_pruned, bern_p0, bern_p1, gamma_a_pruned, gamma_a0, gamma_a1, gamma_scale_pruned, gamma_scale0, gamma_scale1 = accuracy_increased_cells()
    figure_maker(ax, x, bern_pruned, bern_p0, bern_p1, gamma_a_pruned, gamma_a0, gamma_a1, gamma_scale_pruned, gamma_scale0, gamma_scale1)
    
    return f

def accuracy_increased_cells():
    """ 
    Calculates parameter estimation by increasing the number of cells in a lineage for a two-state model. 
    """

    # pi: the initial probability vector
    piiii = np.array([0.6, 0.4], dtype="float")

    # T: transition probability matrix
    T = np.array([[0.85, 0.15],
                  [0.15, 0.85]], dtype="float")

    # State 0 parameters "Resistant"
    state0 = 0
    bern_p0 = 0.99
    gamma_a0 = 20
    gamma_loc = 0
    gamma_scale0 = 5

    # State 1 parameters "Susceptible"
    state1 = 1
    bern_p1 = 0.88
    gamma_a1 = 10
    gamma_scale1 = 1

    state_obj0 = StateDistribution(state0, bern_p0, gamma_a0, gamma_loc, gamma_scale0)
    state_obj1 = StateDistribution(state1, bern_p1, gamma_a1, gamma_loc, gamma_scale1)
    E = [state_obj0, state_obj1]

    x = []
    bern_pruned = []  
    gamma_a_pruned = []
    gamma_scale_pruned = []
    
    
    times = np.linspace(100, 1000, 25)

    for experiment_time in times:
        lineage = LineageTree(piiii, T, E, (2**12)-1, experiment_time, prune_condition='both', prune_boolean=True)
        while len(lineage.output_lineage) < 16:
            del lineage
            lineage = LineageTree(piiii, T, E, (2**12)-1, experiment_time, prune_condition='both', prune_boolean=True)

        # Setting then into a list or a population of lineages and collecting the length of each lineage
        X1 = [lineage]
        x.append(len(lineage.output_lineage))

        # Analyzing the lineages
        deltas, _, all_states, tHMMobj, _, _ = Analyze(X1, 2)

        # Collecting the accuracies of the lineages
        acc1 = accuracy(tHMMobj, all_states)[0]*100
        while acc1 < 50:
            # Analyzing the lineages
            deltas, _, all_states, tHMMobj, _, _ = Analyze(X1, 2)

            # Collecting the accuracies of the lineages
            acc1 = accuracy(tHMMobj, all_states)[0]*100

        # Collecting the parameter estimations
        bern_p_total = ()
        gamma_a_total = ()
        gamma_scale_total = ()

        for state in range(tHMMobj.numStates):
            bern_p_total += (tHMMobj.estimate.E[state].bern_p,)
            gamma_a_total += (tHMMobj.estimate.E[state].gamma_a,)
            gamma_scale_total += (tHMMobj.estimate.E[state].gamma_scale,)


        bern_pruned.append(bern_p_total)
        gamma_a_pruned.append(gamma_a_total)
        gamma_scale_pruned.append(gamma_scale_total)

        
    return x, bern_pruned, bern_p0, bern_p1, gamma_a_pruned, gamma_a0, gamma_a1, gamma_scale_pruned, gamma_scale0, gamma_scale1


def figure_maker(ax, x, bern_pruned, bern_p0, bern_p1, gamma_a_pruned, gamma_a0, gamma_a1, gamma_scale_pruned, gamma_scale0, gamma_scale1):
    """
    Makes figure 8.
    """
    i = 0
    res = [[i for i, j in bern_pruned], [j for i, j in bern_pruned]]
    ax[i].set_xlim((16, int(np.ceil(4 * max(x)))))
    ax[i].set_xlabel('Number of Cells')
    ax[i].scatter(x, res[0], c='#F9Cb9C', edgecolors='k', marker="o", alpha=0.5)
    ax[i].scatter(x, res[1], c='#A4C2F4', edgecolors='k', marker="o", alpha=0.5)   
    ax[i].set_ylabel('Bernoulli $p$')
    ax[i].set_ylim([0.85,1.1])
    ax[i].axhline(y=bern_p0, linestyle='--', linewidth=2, label = 'Resistant', color='#F9Cb9C', alpha=1)
    ax[i].axhline(y=bern_p1, linestyle='--', linewidth=2, label = 'Susceptible', color='#A4C2F4', alpha=1)
    ax[i].set_title(r'Bernoulli $p$')
    ax[i].grid(linestyle='--')
    ax[i].set_xscale('log', basex=2)
    ax[i].tick_params(axis='both', which='major', grid_alpha=0.25)

    i += 1
    res = [[i for i, j in gamma_a_pruned], [j for i, j in gamma_a_pruned]]
    ax[i].set_xlim((16, int(np.ceil(4 * max(x)))))
    ax[i].set_xlabel('Number of Cells')
    ax[i].scatter(x, res[0], c='#F9Cb9C', edgecolors='k', marker="o", alpha=0.5)
    ax[i].scatter(x, res[1], c='#A4C2F4', edgecolors='k', marker="o", alpha=0.5)
    ax[i].set_ylabel(r'Gamma $k$')
    ax[i].set_ylim([5,25])
    ax[i].axhline(y=gamma_a0, linestyle='--', linewidth=2, label = 'Resistant', color='#F9Cb9C', alpha=1)
    ax[i].axhline(y=gamma_a1, linestyle='--', linewidth=2, label = 'Susceptible', color='#A4C2F4', alpha=1)
    ax[i].set_title(r'Gamma $k$')
    ax[i].grid(linestyle='--')
    ax[i].set_xscale('log', basex=2)
    ax[i].tick_params(axis='both', which='major', grid_alpha=0.25)

    i += 1
    res = [[i for i, j in gamma_scale_pruned], [j for i, j in gamma_scale_pruned]]
    ax[i].set_xlim((16, int(np.ceil(4 * max(x)))))
    ax[i].set_xlabel('Number of Cells')
    ax[i].scatter(x, res[0], c='#F9Cb9C', edgecolors='k', marker="o", alpha=0.5)
    ax[i].scatter(x, res[1], c='#A4C2F4', edgecolors='k', marker="o", alpha=0.5)
    ax[i].set_ylabel(r'Gamma $\theta$')
    ax[i].set_ylim([0,7])
    ax[i].axhline(y=gamma_scale0, linestyle='--', linewidth=2, label = 'Resistant', color='#F9Cb9C', alpha=1)
    ax[i].axhline(y=gamma_scale1, linestyle='--', linewidth=2, label = 'Susceptible', color='#A4C2F4', alpha=1)
    ax[i].set_title(r'Gamma $\theta$')
    ax[i].grid(linestyle='--')
    ax[i].set_xscale('log', basex=2)
    ax[i].tick_params(axis='both', which='major', grid_alpha=0.25)
    ax[i].legend()

    
    
    
    
