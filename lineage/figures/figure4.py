"""
This creates Figure 4.
"""
from .figureCommon import subplotLabel, getSetup
from matplotlib.ticker import MaxNLocator
from ..Analyze import accuracy, accuracyG, Analyze
from ..LineageTree import LineageTree
from ..StateDistribution import StateDistribution
from ..StateDistribution2 import StateDistribution2

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
# for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
plt.rc('text', usetex=True)
mpl.rcParams.update({'font.size': 60})

def makeFigure():
    """ makes figure 1 """
    # Get list of axis objects
    ax, f = getSetup((12, 4), (1, 3))
    x, bern, bern_p0, gamma_a, gamma_a0, gamma_scale, gamma_scale0 = accuracy_increased_cells()
    figure_maker(ax, x, bern, bern_p0, gamma_a, gamma_a0, gamma_scale, gamma_scale0)
    
    return f

def accuracy_increased_cells():
    """
    Calculates parameter estimation for a one state model.
    """

    # pi: the initial probability vector
    piiii = np.array([1.0, 0.0], dtype="float")

    # T: transition probability matrix
    T = np.array([[1.0, 0.0],
                  [0.0, 1.0]], dtype="float")

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

    desired_num_cells = np.logspace(5, 10, num=200, base=2.0)
    desired_num_cells = [num_cell - 1 for num_cell in desired_num_cells]

    x = []
    bern = []
    gamma_a = []
    gamma_scale = []

    for num in desired_num_cells:
        # Creating an unpruned and pruned lineage
        lineage = LineageTree(piiii, T, E, num, desired_experiment_time=1000000, prune_condition='fate', prune_boolean=False)

        # Setting then into a list or a population of lineages and collecting the length of each lineage
        X1 = [lineage]
        x.append(len(lineage.output_lineage))

        # Analyzing the lineages
        deltas, _, all_states, tHMMobj, _, _ = Analyze(X1, 1)

        bern.append(tHMMobj.estimate.E[0].bern_p)
        gamma_a.append(tHMMobj.estimate.E[0].gamma_a)
        gamma_scale.append(tHMMobj.estimate.E[0].gamma_scale)
        
    return x, bern, bern_p0, gamma_a, gamma_a0, gamma_scale, gamma_scale0


def figure_maker(ax, x, bern, bern_p0, gamma_a, gamma_a0, gamma_scale, gamma_scale0):
    
    i = 0
    ax[i].set_xlim((0, int(np.ceil(1.1 * max(x)))))
    ax[i].set_xlabel('Number of Cells')
    ax[i].scatter(x, bern, c='#F9Cb9C', marker="o", alpha=0.75)
    ax[i].set_ylabel(r'Bernoulli Parameter $p$')
    ax[i].axhline(y=bern_p0, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='#F9Cb9C', alpha=0.6)
    ax[i].set_title('Bernoulli')
    ax[i].grid(linestyle='--')
    ax[i].tick_params(axis='both', which='major', labelsize=10, grid_alpha=0.25)

    i += 1
    ax[i].set_xlim((0, int(np.ceil(1.1 * max(x)))))
    ax[i].set_xlabel('Number of Cells')
    ax[i].scatter(x, gamma_a , c='#F9Cb9C', marker="o", alpha=0.75)
    ax[i].set_ylabel(r'Gamma $k$')
    ax[i].axhline(y=gamma_a0, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='#F9Cb9C', alpha=0.6)
    ax[i].set_title('Gamma')
    ax[i].grid(linestyle='--')
    ax[i].tick_params(axis='both', which='major', labelsize=10, grid_alpha=0.25)

    i += 1
    ax[i].set_xlim((0, int(np.ceil(1.1 * max(x)))))
    ax[i].set_xlabel('Number of Cells')
    ax[i].scatter(x, gamma_scale, c='#F9Cb9C', marker="o", alpha=0.75)
    ax[i].set_ylabel(r'Gamma $\theta$')
    ax[i].axhline(y=gamma_scale0, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='#F9Cb9C', alpha=0.6)
    ax[i].set_title('Gamma')
    ax[i].grid(linestyle='--')
    ax[i].tick_params(axis='both', which='major', labelsize=10, grid_alpha=0.25)
