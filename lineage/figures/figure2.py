"""
File: figure2.py
Purpose: Generates figure 2.

Figure 2 is the distribution of cells in a state over generations (pruned).
"""
import numpy as np

from .figureCommon import getSetup
from ..LineageTree import LineageTree
from ..StateDistribution import StateDistribution, track_population_generation_histogram


def makeFigure():
    """
    Makes figure 2.
    """
    # pi: the initial probability vector
    pi = np.array([0.6, 0.4], dtype="float")

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

    # creating a population
    population_unpruned = []
    population_pruned = []
    desired_experiment_time = 300
    for i in range(20):
        population_unpruned.append(LineageTree(pi, T, E, (2**12) - 1, desired_experiment_time=desired_experiment_time, prune_condition='fate', prune_boolean=False))
        population_pruned.append(LineageTree(pi, T, E, (2**12) - 1, desired_experiment_time=desired_experiment_time, prune_condition='both', prune_boolean=True))

    hist_unpruned = track_population_generation_histogram(population_unpruned)
    hist_pruned = track_population_generation_histogram(population_pruned)

    # Get list of axis objects
    ax, f = getSetup((16, 16), (2, 2))
    
    # unpruned

    x_0_unp = [i for i in range(len(hist_unpruned[0]))]
    x_1_unp = [i for i in range(len(hist_unpruned[1]))]

    ax[0].set_xlim([-0.01, 12])
    ax[0].set_xlabel(r'Generation')
    ax[0].bar(x_0_unp, hist_unpruned[0], color='#F9Cb9C', label='Resistant')
    ax[0].bar(x_1_unp, hist_unpruned[1], bottom=hist_unpruned[0], color='#A4C2F4', label='Susceptible')
    ax[0].set_ylabel('Number of alive cells')
    ax[0].set_title('Unpruned population growth')
    ax[0].grid(linestyle='--')

    y_0_unp = [a/(a+b) for a, b in zip(hist_unpruned[0], hist_unpruned[1])]
    y_1_unp = [b/(a+b) for a, b in zip(hist_unpruned[0], hist_unpruned[1])]

    ax[1].set_xlim([-0.01, 12])
    ax[1].set_xlabel(r'Generation')
    ax[1].plot(x_0_unp, y_0_unp, color='#F9Cb9C', label='Resistant')
    ax[1].plot(x_1_unp, y_1_unp, color='#A4C2F4', label='Susceptible')
    ax[1].set_ylabel('Proportion of alive cells')
    ax[1].set_ylim([-0.01, 1.01])
    ax[1].set_title('Unpruned population distribution')
    ax[1].grid(linestyle='--')
    ax[1].legend()
    
    # pruned
    
    x_0_p = [i for i in range(len(hist_pruned[0]))]
    x_1_p = [i for i in range(len(hist_pruned[1]))]

    ax[2].set_xlim([-0.01, 12])
    ax[2].set_xlabel(r'Generation')
    ax[2].bar(x_0_p, hist_pruned[0], color='#F9Cb9C', label='Resistant')
    ax[2].bar(x_1_p, hist_pruned[1], bottom=hist_pruned[0], color='#A4C2F4', label='Susceptible')
    ax[2].set_ylabel('Number of alive cells')
    ax[2].set_title('Pruned population growth')
    ax[2].grid(linestyle='--')

    y_0_p = [a/(a+b) for a, b in zip(hist_pruned[0], hist_pruned[1])]
    y_1_p = [b/(a+b) for a, b in zip(hist_pruned[0], hist_pruned[1])]

    ax[3].set_xlim([-0.01, 12])
    ax[3].set_xlabel(r'Generation')
    ax[3].plot(x_0_p, y_0_p, color='#F9Cb9C', label='Resistant')
    ax[3].plot(x_1_p, y_1_p, color='#A4C2F4', label='Susceptible')
    ax[3].set_ylabel('Proportion of alive cells')
    ax[3].set_ylim([-0.01, 1.01])
    ax[3].set_title('Pruned population distribution')
    ax[3].grid(linestyle='--')
    ax[3].legend()

    f.tight_layout()

    return f
