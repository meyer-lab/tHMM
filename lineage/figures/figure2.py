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
    population = []
    desired_experiment_time = 300
    for i in range(20):
        population.append(LineageTree(pi, T, E, (2**12) - 1, desired_experiment_time=desired_experiment_time, prune_condition='both', prune_boolean=True))

    hist = track_population_generation_histogram(population)

    # Get list of axis objects
    ax, f = getSetup((16, 6), (1, 2))

    x_0 = [i for i in range(len(hist[0]))]
    x_1 = [i for i in range(len(hist[1]))]

    ax[0].set_xlim([-0.01, 13])
    ax[0].set_xlabel(r'Generation')
    ax[0].bar(x_0, hist[0], color='#F9Cb9C', label='Resistant')
    ax[0].bar(x_1, hist[1], bottom=hist[0], color='#A4C2F4', label='Susceptible')
    ax[0].set_ylabel('Number of alive Cells')
    ax[0].set_title('Pruned population growth')
    ax[0].grid(linestyle='--')

    y_0 = [a / (a + b) for a, b in zip(hist[0], hist[1])]
    y_1 = [b / (a + b) for a, b in zip(hist[0], hist[1])]

    ax[1].set_xlim([-0.01, 13])
    ax[1].set_xlabel(r'Generation')
    ax[1].plot(x_0, y_0, color='#F9Cb9C', label='Resistant')
    ax[1].plot(x_1, y_1, color='#A4C2F4', label='Susceptible')
    ax[1].set_ylabel('Proportion of alive Cells')
    ax[1].set_ylim([-0.01, 1.01])
    ax[1].set_title('Pruned population distribution')
    ax[1].grid(linestyle='--')
    ax[1].legend()

    f.tight_layout()

    return f
