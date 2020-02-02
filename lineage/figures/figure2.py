"""
File: figure2.py
Purpose: Generates figure 2.

Figure 2 is the distribution of cells in a state over generations (pruned) and over time.
"""
import numpy as np
from matplotlib.ticker import MaxNLocator

from .figureCommon import getSetup
from ..LineageTree import LineageTree
from ..StateDistribution import StateDistribution, track_population_generation_histogram, track_population_growth_histogram


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
    gamma_scale0 = 5

    # State 1 parameters "Susceptible"
    state1 = 1
    bern_p1 = 0.88
    gamma_a1 = 10
    gamma_scale1 = 1

    state_obj0 = StateDistribution(state0, bern_p0, gamma_a0, gamma_scale0)
    state_obj1 = StateDistribution(state1, bern_p1, gamma_a1, gamma_scale1)
    E = [state_obj0, state_obj1]

    # creating a population
    population_pruned = []
    desired_num_cells = (2**12) - 1
    desired_experiment_time = 300
    for _ in range(20):
        population_pruned.append(
            LineageTree(
                pi,
                T,
                E,
                desired_num_cells,
                desired_experiment_time=desired_experiment_time,
                prune_condition='both',
                prune_boolean=True))

    hist_gen_pruned = track_population_generation_histogram(population_pruned)
    delta_time = 0.1
    hist_tim_pruned = track_population_growth_histogram(population_pruned, delta_time)

    # Get list of axis objects
    ax, f = getSetup((16, 16), (2, 2))

    # generations

    x0_gen = [i + 1 for i in list(range(len(hist_gen_pruned[0])))]
    x1_gen = [i + 1 for i in list(range(len(hist_gen_pruned[1])))]

    i = 0
    ax[i].set_xlim([-0.01, 13])
    ax[i].xaxis.set_major_locator(MaxNLocator(integer=True))
    ax[i].set_xlabel(r'Generation')
    ax[i].bar(x0_gen, hist_gen_pruned[0], color='#F9Cb9C', label='Resistant')
    ax[i].bar(x1_gen, hist_gen_pruned[1], bottom=hist_gen_pruned[0], color='#A4C2F4', label='Susceptible')
    ax[i].set_ylabel('Number of alive cells')
    ax[i].set_title('Pruned population growth')
    ax[i].grid(linestyle='--')

    y_0_gen_p = [a / (a + b) for a, b in zip(hist_gen_pruned[0], hist_gen_pruned[1])]
    y_1_gen_p = [b / (a + b) for a, b in zip(hist_gen_pruned[0], hist_gen_pruned[1])]

    i += 1
    ax[i].set_xlim([-0.01, 13])
    ax[i].xaxis.set_major_locator(MaxNLocator(integer=True))
    ax[i].set_xlabel(r'Generation')
    ax[i].plot(x0_gen, y_0_gen_p, color='#F9Cb9C', label='Resistant')
    ax[i].plot(x1_gen, y_1_gen_p, color='#A4C2F4', label='Susceptible')
    ax[i].set_ylabel('Proportion of alive cells')
    ax[i].set_ylim([-0.01, 1.01])
    ax[i].set_title('Pruned population distribution')
    ax[i].grid(linestyle='--')
    ax[i].legend()

    # time

    i += 1
    ax[i].set_xlabel(r'Time [$\mathrm{hours}$]')
    ax[i].bar([delta_time * i for i in range(len(hist_tim_pruned[0]))], hist_tim_pruned[0], color='#F9Cb9C', label='Resistant')
    ax[i].bar([delta_time * i for i in range(len(hist_tim_pruned[1]))], hist_tim_pruned[1],
              bottom=hist_tim_pruned[0], color='#A4C2F4', label='Susceptible')
    ax[i].set_ylabel('Number of alive cells')
    ax[i].set_title('Pruned population growth')
    ax[i].set_xlim([-0.01, 300])
    ax[i].grid(linestyle='--')

    y_0_p = [a / (a + b) if a + b > 0 else 0 for a, b in zip(hist_tim_pruned[0], hist_tim_pruned[1])]
    y_1_p = [b / (a + b) if a + b > 0 else 0 for a, b in zip(hist_tim_pruned[0], hist_tim_pruned[1])]

    i += 1
    ax[i].set_xlabel(r'Time [$\mathrm{hours}$]')
    ax[i].plot([delta_time * i for i in range(len(hist_tim_pruned[0]))], y_0_p, color='#F9Cb9C', label='Resistant')
    ax[i].plot([delta_time * i for i in range(len(hist_tim_pruned[1]))], y_1_p, color='#A4C2F4', label='Susceptible')
    ax[i].set_ylabel('Proportion of alive cells')
    ax[i].set_xlim([-0.01, 300])
    ax[i].set_ylim([-0.01, 1.01])
    ax[i].set_title('Pruned population distribution')
    ax[i].grid(linestyle='--')
    ax[i].legend()

    return f
