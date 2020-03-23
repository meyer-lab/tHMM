"""
File: figure2.py
Purpose: Generates figure 2.

Figure 2 is the distribution of cells in a state over generations (pruned) and over time.
"""
from matplotlib.ticker import MaxNLocator

from .figureCommon import getSetup, pi, T, E, subplotLabel
from ..LineageTree import LineageTree
from ..StateDistribution import track_population_generation_histogram, track_population_growth_histogram


def makeFigure():
    """
    Makes figure 2.
    """

    # Get list of axis objects
    ax, f = getSetup((7, 7), (2, 2))

    figure_maker(ax)

    subplotLabel(ax)

    return f


def figure_maker(ax):
    """
    Makes figure 2.
    """
    # creating a population
    population_censored = []
    for _ in range(20):
        population_censored.append(LineageTree(pi, T, E, desired_num_cells=(2 ** 12) - 1, censor_condition=3, desired_experiment_time=500))

    hist_gen_censored = track_population_generation_histogram(population_censored)
    delta_time = 0.1
    hist_tim_censored = track_population_growth_histogram(population_censored, delta_time)

    i = 2
    ax[i].set_xlabel(r"Time [$\mathrm{hours}$]")
    ax[i].bar([delta_time * i for i in range(len(hist_tim_censored[0]))], hist_tim_censored[0], color="#F9Cb9C", label="Resistant")
    ax[i].bar(
        [delta_time * i for i in range(len(hist_tim_censored[1]))],
        hist_tim_censored[1],
        bottom=hist_tim_censored[0],
        color="#A4C2F4",
        label="Susceptible",
    )
    ax[i].set_ylabel("Number of alive cells")
    ax[i].set_title("Uncensored population growth")
    ax[i].set_xlim([-0.01, 300])
    ax[i].grid(linestyle="--")

    y_0_p = [a / (a + b) if a + b > 0 else 0 for a, b in zip(hist_tim_censored[0], hist_tim_censored[1])]
    y_1_p = [b / (a + b) if a + b > 0 else 0 for a, b in zip(hist_tim_censored[0], hist_tim_censored[1])]

    i = 3
    ax[i].set_xlabel(r"Time [$\mathrm{hours}$]")
    ax[i].plot([delta_time * i for i in range(len(hist_tim_censored[0]))], y_0_p, color="#F9Cb9C", label="Resistant")
    ax[i].plot([delta_time * i for i in range(len(hist_tim_censored[1]))], y_1_p, color="#A4C2F4", label="Susceptible")
    ax[i].set_ylabel("Proportion of alive cells")
    ax[i].set_xlim([-0.01, 300])
    ax[i].set_ylim([-0.01, 1.01])
    ax[i].set_title("Uncensored population distribution")
    ax[i].grid(linestyle="--")
    ax[i].legend()

    x0_gen = [i + 1 for i in list(range(len(hist_gen_censored[0])))]
    x1_gen = [i + 1 for i in list(range(len(hist_gen_censored[1])))]

    i = 0
    ax[i].set_xlim([-0.01, 13])
    ax[i].xaxis.set_major_locator(MaxNLocator(integer=True))
    ax[i].set_xlabel(r"Generation")
    ax[i].bar(x0_gen, hist_gen_censored[0], color="#F9Cb9C", label="Resistant")
    ax[i].bar(x1_gen, hist_gen_censored[1], bottom=hist_gen_censored[0], color="#A4C2F4", label="Susceptible")
    ax[i].set_ylabel("Number of alive cells")
    ax[i].set_title("Censored population growth")
    ax[i].grid(linestyle="--")

    y_0_gen_p = [a / (a + b) for a, b in zip(hist_gen_censored[0], hist_gen_censored[1])]
    y_1_gen_p = [b / (a + b) for a, b in zip(hist_gen_censored[0], hist_gen_censored[1])]

    i = 1
    ax[i].set_xlim([-0.01, 13])
    ax[i].xaxis.set_major_locator(MaxNLocator(integer=True))
    ax[i].set_xlabel(r"Generation")
    ax[i].plot(x0_gen, y_0_gen_p, color="#F9Cb9C", label="Resistant")
    ax[i].plot(x1_gen, y_1_gen_p, color="#A4C2F4", label="Susceptible")
    ax[i].set_ylabel("Proportion of alive cells")
    ax[i].set_ylim([-0.01, 1.01])
    ax[i].set_title("Censored population distribution")
    ax[i].grid(linestyle="--")
    ax[i].legend()
