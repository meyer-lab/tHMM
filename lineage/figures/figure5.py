""" This file contains figures related to how big the experment needs to be. """
import numpy as np
import pandas as pd
import seaborn as sns
from .figureCommon import (
    getSetup,
    subplotLabel,
    pi,
    E2,
    T,
    min_desired_num_cells,
    min_num_lineages,
    max_num_lineages,
    num_data_points,
    figureMaker,
    commonAnalyze
)
from ..LineageTree import LineageTree
from ..states.StateDistributionGamma import StateDistribution as gamma_state

pi2 = np.array([1])
T2 = np.array([[1]])
E3 = [gamma_state(bern_p=1., gamma_a=7, gamma_scale=4.5)]

# Creating a list of populations to analyze over
num_lineages = np.linspace(min_num_lineages, max_num_lineages, num_data_points, dtype=int)
experiment_times = np.linspace(1200, int(2.5 * 1000), num_data_points)

list_of_populations = []
for indx, num in enumerate(num_lineages):
    population = []
    for _ in range(num):
        tmp_lineage = LineageTree.init_from_parameters(pi2, T2, E3, desired_num_cells=min_desired_num_cells, censor_condition=3, desired_experiment_time=experiment_times[indx])
        if len(tmp_lineage.output_lineage) < 3:
            pass
        else:
            population.append(tmp_lineage)

    # Adding populations into a holder for analysing
    list_of_populations.append(population)


def makeFigure():
    """
    Makes fig 5.
    """

    # Get list of axis objects
    ax, f = getSetup((11, 8), (3, 3))
    # dist_dist is distribution distance
    figureMaker(ax, *commonAnalyze(list_of_populations, 1), num_lineages=num_lineages, dist_dist=True)

    subplotLabel(ax)

    return f
