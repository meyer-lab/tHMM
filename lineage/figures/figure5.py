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
    scatter_kws_list,
    figureMaker,
    commonAnalyze
)
# from ..Analyze import run_Analyze_over, run_Results_over
from ..LineageTree import LineageTree


def makeFigure():
    """
    Makes fig 5.
    """

    # Get list of axis objects
    ax, f = getSetup((13, 6.66), (2, 4))
    x, paramEst, dictOut, paramTrues = accuracy()
    figureMaker(ax, x, paramEst, dictOut, paramTrues, dist_dist=True)

    subplotLabel(ax)

    return f


def accuracy():
    """
    Calculates accuracy and parameter estimation
    over an increasing number of lineages in a population for
    a uncensored two-state model.
    We increase the desired number of cells in a lineage by
    the experiment time.
    """


    # Creating a list of populations to analyze over
    num_lineages = np.linspace(min_num_lineages, max_num_lineages, num_data_points, dtype=int)
    experiment_times = np.linspace(1200, int(2.5 * 1000), num_data_points)

    list_of_populations = []
    for indx, num in enumerate(num_lineages):
        population = []
        for _ in range(num):
            tmp_lineage = LineageTree.init_from_parameters(pi, T, E2, desired_num_cells=min_desired_num_cells, censor_condition=3, desired_experiment_time=experiment_times[indx])
            if len(tmp_lineage.output_lineage) < 3:
                pass
            else:
                population.append(tmp_lineage)

        # Adding populations into a holder for analysing
        list_of_populations.append(population)

    return commonAnalyze(list_of_populations, 2)
