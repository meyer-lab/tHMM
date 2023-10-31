"""
File: figureS03.py
Purpose: Generates figure S03.
Figure S03 analyzes heterogeneous (2 state), censored (by both time and fate),
populations of lineages (more than one lineage per populations).
"""
import numpy as np

from .common import (
    getSetup,
    subplotLabel,
    commonAnalyze,
    figureMaker,
    pi,
    T,
    E,
    min_desired_num_cells,
    min_experiment_time,
    min_num_lineages,
    max_num_lineages,
    num_data_points,
)
from ..LineageTree import LineageTree

rng = np.random.default_rng(1)

# Creating a list of populations to analyze over
num_lineages = np.linspace(
    min_num_lineages, max_num_lineages, num_data_points, dtype=int
)


def func():
    return LineageTree.rand_init(
        pi,
        T,
        E,
        min_desired_num_cells,
        censor_condition=3,
        desired_experiment_time=min_experiment_time,
        rng=rng,
    )


# Build population
populations = [[func() for _ in range(num)] for num in num_lineages]


def makeFigure():
    """
    Makes figure 5.
    """
    # Get list of axis objects
    ax, f = getSetup((10, 10), (3, 3))

    figureMaker(ax, *commonAnalyze(populations, 2), num_lineages=num_lineages)

    subplotLabel(ax)
    return f
