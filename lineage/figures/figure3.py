""" This file contains figures related to how big the experment needs to be. """
import numpy as np
import pandas as pd
import seaborn as sns
from .figureCommon import (
    getSetup,
    subplotLabel,
    commonAnalyze,
    pi,
    E2,
    T,
    min_desired_num_cells,
    min_num_lineages,
    max_num_lineages,
    lineage_good_to_analyze,
    num_data_points
)
from .figureS53 import figureMaker3, accuracy
from ..LineageTree import LineageTree


def makeFigure():
    """
    Makes fig 3A.
    """

    # Get list of axis objects
    ax, f = getSetup((10, 10), (3, 3))
    lin_params = {"pi": pi, "T": T, "E": E2, "desired_num_cells": min_desired_num_cells, "censor_condition": 3, "desired_experiment_time": 500}
    number_of_columns = 5
    figureMaker3(ax, *accuracy(lin_params, number_of_columns))

    subplotLabel(ax)

    return f
