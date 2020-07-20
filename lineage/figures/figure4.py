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
from .figureS54 import figureMaker4, accuracy
from ..LineageTree import LineageTree


def makeFigure():
    """
    Makes fig 4.
    """

    # Get list of axis objects
    ax, f = getSetup((10, 10), (3, 3))
    number_of_columns = 25
    figureMaker4(ax, *accuracy(number_of_columns))

    subplotLabel(ax)

    return f
