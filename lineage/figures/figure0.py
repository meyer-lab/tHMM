"""
File: figure0.py
Purpose: Generates figure 1.

Figure 0 is the distribution of cells in a state over generations (uncensored) and over time.
"""
import numpy as np

from .figureCommon import (
    getSetup,
    subplotLabel,
    pi,
    T,
    E,
    max_desired_num_cells,
    num_data_points,
)
from ..LineageTree import LineageTree

def makeFigure():
    """
    Makes figure 0.
    """

    # Get list of axis objects
    ax, f = getSetup((7, 7), (2, 2))
    
    figureMaker(ax)

    subplotLabel(ax)

    return f


def figureMaker(ax):
    """
    Creates the data for figure 0.
    """
    
   