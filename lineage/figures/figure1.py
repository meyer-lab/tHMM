"""
File: figure1.py
Purpose: Generates figure 1.

Figure 1 is the distribution of cells in a state over generations (censored) and over time.
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
    Makes figure 2.
    """

    # Get list of axis objects
    ax, f = getSetup((7, 7), (2, 2))
    
    figureMaker(ax)

    subplotLabel(ax)

    return f
   