"""
This creates Figure 6.
"""
from .figureCommon import getSetup
import numpy as np
from matplotlib import ticker
from .Fig_Gen import KL_per_lineage
from .Matplot_gen import moving_average


def makeFigure():
    """ Main figure generating function for Fig. 6 """
    ax, f = getSetup((7, 7), (1, 1))

    f.tight_layout()

    return f
