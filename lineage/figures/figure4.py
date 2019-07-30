"""
This creates Figure 4.
"""
from .figureCommon import getSetup
from .Matplot_gen import Matplot_gen
from .Fig_Gen import Lineage_Length


def makeFigure():
    # Get list of axis objects
    ax, f = getSetup((12, 3), (1, 3))

    f.tight_layout()

    return f
