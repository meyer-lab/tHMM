"""
This creates Figure 5.
"""
from .figureCommon import getSetup
from .Matplot_gen import Matplot_gen
from .Fig_Gen import Lineages_per_Population_Figure


def makeFigure():
    # Get list of axis objects
    ax, f = getSetup((12, 3), (1, 3))

    f.tight_layout()

    return f
