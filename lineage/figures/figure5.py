"""
This creates Figure 5.
"""
from .figureCommon import subplotLabel, getSetup


def makeFigure():
    """ makes figure 5 """
    # Get list of axis objects
    ax, f = getSetup((7, 6), (3, 4))

    subplotLabel(ax[0], 'A')

    return f
