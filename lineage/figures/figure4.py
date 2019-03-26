"""
This creates Figure 4.
"""
from .figureCommon import subplotLabel, getSetup


def makeFigure():
    # Get list of axis objects
    ax, f = getSetup((7, 6), (3, 4))

    subplotLabel(ax[0], 'A')

    f.tight_layout()

    return f
