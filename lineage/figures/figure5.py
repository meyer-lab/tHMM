"""
This creates Figure 5.
"""
from .figureCommon import getSetup


def makeFigure():
    # Get list of axis objects
    ax, f = getSetup((12, 3), (1, 3))

    f.tight_layout()

    return f
