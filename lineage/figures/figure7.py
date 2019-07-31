"""
This creates Figure 7. AIC Figure.
"""
from .figureCommon import getSetup


def makeFigure():
    # Get list of axis objects
    ax, f = getSetup((5, 5), (1, 1))

    f.tight_layout()

    return f
