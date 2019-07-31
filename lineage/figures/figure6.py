"""
This creates Figure 6.
"""
from .figureCommon import getSetup


def makeFigure():
    """ Main figure generating function for Fig. 6 """
    ax, f = getSetup((7, 7), (1, 1))

    f.tight_layout()

    return f
