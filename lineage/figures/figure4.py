"""
This creates Figure 4.
"""
from .figureCommon import getSetup


def makeFigure():
    """ makes figure 4 """

    # Get list of axis objects
    _, f = getSetup((12, 3), (1, 3))

    return f
