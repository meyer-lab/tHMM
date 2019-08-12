"""
This creates Figure 5.
"""
from .figureCommon import getSetup


def makeFigure():
    """ makes figure 5 """
    
    # Get list of axis objects
    ax, f = getSetup((12, 3), (1, 3))

    f.tight_layout()

    return f
