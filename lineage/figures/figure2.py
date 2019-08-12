"""
This creates Figure 2.
Should be the model figure.
"""
from .figureCommon import subplotLabel, getSetup


def makeFigure():
    """ makes figure 2 """
    
    # Get list of axis objects
    ax, f = getSetup((7, 6), (3, 4))

    subplotLabel(ax[0], 'A')

    f.tight_layout()

    return f
