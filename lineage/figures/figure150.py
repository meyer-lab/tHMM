""" This is a file to put together 4 conditions of gemcitabine together. """

from .figureCommon import getSetup, subplotLabel

def makeFigure():
    """
    Makes figure 150.
    """

    ax, f = getSetup((10, 7), (1, 4))
    subplotLabel(ax)
    for i in range(4):
        ax[i].axis('off')

    return f
