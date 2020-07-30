"""
File: figure2.py
Purpose: Generates figure 2.
Figure 2 is the tHMM model interface.
"""

from .figureCommon import getSetup

def makeFigure():
    """
    Makes figure 2.
    """
    ax, f = getSetup((10, 5), (1, 1))
    figureMaker(ax)

    return f

def figureMaker(ax):
    """
    Makes figure 2.
    """
    i = 0
    ax[i].axis('off')
