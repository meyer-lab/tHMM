"""
File: figure1.py
Purpose: Generates figure 1.
Figure 1 is the tHMM model and its purpose.
"""
from string import ascii_lowercase

from .figureCommon import getSetup
from ..plotTree import plotLineage
from lineage.data.Lineage_collections import Gem5uM, gemControl as gem


def makeFigure():
    """
    Makes figure 1.
    """

    # Get list of axis objects
    ax, f = getSetup((5.1, 2.0), (4, 2))
    figureMaker(ax)
    ax[0].text(-0.2, 1.7, ascii_lowercase[0], transform=ax[0].transAxes, fontsize=16, fontweight="bold", va="top")
    ax[1].text(-0.2, 1.7, ascii_lowercase[1], transform=ax[1].transAxes, fontsize=16, fontweight="bold", va="top")

    return f


def figureMaker(ax):
    """
    Makes figure 1.
    """
    i = 0
    plotLineage(Gem5uM[13], ax[i], censore=False)
    ax[i].axis('off')

    i = 2
    plotLineage(Gem5uM[4], ax[i], censore=False)
    ax[i].axis('off')

    i = 4
    plotLineage(Gem5uM[10], ax[i], censore=False)
    ax[i].axis('off')

    for cell in gem[3].output_lineage:
        cell.state = 0
    for cell in gem[9].output_lineage:
        cell.state = 1
    gem[2].output_lineage[0].state = 1

    i = 1
    plotLineage(gem[3], ax[i], censore=False)
    ax[i].axis('off')

    i = 3
    plotLineage(gem[9], ax[i], censore=False)
    ax[i].axis('off')

    i = 5
    plotLineage(gem[2], ax[i], censore=False)
    ax[i].axis('off')

    i += 1
    ax[i].axis('off')
    i += 1
    ax[i].axis('off')
