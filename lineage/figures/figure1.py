"""
File: figure1.py
Purpose: Generates figure 1.
Figure 1 is the tHMM model and its purpose.
"""
from string import ascii_lowercase

from .figureCommon import getSetup
from ..plotTree import plotLineage
from lineage.data.Lineage_collections import gem5uM, Gemcitabine_Control as gem


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
    plotLineage(gem5uM[13], ax[1], censore=False, color=False)
    ax[1].set_title("Gemcitabine 5 nM", fontsize=10)
    ax[1].axis('off')

    plotLineage(gem5uM[4], ax[3], censore=False, color=False)
    ax[3].axis('off')

    plotLineage(gem5uM[10], ax[5], censore=False, color=False)
    ax[5].axis('off')

    for cell in gem[3].output_lineage:
        cell.state = 0
    for cell in gem[9].output_lineage:
        cell.state = 1
    gem[3].output_lineage[0].state = 1

    plotLineage(gem[3], ax[0], censore=False, color=False)
    ax[0].set_title("Control", fontsize=10)
    ax[0].axis('off')

    plotLineage(gem[16], ax[2], censore=False, color=False)
    ax[2].axis('off')

    plotLineage(gem[2], ax[4], censore=False, color=False)
    ax[4].axis('off')

    ax[7].axis('off')
    ax[6].axis('off')
