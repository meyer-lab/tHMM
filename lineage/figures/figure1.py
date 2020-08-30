"""
File: figure1.py
Purpose: Generates figure 1.
Figure 1 is the tHMM model and its purpose.
"""
from string import ascii_lowercase

from .figureCommon import getSetup
from ..plotTree import plotLineage
from lineage.data.Lineage_collections import Gemcitabine_Control as gem


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
    index = [1, 3, 4, 6]
    i = 0
    plotLineage(gem[101], ax[i], censore=False)
    ax[i].axis('off')
    ax[i].set_title("Homogeneous")

    i = 2
    plotLineage(gem[83], ax[i], censore=False)
    ax[i].axis('off')

    i = 4
    plotLineage(gem[84], ax[i], censore=False)
    ax[i].axis('off')

    
    for j in index:
        gem[101].output_lineage[j].state = 0
        gem[83].output_lineage[j].state = 0
        gem[84].output_lineage[j].state = 0

    i = 1
    plotLineage(gem[101], ax[i], censore=False)
    ax[i].set_title("Heterogeneous")
    ax[i].axis('off')

    i = 3
    plotLineage(gem[83], ax[i], censore=False)
    ax[i].axis('off')

    i = 5
    plotLineage(gem[84], ax[i], censore=False)
    ax[i].axis('off')

    i += 1
    ax[i].axis('off')
    i += 1
    ax[i].axis('off')
