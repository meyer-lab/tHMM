"""
File: figure1.py
Purpose: Generates figure 1.
Figure 1 is the tHMM model and its purpose.
"""
from string import ascii_lowercase
import random
import numpy as np

from .figureCommon import getSetup
from ..plotTree import plotLineage
from lineage.data.Lineage_collections import Gem10uM, Gemcitabine_Control as control


def makeFigure():
    """
    Makes figure 1.
    """
    # Get list of axis objects
    ax, f = getSetup((7.6, 3.1), (8, 3))
    figureMaker(ax)
    ax[0].text(-0.2, 1.7, ascii_lowercase[0], transform=ax[0].transAxes, fontsize=16, fontweight="bold", va="top")
    ax[1].text(-0.2, 1.7, ascii_lowercase[1], transform=ax[1].transAxes, fontsize=16, fontweight="bold", va="top")
    ax[2].text(-0.2, 1.7, ascii_lowercase[2], transform=ax[2].transAxes, fontsize=16, fontweight="bold", va="top")

    return f


def figureMaker(ax):
    """
    Makes figure 1.
    """
    indxs_gem = [random.randint(0, (len(Gem10uM) - 1)) for _ in range(7)]
    indxs_c1 = [0, 2, 6, 8, 9, 10, 24]
    indxs_c2 = [1, 4, 5, 13, 17, 18, 19]
    # titles
    ax[0].set_title("Control 1", fontsize=10)
    ax[1].set_title("Control 2", fontsize=10)
    ax[2].set_title("Gemcitabine 10 nM - random", fontsize=10)
    # lineages
    for i in range(7):
        Gem10uM[indxs_gem[i]].state = np.nan
        plotLineage(control[indxs_c1[i]], ax[3 * i], censore=False, color=False)
        plotLineage(control[indxs_c2[i]], ax[3 * i + 1], censore=False, color=False)
        plotLineage(Gem10uM[indxs_gem[i]], ax[3 * i + 2], censore=False, color=False)
        
    for i in range(24):
        ax[i].axis('off')
    
