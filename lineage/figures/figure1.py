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
    ax, f = getSetup((7.6, 3.1), (11, 3))
    figureMaker(ax)
    ax[0].text(-0.2, 1.7, ascii_lowercase[0], transform=ax[0].transAxes, fontsize=16, fontweight="bold", va="top")
    ax[1].text(-0.2, 1.7, ascii_lowercase[1], transform=ax[1].transAxes, fontsize=16, fontweight="bold", va="top")
    ax[2].text(-0.2, 1.7, ascii_lowercase[2], transform=ax[2].transAxes, fontsize=16, fontweight="bold", va="top")

    return f


def figureMaker(ax):
    """
    Makes figure 1.
    """
    indxs_control = [random.randint(0, (len(control) - 1)) for _ in range(10)]
    indxs_gem = [random.randint(0, (len(Gem10uM) - 1)) for _ in range(10)]
    # titles
    ax[0].set_title("Control", fontsize=10)
    ax[1].set_title("Gemcitabine 10 nM", fontsize=10)
    ax[1].set_title("Control - random", fontsize=10)
    # lineages
    for i in range(10):
        control[indxs_control[i]].state = np.nan
        Gem10uM[indxs_gem[i]].state = np.nan
        plotLineage(control[2 * i], ax[3 * i], censore=False, color=False)
        plotLineage(Gem10uM[2 * i], ax[3 * i + 1], censore=False, color=False)
        plotLineage(control[indxs_control[i]], ax[3 * i + 2], censore=False, color=False)

    for i in range(33):
        ax[i].axis('off')
