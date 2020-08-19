"""
File: figure1.py
Purpose: Generates figure 1.
Figure 1 is the tHMM model and its purpose.
"""
from string import ascii_lowercase
import numpy as np
from random import randint

from .figureCommon import getSetup
from ..plotTree import plotLineage
from lineage.data.Lineage_collections import Gemcitabine_Control as gem


def makeFigure():
    """
    Makes figure 1.
    """

    plotLineage(gem[4], 'lineage/figures/cartoons/figure1a.svg', censore=False)
    plotLineage(gem[13], 'lineage/figures/cartoons/figure1b.svg', censore=False)
    plotLineage(gem[14], 'lineage/figures/cartoons/figure1c.svg', censore=False)

    index = [1, 3, 4, 7]
    for i in index:
        gem[4].output_lineage[i].state = 0
        gem[13].output_lineage[i].state = 0
        gem[14].output_lineage[i].state = 0

    plotLineage(gem[4], 'lineage/figures/cartoons/figure1d.svg', censore=False)
    plotLineage(gem[13], 'lineage/figures/cartoons/figure1e.svg', censore=False)
    plotLineage(gem[14], 'lineage/figures/cartoons/figure1f.svg', censore=False)

    # Get list of axis objects
    ax, f = getSetup((4.5, 5.5), (3, 2))
    figureMaker(ax)
    ax[0].text(-0.2, 1.22, ascii_lowercase[0], transform=ax[0].transAxes, fontsize=16, fontweight="bold", va="top")
    ax[1].text(-0.2, 1.22, ascii_lowercase[1], transform=ax[1].transAxes, fontsize=16, fontweight="bold", va="top")

    return f


def figureMaker(ax):
    """
    Makes figure 1.
    """
    i = 0
    ax[i].axis('off')
    i += 1
    ax[i].axis('off')
    i += 1
    ax[i].axis('off')
    i += 1
    ax[i].axis('off')
    i += 1
    ax[i].axis('off')
    i += 1
    ax[i].axis('off')
