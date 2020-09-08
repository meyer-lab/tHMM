""" This file plots the trees with their predicted states. """

import numpy as np
from matplotlib.ticker import MaxNLocator
import itertools

from ..Analyze import Analyze
from ..data.Lineage_collections import Gemcitabine_Control, Gem5uM, Gem10uM, Gem30uM, Lapatinib_Control, Lapt25uM, Lapt50uM, Lap250uM
from .figureCommon import getSetup, subplotLabel
from ..tHMM import tHMM
from ..plotTree import plotLineage


def makeFigure():
    """
    Makes figure 10.
    """
    ax, f = getSetup((7, 12), (8, 2))
    Gem = [Gemcitabine_Control, Gem5uM, Gem10uM, Gem30uM]
    Lap = [Lapatinib_Control, Lapt25uM, Lapt50uM, Lap250uM]

    lapatinib = []
    gemcitabine = []
    # Run fitting
    for indx, data in enumerate(Gem):
        gemc_tHMMobj = Analyze(data, 2)[0] # 4 states predicted by AIC
        gemcitabine.append([gemc_tHMMobj.X[3], gemc_tHMMobj.X[6]])
        lapt_tHMMobj = Analyze(Lap[indx], 2)[0] # 3 states predicted by AIC
        lapatinib.append([lapt_tHMMobj.X[4], lapt_tHMMobj.X[7]])

    # Plotting the lineages
    figure_maker(ax, list(itertools.chain(*lapatinib)), list(itertools.chain(*gemcitabine)))

    return f


def figure_maker(ax, lapatinib, gemcitabine):
    """
    Makes figure 10.
    """
    ax[0].set_title("Lapatinib")
    ax[1].set_title("Gemcitabine")

    i = 0
    ax[i].axis('off')
    plotLineage(lapatinib[0], ax[i], censore=True)
    i += 1
    ax[i].axis('off')
    plotLineage(gemcitabine[0], ax[i], censore=True)
    i += 1
    ax[i].axis('off')
    plotLineage(lapatinib[1], ax[i], censore=True)
    i += 1
    ax[i].axis('off')
    plotLineage(gemcitabine[1], ax[i], censore=True)
    i += 1
    ax[i].axis('off')
    plotLineage(lapatinib[2], ax[i], censore=True)
    i += 1
    ax[i].axis('off')
    plotLineage(gemcitabine[2], ax[i], censore=True)
    i += 1
    ax[i].axis('off')
    plotLineage(lapatinib[3], ax[i], censore=True)
    i += 1
    ax[i].axis('off')
    plotLineage(gemcitabine[3], ax[i], censore=True)
    i += 1
    ax[i].axis('off')
    plotLineage(lapatinib[4], ax[i], censore=True)
    i += 1
    ax[i].axis('off')
    plotLineage(gemcitabine[4], ax[i], censore=True)
    i += 1
    ax[i].axis('off')
    plotLineage(lapatinib[5], ax[i], censore=True)
    i += 1
    ax[i].axis('off')
    plotLineage(gemcitabine[5], ax[i], censore=True)
    i += 1
    ax[i].axis('off')
    plotLineage(lapatinib[6], ax[i], censore=True)
    i += 1
    ax[i].axis('off')
    plotLineage(gemcitabine[6], ax[i], censore=True)
    i += 1
    ax[i].axis('off')
    plotLineage(lapatinib[7], ax[i], censore=True)
    i += 1
    ax[i].axis('off')
    plotLineage(gemcitabine[7], ax[i], censore=True)
