""" This file depicts the distribution of phase lengths versus the states. """

import numpy as np
import pandas as pd
import seaborn as sns
import itertools

from ..data.Lineage_collections import Gemcitabine_Control, Gem5uM, Gem10uM, Gem30uM, Lapatinib_Control, Lapt25uM, Lapt50uM, Lap250uM
from .figureCommon import getSetup, subplotLabel
from .figure11 import lpt12_g1, lpt12_g2, gmc12_g1, gmc12_g2


def makeFigure():
    """
    Makes figure 12.
    """
    ax, f = getSetup((13.2, 6.66), (2, 4))
    subplotLabel(ax)

    concs = ["cntrl", "Lapt 25uM", "Lapt 50uM", "Lapt 250uM", "cntrl", "Gem 5uM", "Gem 10uM", "Gem 30uM"]

    LAP = pd.DataFrame(columns=["state", "phase"])
    GEM = pd.DataFrame(columns=["state", "phase"])
    for i in range(4):
        # lapatinib
        LAP["state "+str(concs[i])] = [a[0] for a in lpt12_g1[i]] + [a[0] for a in lpt12_g2[i]]
        LAP["phase lengths "+str(concs[i])] = [a[1] for a in lpt12_g1[i]] + [a[1] for a in lpt12_g2[i]]
        LAP["phase "+str(concs[i])] = len(lpt12_g1[i]) * ["G1"] + len(lpt12_g2[i]) * ["G2"]
    
        sns.stripplot(x="state "+str(concs[i]), y="phase lengths "+str(concs[i]), hue="phase "+str(concs[i]), data=LAP, size=1, palette="Set2", linewidth=0.05, dodge=True, ax=ax[i])

        # gemcitabine
        GEM["state "+str(concs[i+4])] = [a[0] for a in gmc12_g1[i]] + [a[0] for a in gmc12_g2[i]]
        GEM["phase lengths "+str(concs[i+4])] = [a[1] for a in gmc12_g1[i]] + [a[1] for a in gmc12_g2[i]]
        GEM["phase "+str(concs[i+4])] = len(gmc12_g1[i]) * ["G1"] + len(gmc12_g2[i]) * ["G2"]
    
        sns.stripplot(x="state "+str(concs[i+4]), y="phase lengths "+str(concs[i+4]), hue="phase "+str(concs[i+4]), data=GEM, size=1, palette="Set2", linewidth=0.05, dodge=True, ax=ax[i+4])
        ax[i].set_title(concs[i])
        ax[i+4].set_title(concs[i+4])
        ax[i].set_ylabel("phase lengths")
        ax[i+4].set_ylabel("phase lengths")

        # this removes title of legends
        handles, labels = ax[i].get_legend_handles_labels()
        ax[i].legend(handles=handles[1:], labels=labels[1:])
        handles, labels = ax[i+4].get_legend_handles_labels()
        ax[i+4].legend(handles=handles[1:], labels=labels[1:])

    return f
