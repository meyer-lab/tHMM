""" This file depicts the distribution of phase lengths versus the states for each concentration of gemcitabine. """
import numpy as np
import itertools
import seaborn as sns
from string import ascii_lowercase


from ..Analyze import Analyze_list
from ..tHMM import tHMM
from ..data.Lineage_collections import gemControl, gem5uM, Gem10uM, Gem30uM, Lapatinib_Control
from .figureCommon import getSetup, subplotLabel
from .figure11 import twice, plot_networkx, plot_all


data = [gemControl + Lapatinib_Control, gem5uM, Gem10uM, Gem30uM]
concs = ["control", "gemcitabine 5 nM", "gemcitabine 10 nM", "gemcitabine 30 nM"]
concsValues = ["control", "5 nM", "10 nM", "30 nM"]


num_states = 3
gemc_tHMMobj_list, gemc_states_list, _ = Analyze_list(data, num_states, fpi=True)
T_gem = gemc_tHMMobj_list[0].estimate.T


def makeFigure():
    """ Makes figure 12. """

    ax, f = getSetup((16, 6.0), (2, 5))
    plot_all(ax, num_states, gemc_tHMMobj_list, gemc_states_list, "Gemcitabine")
    return f

plot_networkx(T_gem.shape[0], T_gem, 'gemcitabine')
