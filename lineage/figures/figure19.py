""" To plot a summary of cross validation. """
from .common import getSetup
import numpy as np
from ..LineageTree import LineageTree
from ..figures.common import pi, T, E
from ..CrossVal import hide_for_population, crossval
from ..Lineage_collections import Gemcitabine_Control, Gem5uM, Gem10uM, Gem30uM, Lapatinib_Control, Lapt25uM, Lapt50uM, Lap250uM

desired_num_states = np.arange(1, 8)

def makeFigure():
    """
    Makes figure 19.
    """
    ax, f = getSetup((8, 4), (1, 2))

    complete_population = [Lapatinib_Control + Gemcitabine_Control, Lapt25uM, Lapt50uM, Lap250uM]
    # create training data by hiding 20% of cells in each lineage
    train_population, hidden_indexes, hidden_obs = hide_for_population(complete_population)

    ll = []
    for i in desired_num_states:
        ll.append(crossval(train_population, hidden_indexes, hidden_obs, i))

    complete_population = [Gemcitabine_Control + Lapatinib_Control, Gem5uM, Gem10uM, Gem30uM]
    # create training data by hiding 20% of cells in each lineage
    train_population, hidden_indexes, hidden_obs = hide_for_population(complete_population)

    ll2 = []
    for i in desired_num_states:
        ll2.append(crossval(train_population, hidden_indexes, hidden_obs, i))

    ax[0].plot(desired_num_states, ll)
    ax[0].set_title("lapatinib")
    ax[1].plot(desired_num_states, ll2)
    ax[1].set_title("gemcitabine")
    return f
