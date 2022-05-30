""" To plot a summary of cross validation. """
from .common import getSetup
import numpy as np
from ..LineageTree import LineageTree
from ..figures.common import pi, T, E
from ..CrossVal import hide_observation, crossval
from ..Lineage_collections import Gemcitabine_Control, Gem5uM, Gem10uM, Gem30uM, Lapatinib_Control, Lapt25uM, Lapt50uM, Lap250uM

desired_num_states = np.arange(1, 8)

def makeFigure():
    """
    Makes figure 19.
    """
    ax, f = getSetup((4, 4), (1, 1))

    complete_lineages = Gemcitabine_Control + Lapatinib_Control

    # create training data by hiding 20% of cells in each lineage
    train_lineages, hidden_indexes, hidden_obs = [], [], []
    for complete_lin in complete_lineages:
        lineage, hide_index, hide_obs = hide_observation(complete_lin, 0.25)
        train_lineages.append(lineage)
        hidden_indexes.append(hide_index)
        hidden_obs.append(hide_obs)

    ll = []
    for i in desired_num_states:
        ll.append(crossval(train_lineages, hidden_indexes, hidden_obs, i))

    ax[0].plot(desired_num_states, ll)
    return f
