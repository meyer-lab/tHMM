""" To plot a summary of cross validation. """
from .common import getSetup
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from ..LineageTree import LineageTree
from ..figures.common import pi, T, E
from ..CrossVal import hide_for_population, crossval
from ..Lineage_collections import Gemcitabine_Control, Gem5uM, Gem10uM, Gem30uM, Lapatinib_Control, Lapt25uM, Lapt50uM, Lap250uM

desired_num_states = np.arange(1, 5)

exe = ProcessPoolExecutor()

def makeFigure():
    """
    Makes figure 19.
    """
    ax, f = getSetup((8, 4), (1, 2))

    complete_population = [[Lapatinib_Control + Gemcitabine_Control, Lapt25uM, Lapt50uM, Lap250uM], [Gemcitabine_Control + Lapatinib_Control, Gem5uM, Gem10uM, Gem30uM]]

    prom_holder = []
    for idx, population in enumerate(complete_population):
        prom_holder.append(exe.submit(parallel_LL, population))

    output = [prom.result() for prom in prom_holder]

    ax[0].plot(desired_num_states, output[0])
    ax[0].set_title("lapatinib")
    ax[1].plot(desired_num_states, output[1])
    ax[1].set_title("gemcitabine")
    return f

def parallel_LL(complete_population):
    # create training data by hiding 20% of cells in each lineage
    train_population, hidden_indexes, hidden_obs = hide_for_population(complete_population, 0.25)

    ll = []
    for i in desired_num_states:
        ll.append(crossval(train_population, hidden_indexes, hidden_obs, i))
    return ll