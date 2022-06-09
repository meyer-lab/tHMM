""" To plot a summary of cross validation. """
from .common import getSetup
import numpy as np
from ..crossval import hide_observation, crossval
from ..Lineage_collections import Gemcitabine_Control, Gem5uM, Gem10uM, Gem30uM, Lapatinib_Control, Lapt25uM, Lapt50uM, Lap250uM

desired_num_states = np.arange(1, 8)

def makeFigure():
    """
    Makes figure 19.
    """
    ax, f = getSetup((8, 4), (1, 2))

    lapatinib = [Lapatinib_Control + Gemcitabine_Control, Lapt25uM, Lapt50uM, Lap250uM]
    gemcitabine = [Gemcitabine_Control + Lapatinib_Control, Gem5uM, Gem10uM, Gem30uM]

    output1 = output_LL(lapatinib)
    output2 = output_LL(gemcitabine)


    ax[0].plot(desired_num_states, output1)
    ax[0].set_title("lapatinib")
    ax[1].plot(desired_num_states, output2)
    ax[1].set_title("gemcitabine")
    return f

def output_LL(complete_population):
    # create training data by hiding 20% of cells in each lineage
    train_population = [hide_observation(complete_pop, 0.2) for complete_pop in complete_population]
    # Copy out data to full set
    dataFull = []
    for _ in desired_num_states:
        dataFull.append(train_population)

    return crossval(dataFull, desired_num_states)
