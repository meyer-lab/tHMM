""" Figure 21 to perform cross validation on experimental data. """
from .common import getSetup
import pickle
import pandas as pd
import numpy as np
from ..Lineage_collections import Gemcitabine_Control, Gem5uM, Gem10uM, Gem30uM, Lapatinib_Control, Lapt25uM, Lapt50uM, Lap250uM
from ..crossval import output_LL

desired_num_states = np.arange(1, 8)

def makeFigure():
    """
    Makes figure 21.
    """
    ax, f = getSetup((9, 4), (1, 2))

    lapatinib = [Lapatinib_Control + Gemcitabine_Control, Lapt25uM, Lapt50uM, Lap250uM]
    gemcitabine = [Lapatinib_Control + Gemcitabine_Control, Gem5uM, Gem10uM, Gem30uM]

    lap_out = output_LL(lapatinib, desired_num_states, 'lap')
    gem_out = output_LL(gemcitabine, desired_num_states, 'gem')

    # lap_eqT = pd.read_csv('lap_all_LLs.csv', index_col=[0])
    # lap_estT = pd.read_csv('lap_all_LLs_estimateT.csv', index_col=[0])
    # gem_eqT = pd.read_csv('gem_all_LLs.csv', index_col=[0])
    # gem_estT = pd.read_csv('gem_all_LLs_estimateT.csv', index_col=[0])

    ax[0].plot(desired_num_states, np.mean(np.asarray(lap_eqT), axis=0), label='equal T')
    ax[0].plot(desired_num_states, np.mean(np.asarray(lap_estT), axis=0), label='estimate T')
    ax[0].set_title("Lapatinib-treated")
    ax[1].plot(desired_num_states, np.mean(np.asarray(gem_eqT), axis=0), label='equal T')
    ax[1].plot(desired_num_states, np.mean(np.asarray(gem_estT), axis=0), label='estimate T')
    ax[1].set_title("Gemcitabine-treated")

    for i in range(2):
        ax[i].set_ylabel("Log-likelihood")
        ax[i].set_xlabel("Number of States")
        ax[i].legend()

    return f
