""" Figure 21 to perform cross validation on experimental data. """
from .common import getSetup
import pandas as pd
import numpy as np
from ..Lineage_collections import AllLapatinib, AllGemcitabine
from ..crossval import output_LL

desired_num_states = np.arange(1, 8)


def makeFigure():
    """
    Makes figure 21.
    """
    ax, f = getSetup((9, 4), (1, 2))

    lap_out = output_LL(AllLapatinib, desired_num_states)
    gem_out = output_LL(AllGemcitabine, desired_num_states)

    # ax[0].plot(desired_num_states, np.mean(np.asarray(lap_eqT), axis=0), label='equal T')
    ax[0].plot(desired_num_states, np.mean(np.asarray(lap_estT), axis=0), label='estimate T')
    ax[0].set_title("Lapatinib-treated")
    # ax[1].plot(desired_num_states, np.mean(np.asarray(gem_eqT), axis=0), label='equal T')
    ax[1].plot(desired_num_states, np.mean(np.asarray(gem_estT), axis=0), label='estimate T')
    ax[1].set_title("Gemcitabine-treated")

    for i in range(2):
        ax[i].set_ylabel("Log-likelihood")
        ax[i].set_xlabel("Number of States")
        ax[i].legend()

    return f
