"""Figure 21 to perform cross validation on experimental data."""

import numpy as np

from ..crossval import output_LL
from ..Lineage_collections import AllGemcitabine, AllLapatinib
from .common import getSetup

desired_num_states = np.arange(1, 8)


def makeFigure():
    """
    Makes figure 21.
    """
    ax, f = getSetup((9, 4), (1, 2))

    lap_out = output_LL(AllLapatinib, desired_num_states)
    gem_out = output_LL(AllGemcitabine, desired_num_states)

    ax[0].plot(desired_num_states, lap_out, label="estimate T")
    ax[0].set_title("Lapatinib-treated")
    ax[1].plot(desired_num_states, gem_out, label="estimate T")
    ax[1].set_title("Gemcitabine-treated")

    for i in range(2):
        ax[i].set_ylabel("Log-likelihood")
        ax[i].set_xlabel("Number of States")
        ax[i].legend()

    return f
