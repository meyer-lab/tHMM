""" Figure 21 to perform cross validation on experimental data. """
from .common import getSetup
import pickle
import numpy as np
from ..Lineage_collections import Gemcitabine_Control, Gem5uM, Gem10uM, Gem30uM, Lapatinib_Control, Lapt25uM, Lapt50uM, Lap250uM
from .figure19 import output_LL

desired_num_states = np.arange(1, 8)

def makeFigure():
    """
    Makes figure 21.
    """
    ax, f = getSetup((9, 4), (1, 2))

    lapatinib = [Lapatinib_Control + Gemcitabine_Control, Lapt25uM, Lapt50uM, Lap250uM]
    gemcitabine = [Lapatinib_Control + Gemcitabine_Control, Gem5uM, Gem10uM, Gem30uM]

    lap_out = output_LL(lapatinib)
    gem_out = output_LL(gemcitabine)

    outs = [lap_out, gem_out]

    pik1 = open("sepT_bic.pkl", "wb")
    for bic in outs:
        pickle.dump(bic, pik1)
    pik1.close()

    # pik1 = open("sepT_bic.pkl", "rb")
    # pik2 = open("sharedT_bic.pkl", "rb")

    # seps, shareds = [], []
    # for bic in range(2):
    #     seps.append(pickle.load(pik1))
    #     shareds.append(pickle.load(pik2))

    ax[0].plot(desired_num_states, lap_out)
    ax[0].set_title("Lapatinib-treated")
    ax[1].plot(desired_num_states, gem_out)
    ax[1].set_title("Gemcitabine-treated")
    # ax[0].plot(desired_num_states, seps[0], label='equal transitions')
    # ax[0].plot(desired_num_states, shareds[0], label='estimate transitions')
    # ax[0].set_title("Lapatinib-treated")
    # ax[0].legend()
    # ax[1].plot(desired_num_states, seps[1], label='equal transitions')
    # ax[1].plot(desired_num_states, shareds[1], label='estimate transitions')
    # ax[1].legend()
    # ax[1].set_title("Gemcitabine-treated")

    for i in range(2):
        ax[i].set_ylabel("Log-likelihood")
        ax[i].set_xlabel("Number of States")

    return f
