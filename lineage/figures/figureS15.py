""" In this file we plot the abundance of states over time for experimental data. """
import numpy as np
import math
import pickle
from .common import getSetup, subplotLabel

pik1 = open("lapatinibs.pkl", "rb")
lapt_tHMMobj_list = []
for i in range(4):
    lapt_tHMMobj_list.append(pickle.load(pik1))

pik1 = open("gemcitabines.pkl", "rb")
gemc_tHMMobj_list = []
for i in range(4):
    gemc_tHMMobj_list.append(pickle.load(pik1))

times = np.linspace(0.0, 96.0, 48)


def find_state_proportions(lapt_tHMMobj, control=False):
    states = np.zeros((len(times), 3))
    for indx, t in enumerate(times):
        st0 = 0
        st1 = 0
        st2 = 0
        if control:
            thmm = control
        else:
            thmm = lapt_tHMMobj.X

        for lineage in thmm:
            for cell in lineage.output_lineage:
                if math.isnan(cell.time.startT):  # left censored. startT = 0
                    cell.time.startT = 0.0
                if math.isnan(cell.time.endT):  # right censored. endT = 96
                    cell.time.endT = 96.0
                if cell.time.startT <= t <= cell.time.endT:
                    if cell.state == 0:
                        st0 += 1
                    elif cell.state == 1:
                        st1 += 1
                    else:
                        st2 += 1
        states[indx, 0] = 100.0 * st0 / (st0 + st1 + st2)
        states[indx, 1] = 100.0 * st1 / (st0 + st1 + st2)
        states[indx, 2] = 100.0 * st2 / (st0 + st1 + st2)

    return states


# labels
concs = ["control", "lapatinib 25 nM", "lapatinib 50 nM", "lapatinib 250 nM", "control", "gemcitabine 5 nM", "gemcitabine 10 nM", "gemcitabine 30 nM"]
# control
control_L = find_state_proportions(lapt_tHMMobj_list[0], control=lapt_tHMMobj_list[0].X[0:100])

# 25 nM
conc1_L = find_state_proportions(lapt_tHMMobj_list[1])

# 50 nM
conc2_L = find_state_proportions(lapt_tHMMobj_list[2])

# 250 nM
conc3_L = find_state_proportions(lapt_tHMMobj_list[3])

# control
control_G = find_state_proportions(gemc_tHMMobj_list[0], control=gemc_tHMMobj_list[0].X[101:])

# 5 nM
conc1_G = find_state_proportions(gemc_tHMMobj_list[1])

# 10 nM
conc2_G = find_state_proportions(gemc_tHMMobj_list[2])

# 30 nM
conc3_G = find_state_proportions(gemc_tHMMobj_list[3])


def makeFigure():
    """ Makes figure S15. """

    ax, f = getSetup((10, 5), (2, 4))

    ax[0].stackplot(times, control_L[:, 0], control_L[:, 1], control_L[:, 2], labels=['state 1', 'state 2', 'state 3'], alpha=0.6)
    ax[1].stackplot(times, conc1_L[:, 0], conc1_L[:, 1], conc1_L[:, 2], labels=['state 1', 'state 2', 'state 3'], alpha=0.6)
    ax[2].stackplot(times, conc2_L[:, 0], conc2_L[:, 1], conc2_L[:, 2], labels=['state 1', 'state 2', 'state 3'], alpha=0.6)
    ax[3].stackplot(times, conc3_L[:, 0], conc3_L[:, 1], conc3_L[:, 2], labels=['state 1', 'state 2', 'state 3'], alpha=0.6)

    ax[4].stackplot(times, control_G[:, 0], control_G[:, 1], control_G[:, 2], labels=['state 1', 'state 2', 'state 3'], alpha=0.6)
    ax[5].stackplot(times, conc1_G[:, 0], conc1_G[:, 1], conc1_G[:, 2], labels=['state 1', 'state 2', 'state 3'], alpha=0.6)
    ax[6].stackplot(times, conc2_G[:, 0], conc2_G[:, 1], conc2_G[:, 2], labels=['state 1', 'state 2', 'state 3'], alpha=0.6)
    ax[7].stackplot(times, conc3_G[:, 0], conc3_G[:, 1], conc3_G[:, 2], labels=['state 1', 'state 2', 'state 3'], alpha=0.6)

    for i in range(8):
        ax[i].legend()
        ax[i].set_title(concs[i])
        ax[i].set_xlabel("time [hr]")
        ax[i].set_ylabel("percentage")
        ax[i].set_ylim([0.0, 101.0])

    subplotLabel(ax)
    return f
