""" In this file we plot the abundance of states over time for experimental data. """
import numpy as np
import pickle
from .figureCommon import getSetup

pik1 = open("lapatinibs.pkl", "rb")
lapt_tHMMobj_list = []
for i in range(4):
    lapt_tHMMobj_list.append(pickle.load(pik1))

pik1 = open("gemcitabines.pkl", "rb")
gemc_tHMMobj_list = []
for i in range(4):
    gemc_tHMMobj_list.append(pickle.load(pik1))

times = np.linspace(0.0, 96.0, 48)
def find_state_proportions(lapt_tHMMobj):
    state0 = []
    state1 = []
    state2 = []
    for t in times:
        st0 = 0
        st1 = 0
        st2 = 0
        for lineage in lapt_tHMMobj.X:
            for cell in lineage.output_lineage:
                if cell.time.startT <= t <= cell.time.endT:
                    if cell.state == 0:
                        st0 += 1
                    elif cell.state == 1:
                        st1 += 1
                    else:
                        st2 += 1
        state0.append(st0)
        state1.append(st1)
        state2.append(st2)

    return state0, state1, state2

# control
CT_state0, CT_state1, CT_state2 = find_state_proportions(lapt_tHMMobj_list[0])

# 25 nM
one_state0, one_state1, one_state2 = find_state_proportions(lapt_tHMMobj_list[1])

# 50 nM
two_state0, two_state1, two_state2 = find_state_proportions(lapt_tHMMobj_list[2])

# 250 nM
three_state0, three_state1, three_state2 = find_state_proportions(lapt_tHMMobj_list[3])

# control
C_state0, C_state1, C_state2 = find_state_proportions(gemc_tHMMobj_list[0])

# 5 nM
oneG_state0, oneG_state1, oneG_state2 = find_state_proportions(gemc_tHMMobj_list[1])

# 10 nM
twoG_state0, twoG_state1, twoG_state2 = find_state_proportions(gemc_tHMMobj_list[2])

# 30 nM
threeG_state0, threeG_state1, threeG_state2 = find_state_proportions(gemc_tHMMobj_list[3])

def makeFigure():
    """ Makes figure S15. """

    ax, f = getSetup((8, 3.0), (2, 4))

    # lapatinibs
    ax[0].plot(times, CT_state0)
    ax[0].plot(times, CT_state1)
    ax[0].plot(times, CT_state2)

    ax[1].plot(times, one_state0)
    ax[1].plot(times, one_state1)
    ax[1].plot(times, one_state2)

    ax[2].plot(times, two_state0)
    ax[2].plot(times, two_state1)
    ax[2].plot(times, two_state2)

    ax[3].plot(times, three_state0)
    ax[3].plot(times, three_state1)
    ax[3].plot(times, three_state2)

    # gemcitabines
    ax[4].plot(times, C_state0)
    ax[4].plot(times, C_state1)
    ax[4].plot(times, C_state2)

    ax[5].plot(times, oneG_state0)
    ax[5].plot(times, oneG_state1)
    ax[5].plot(times, oneG_state2)

    ax[6].plot(times, twoG_state0)
    ax[6].plot(times, twoG_state1)
    ax[6].plot(times, twoG_state2)

    ax[7].plot(times, threeG_state0)
    ax[7].plot(times, threeG_state1)
    ax[7].plot(times, threeG_state2)
    return f