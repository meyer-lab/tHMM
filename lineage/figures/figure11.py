""" This file depicts the distribution of phase lengths versus the states for each concentration of lapatinib. """
import pickle

from .figureCommon import getSetup, subplotLabel, plot_all
from ..plotTree import plot_networkx

concs = ["control", "lapatinib 25 nM", "lapatinib 50 nM", "lapatinib 250 nM"]
concsValues = ["control", "25 nM", "50 nM", "250 nM"]
pik1 = open("lapatinibs.pkl", "rb")
lapt_tHMMobj_list = []
for i in range(4):
    lapt_tHMMobj_list.append(pickle.load(pik1))

#  # run analysis for the found number if states
# lapt_tHMMobj_list, lapt_states_list, _ = Analyze_list(lapatinib, 3, fpi=True)

# # assign the predicted states to each cell
# for idx, lapt_tHMMobj in enumerate(lapt_tHMMobj_list):
#     for lin_indx, lin in enumerate(lapt_tHMMobj.X):
#         for cell_indx, cell in enumerate(lin.output_lineage):
#             cell.state = lapt_states_list[idx][lin_indx][cell_indx]

# # create a pickle file for lapatinib
# pik1 = open("lapatinibs.pkl", "wb")
# for laps in lapt_tHMMobj_list:
#     pickle.dump(laps, pik1)
# pik1.close()

T_lap = lapt_tHMMobj_list[0].estimate.T
num_states = lapt_tHMMobj_list[0].num_states


def makeFigure():
    """ Makes figure 11. """

    ax, f = getSetup((22, 7.0), (2, 6))
    for u in range(4, 6):
        ax[u].axis("off")
    for u in range(10, 12):
        ax[u].axis("off")

    plot_all(ax, num_states, lapt_tHMMobj_list, "Laptinib", concs, concsValues)
    return f


plot_networkx(T_lap.shape[0], T_lap, 'lapatinib')
