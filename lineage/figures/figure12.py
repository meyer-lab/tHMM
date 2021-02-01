""" This file depicts the distribution of phase lengths versus the states for each concentration of gemcitabine. """
import pickle

from .figureCommon import getSetup, subplotLabel, plot_all
from ..plotTree import plot_networkx
from ..Analyze import Analyze_list
from ..data.Lineage_collections import Lapatinib_Control, Gemcitabine_Control, Gem5uM, Gem10uM, Gem30uM


concs = ["control", "gemcitabine 5 nM", "gemcitabine 10 nM", "gemcitabine 30 nM"]
concsValues = ["control", "5 nM", "10 nM", "30 nM"]
gemcitabine = [Lapatinib_Control + Gemcitabine_Control, Gem5uM, Gem10uM, Gem30uM]

gemc_tHMMobj_list, gemc_states_list, _ = Analyze_list(gemcitabine, 2, fpi=True)

for idx, gemc_tHMMobj in enumerate(gemc_tHMMobj_list):
    for lin_indx, lin in enumerate(gemc_tHMMobj.X):
        for cell_indx, cell in enumerate(lin.output_lineage):
            cell.state = gemc_states_list[idx][lin_indx][cell_indx]

# create a pickle file for gemcitabine
pik2 = open("gemcitabines.pkl", "wb")
for gemc in gemc_tHMMobj_list:
    pickle.dump(gemc, pik2)
pik2.close()

T_gem = gemc_tHMMobj_list[0].estimate.T
num_states = gemc_tHMMobj_list[0].num_states


def makeFigure():
    """ Makes figure 12. """

    ax, f = getSetup((22, 7.0), (2, 6))
    for u in range(4, 6):
        ax[u].axis("off")
    for u in range(10, 12):
        ax[u].axis("off")
    plot_all(ax, num_states, gemc_tHMMobj_list, "Gemcitabine", concs, concsValues)
    return f

plot_networkx(T_gem.shape[0], T_gem, 'gemcitabine')
