""" This file depicts the distribution of phase lengths versus the states for each concentration of gemcitabine. """
import pickle
from string import ascii_lowercase
from ..plotTree import plot_networkx, plot_lineage_samples
from .common import getSetup, plot_all

concs = ["Control", "Gemcitabine 5 nM", "Gemcitabine 10 nM", "Gemcitabine 30 nM"]
concsValues = ["Control", "5 nM", "10 nM", "30 nM"]

pik1 = open("gemcitabines.pkl", "rb")
alls = []
for i in range(7):
    gemc_tHMMobj_list = []
    for i in range(4):
        gemc_tHMMobj_list.append(pickle.load(pik1))
    alls.append(gemc_tHMMobj_list)

# selected for lapatinib is 5 states which is index 4.
gemc_tHMMobj_list = alls[4]

gemc_states_list = [tHMMobj.predict() for tHMMobj in gemc_tHMMobj_list]

for idx, gemc_tHMMobj in enumerate(gemc_tHMMobj_list):
    for lin_indx, lin in enumerate(gemc_tHMMobj.X):
        for cell_indx, cell in enumerate(lin.output_lineage):
            cell.state = gemc_states_list[idx][lin_indx][cell_indx]

T_gem = gemc_tHMMobj_list[0].estimate.T
num_states = gemc_tHMMobj_list[0].num_states

# plot transition block
plot_networkx(T_gem.shape[0], T_gem, 'gemcitabine')

# plot sample of lineages
plot_lineage_samples(gemc_tHMMobj_list, 'figure02')


def makeFigure():
    """ Makes figure 12. """
    ax, f = getSetup((17, 7.5), (2, 7))
    plot_all(ax, num_states, gemc_tHMMobj_list, "Gemcitabine", concs, concsValues)
    for i in range(3, 7):
        ax[i].set_title(concs[i - 3], fontsize=16)
        ax[i].text(-0.2, 1.25, ascii_lowercase[i - 2], transform=ax[i].transAxes, fontsize=16, fontweight="bold", va="top")
        ax[i].axis('off')

    return f
