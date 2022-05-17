""" Figure 20 for correlations between cell observations in the lineages. """
import numpy as np
import scipy.stats as sp
import itertools as it
import pickle
from .common import getSetup
from ..states.StateDistributionGamma import StateDistribution as gamma_state
from ..LineageTree import LineageTree

num_gens = 5
num_trials = 400
pik1 = open("gemcitabines.pkl", "rb")
gemc = []
for i in range(4):
    gemc.append(pickle.load(pik1))

pik2 = open("lapatinibs.pkl", "rb")
lpt = []
for i in range(4):
    lpt.append(pickle.load(pik2))

T = lpt[0].estimate.T
pi = lpt[0].estimate.pi
E = lpt[0].estimate.E

T2 = lpt[0].estimate.T
pi2 = lpt[0].estimate.pi
E2 = lpt[2].estimate.E

T3 = gemc[0].estimate.T
pi3 = gemc[0].estimate.pi
E3 = gemc[2].estimate.E


def makeFigure():
    """ Make figure 20 heatmap of correlation within lineages between lifetimes. """

    ax, f = getSetup((6, 8), (2, 2))

    corr1 = np.zeros((3, 100))
    corr2 = np.zeros((3, 100))
    corr11 = np.zeros((3, 100))
    corr12 = np.zeros((3, 100))

    for i in range(100):
        corr1[:, i] = repeat_corr(lpt, 2)
        corr2[:, i] = repeat_corr(gemc, 2)
        corr11[:, i] = repeat_corr(lpt, 3)
        corr12[:, i] = repeat_corr(gemc, 3)

    ax[0].boxplot(corr1.T)
    ax[0].set_title("lapatinib, G1 duration")
    ax[1].boxplot(corr2.T)
    ax[1].set_title("Gemcitabine, G1 duration")
    ax[2].boxplot(corr11.T)
    ax[2].set_title("lapatinib, S-G2 duration")
    ax[3].boxplot(corr12.T)
    ax[3].set_title("Gemcitabine, S-G2 duration")

    for i in range(4):
        ax[i].set_ylabel("spearman correlation coefficient")
        ax[i].set_xticklabels(['daughter', 'grand-daughter', 'great-grand-daughter'], rotation=30)
        ax[i].set_ylim((0, 1))

    return f


def get_lifetime_gens(population, obs_ix):

    first_gens = []
    second_gens = []
    third_gens = []
    forth_gens = []
    fifth_gens = []
    for lineage in population:
        gens = sorted({cell.gen for cell in lineage.output_lineage})  # appending the generation of cells in the lineage
        for gen in gens:
            level = [cell.obs[obs_ix] for cell in lineage.output_lineage if (cell.gen == gen and cell.observed)]
            if gen == 1:
                first_gens.append(level)
            elif gen == 2:
                second_gens.append(level)
            elif gen == 3:
                third_gens.append(level)
            elif gen == 4:
                forth_gens.append(level)
            else:
                fifth_gens.append(level)

    return list(it.chain(*first_gens)), list(it.chain(*second_gens)), list(it.chain(*third_gens)), list(it.chain(*forth_gens)), list(it.chain(*fifth_gens))


def corr(all_gens, degree):
    """ To calculate the correlation between mother-daughter cells, it creates the second array with repeated values the same size as the first.
    degree determines whether it is between mother-daughter cells, or between grandmother-daughter cells, or higher.
    """
    array1 = []
    for ix1, gen in enumerate(all_gens[degree:]):
        array1.append(np.repeat(all_gens[ix1], 2 ** degree))

    arr2 = []
    for i in range(degree, len(all_gens)):
        arr2 += all_gens[i]

    arr1 = list(it.chain(*array1))
    assert(len(arr1) == len(arr2))

    return sp.spearmanr(arr1, arr2).correlation


def repeat_corr(drug, ix):
    populations = []
    for i in range(4):
        populations += [LineageTree.init_from_parameters(drug[i].estimate.pi, drug[i].estimate.T, drug[i].estimate.E, 2 ** num_gens - 1) for _ in range(num_trials)]

    all_gens = get_lifetime_gens(populations, ix)

    return corr(all_gens, 1), corr(all_gens, 2), corr(all_gens, 3)
