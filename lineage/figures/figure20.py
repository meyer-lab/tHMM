""" Figure 20 for correlations between cell observations in the lineages. """
import numpy as np
import scipy.stats as sp
# from ..Lineage_collections import Gemcitabine_Control, Gem5uM, Gem10uM, Gem30uM, Lapatinib_Control, Lapt25uM, Lapt50uM, Lap250uM
from .common import getSetup, pi, T, E
from ..states.StateDistributionGamma import StateDistribution as gamma_state
from ..LineageTree import LineageTree

def makeFigure():
    """ Make figure 20 heatmap of correlation within lineages between lifetimes. """

    ax, f = getSetup((4, 4), (1, 1))

    num_gens = 6
    num_trials = 400

    lineages = [LineageTree.init_from_parameters(pi, T, E, 2 ** num_gens - 1) for _ in range(num_trials)]
    corr1 = np.zeros((num_gens-1, num_trials))
    corr2 = np.zeros((num_gens-2, num_trials))
    corr3 = np.zeros((num_gens-3, num_trials))
    all_gens = get_lifetime_gens(lineages)
    for ix1, lins in enumerate(all_gens):
        for ix2, gens in enumerate(lins[1:]):
            corr1[ix2, ix1] = expand_and_corr(lins[ix2], gens, 2)
        for ix3, genns in enumerate(lins[2:]):
            corr2[ix3, ix1] = expand_and_corr(lins[ix3], genns, 4)
        for ix4, gns in enumerate(lins[3:]):
            corr3[ix4, ix1] = expand_and_corr(lins[ix4], gns, 8)

    all_corr = np.zeros((3, num_trials))
    all_corr[0, :] = np.nanmean(corr1, axis=0)
    all_corr[1, :] = np.nanmean(corr2, axis=0)
    all_corr[2, :] = np.nanmean(corr3, axis=0)

    ax[0].boxplot(all_corr.T)
    ax[0].set_xlabel("generation number")
    ax[0].set_ylabel("spearman correlation coefficient")
    ax[0].set_title("correlations")
    ax[0].set_xticklabels(['daughter', 'grand-daughter', 'great-grand-daughter'], rotation=30)

    return f

def get_lifetime_gens(population):
    all_gens = []
    for lineage in population:
        gens = sorted({cell.gen for cell in lineage.output_lineage})  # appending the generation of cells in the lineage
        cells_by_gen = []
        for gen in gens:
            level = [cell.obs[1] for cell in lineage.output_lineage if (cell.gen == gen and cell.observed)]
            cells_by_gen.append(level)
        all_gens.append(cells_by_gen)

    return all_gens

def expand_and_corr(gen1, gen2, rep):
    """ takes two successive generations and expands the first one to fit the number of second, and calculates the spearman correlation. """
    gen1_expanded = np.repeat(gen1, rep)
    assert(len(gen1_expanded) == len(gen2))

    return sp.spearmanr(gen1_expanded, gen2).correlation
