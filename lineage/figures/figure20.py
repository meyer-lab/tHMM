""" Figure 20 for correlations between cell observations in the lineages. """
import numpy as np
import pandas as pd
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

T2 = lpt[2].estimate.T
pi2 = lpt[2].estimate.pi
E2 = lpt[2].estimate.E

T3 = gemc[2].estimate.T
pi3 = gemc[2].estimate.pi
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
    cens_1 =[]
    second_gens = []
    cens_2 = []
    third_gens = []
    cens_3 = []
    forth_gens = []
    cens_4 = []
    fifth_gens = []
    cens_5 = []
    for lineage in population:
        gens = sorted({cell.gen for cell in lineage.output_lineage})  # appending the generation of cells in the lineage
        for gen in gens:
            level = [cell.obs[obs_ix] for cell in lineage.output_lineage if (cell.gen == gen and cell.observed)]
            cens = [cell.obs[obs_ix+2] for cell in lineage.output_lineage if (cell.gen == gen and cell.observed)]
            if gen == 1:
                first_gens.append(level)
                cens_1.append(cens)
            elif gen == 2:
                second_gens.append(level)
                cens_2.append(cens)
            elif gen == 3:
                third_gens.append(level)
                cens_3.append(cens)
            elif gen == 4:
                forth_gens.append(level)
                cens_4.append(cens)
            else:
                fifth_gens.append(level)
                cens_5.append(cens)

    return [list(it.chain(*first_gens)), list(it.chain(*second_gens)), list(it.chain(*third_gens)), list(it.chain(*forth_gens)), list(it.chain(*fifth_gens))], [list(it.chain(*cens_1)), list(it.chain(*cens_2)), list(it.chain(*cens_3)), list(it.chain(*cens_4)), list(it.chain(*cens_5))]


def corr(all_gens, all_cens, degree):
    """ To calculate the correlation between mother-daughter cells, it creates the second array with repeated values the same size as the first.
    degree determines whether it is between mother-daughter cells, or between grandmother-daughter cells, or higher.
    """
    array1 = []
    cens1 = []
    for ix1, gen in enumerate(all_gens[degree:]):
        array1.append(np.repeat(all_gens[ix1], 2 ** degree))
        cens1.append(np.repeat(all_cens[ix1], 2 ** degree))

    arr2 = []
    cens2 = []
    for i in range(degree, len(all_gens)):
        arr2 += all_gens[i]
        cens2 += all_cens[i]

    arr1 = list(it.chain(*array1[1:]))
    c1 = list(it.chain(*cens1[1:]))
    assert(len(arr1) == len(arr2))

    df = pd.DataFrame({"gen1" : arr1, "cen1": c1, "gen2" : arr2, "cen2": cens2})

    return sp.spearmanr(arr1, arr2).correlation, df


def repeat_corr(drug, ix):
    """ Given which ``drug`` (lapatinib or gemcitabine here), and ``ix`` 
    which is the index of observations: 2 for G1 lifetime, 3 for S-G2 lifetime, calculates the correlations 
    between gen 1 & 2, gen 1 & 3, and gen 1 & 4.
    """
    populations = []
    for i in range(4):
        populations += [LineageTree.init_from_parameters(drug[i].estimate.pi, drug[i].estimate.T, drug[i].estimate.E, 2 ** num_gens - 1) for _ in range(num_trials)]

    all_gens, cen = get_lifetime_gens(populations, ix)

    return corr(all_gens[0], cen[0], 1)[0], corr(all_gens[0], cen[0], 2)[0], corr(all_gens[0], cen[0], 3)[0]

def save_df():
    """ Save the arrays that are used for calculating the correlation into dataframes, in the form of column1:gen1, column2: gen2. 
    This functions does this for gen1 & 2, gen 1 & 3, gen 1 & 4, and for both G1 and S-G2 cell lifetimes. """
    pik1 = open("lapatinibs.pkl", "rb")
    lapt_tHMMobj_list = []
    for i in range(4):
        lapt_tHMMobj_list.append(pickle.load(pik1))

    populations = []
    for tHMMobj in lapt_tHMMobj_list:
        populations += tHMMobj.X

    # G1
    all_gens1, all_cens1 = get_lifetime_gens(populations, 2)
    df1 = corr(all_gens1, all_cens1, 1)[1]
    df2 = corr(all_gens1[0], all_cens1[0], 2)[1]
    df3 = corr(all_gens1[0], all_cens1[0], 3)[1]
    df1.to_csv(r'df_g1_gen12.csv', index=False)
    df2.to_csv(r'df_g1_gen13.csv', index=False)
    df3.to_csv(r'df_g1_gen14.csv', index=False)

    # S-G2
    all_gens2, all_cens2 = get_lifetime_gens(populations, 3)
    df11 = corr(all_gens2[0], all_cens2[0], 1)[1]
    df12 = corr(all_gens2[0], all_cens2[0], 2)[1]
    df13 = corr(all_gens2[0], all_cens2[0], 3)[1]
    df11.to_csv(r'df_g2_gen12.csv', index=False)
    df12.to_csv(r'df_g2_gen13.csv', index=False)
    df13.to_csv(r'df_g2_gen14.csv', index=False)