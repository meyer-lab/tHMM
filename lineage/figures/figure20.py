""" Figure 20 for correlations between cell observations in the lineages. """
import numpy as np
import pandas as pd
import scipy.stats as sp
import itertools as it
import pickle
from .common import getSetup
from ..Lineage_collections import Gemcitabine_Control, Gem5uM, Gem10uM, Gem30uM, Lapatinib_Control, Lapt25uM, Lapt50uM, Lap250uM

def get_lifetime_gens(populations, obs_ix):

    # remove signleton lineages
    population = []
    for lineage in populations:
        if len(lineage.output_lineage) > 1:
            population.append(lineage)
    
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

    return [list(it.chain(*first_gens)), list(it.chain(*second_gens)), list(it.chain(*third_gens)), list(it.chain(*forth_gens))], [list(it.chain(*cens_1)), list(it.chain(*cens_2)), list(it.chain(*cens_3)), list(it.chain(*cens_4))]


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

    arr1 = list(it.chain(*array1))
    c1 = list(it.chain(*cens1))
    print(len(arr1), len(arr2))
    assert(len(arr1) == len(arr2))

    return pd.DataFrame({"gen1" : arr1, "cen1": c1, "gen2" : arr2, "cen2": cens2})


def save_df():
    """ Save the arrays that are used for calculating the correlation into dataframes, in the form of column1:gen1, column2: gen2. 
    This functions does this for gen1 & 2, gen 1 & 3, gen 1 & 4, and for both G1 and S-G2 cell lifetimes. """
    
    lapatinib = [Lapatinib_Control[0:8]]
    gemcitabine = [Lapatinib_Control + Gemcitabine_Control, Gem5uM, Gem10uM, Gem30uM]

    # G1
    all_gens, all_cens = get_lifetime_gens(list(it.chain(*lapatinib)), 2)
    # remove empty generations
    all_gens1 = [x for x in all_gens if x]
    all_cens1 = [x for x in all_cens if x]

    df1 = corr(all_gens1, all_cens1, 1)[1]
    df2 = corr(all_gens1, all_cens1, 2)[1]
    df3 = corr(all_gens1, all_cens1, 3)[1]
    df1.to_csv(r'df_g1_gen12.csv', index=False)
    df2.to_csv(r'df_g1_gen13.csv', index=False)
    df3.to_csv(r'df_g1_gen14.csv', index=False)
