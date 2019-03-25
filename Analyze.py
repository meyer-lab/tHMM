import unittest
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties


from lineage.BaumWelch import fit
from lineage.DownwardRecursion import get_root_gammas, get_nonroot_gammas
from lineage.Viterbi import get_leaf_deltas, get_nonleaf_deltas, Viterbi
from lineage.UpwardRecursion import get_leaf_Normalizing_Factors, get_leaf_betas, get_nonleaf_NF_and_betas, calculate_log_likelihood
from lineage.tHMM import tHMM
from lineage.tHMM_utils import max_gen, get_gen, get_parents_for_level
from lineage.Lineage_utils import remove_NaNs, get_numLineages, init_Population
from lineage.Lineage_utils import generatePopulationWithTime as gpt
from lineage.CellNode import CellNode

def Analyze(X, numStates, keepBern=True):
    run = True
    while run == True:
        tHMMobj = tHMM(X, numStates=numStates, keepBern=True) # build the tHMM class with X
        fit(tHMMobj, max_iter=200, verbose=True)
        if tHMMobj.paramlist[0]["E"][0,1] < 1000 and tHMMobj.paramlist[0]["E"][1,1] < 1000:
            run = False
    deltas, state_ptrs = get_leaf_deltas(tHMMobj) # gets the deltas matrix
    get_nonleaf_deltas(tHMMobj, deltas, state_ptrs)
    all_states = Viterbi(tHMMobj, deltas, state_ptrs)
    NF = get_leaf_Normalizing_Factors(tHMMobj)
    betas = get_leaf_betas(tHMMobj, NF)
    get_nonleaf_NF_and_betas(tHMMobj, NF, betas)
    LL = calculate_log_likelihood(tHMMobj, NF)
    lineage = tHMMobj.population[lin]
    T = tHMMobj.paramlist[lin]["T"]
    E = tHMMobj.paramlist[lin]["E"]
    pi = tHMMobj.paramlist[lin]["pi"] 
    return(deltas, state_ptrs, all_states, tHMMobj, NF, LL)