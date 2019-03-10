import unittest
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties

from lineage.BaumWelch import fit
from lineage.DownwardRecursion import get_root_gammas, get_nonroot_gammas
from lineage.Viterbi import get_leaf_deltas, get_nonleaf_deltas, Viterbi
from lineage.UpwardRecursion import get_leaf_Normalizing_Factors, get_leaf_betas, get_nonleaf_NF_and_betas
from lineage.tHMM import tHMM
from lineage.tHMM_utils import max_gen, get_gen, get_parents_for_level
from lineage.Lineage_utils import remove_NaNs, get_numLineages, init_Population
from lineage.Lineage_utils import generatePopulationWithTime as gpt
from lineage.CellNode import CellNode

def Accuracy(tHMMobj, lin, numStates, masterLineage, newLineage, all_states):
    lineage = tHMMobj.population[lin]
    T = tHMMobj.paramlist[lin]["T"]
    E = tHMMobj.paramlist[lin]["E"]
    pi = tHMMobj.paramlist[lin]["pi"] 
    #assign state 1 and state 2
    T_non_diag = np.zeros(numStates)
    for state_j in range(numStates):
        for state_k in range(numStates):
            if state_j != state_k:
                T_non_diag[state_j] = T[state_j,state_k]
    if E[0,0] > E[1,0]:
        state_1 = 0
        state_0 = 1
    elif E[1,0] > E[0,0]:
        state_1 = 1
        state_0 = 0
        
    wrong = 0  
    
    trues = []
    viterbi_est = []
    for cell in range(len(lineage)):
        cell_state = lineage[cell].true_state
        viterbi_state = all_states[lin][cell]
        trues.append(cell_state)
        viterbi_est.append(viterbi_state)
        if cell_state == 0:
            if viterbi_state == state_0:
                pass
            else:
                wrong += 1
        elif cell_state == 1:
            if viterbi_state == state_1:
                pass
            else:
                wrong += 1           
                        
    accuracy = (len(lineage) - wrong)/len(lineage) #must be fixed for more than 1 lineage    
    print('trues', trues)
    print('viterbi',viterbi_est)
    return(T,E,pi,state_0,state_1,accuracy,lineage)