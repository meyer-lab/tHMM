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
            
            
    state_1 = np.argmax(pi)
    state_2 = np.argmin(pi) 
    
    wrong = 0  
    
    trues = []
    for cell in range(len(lineage)):
        trues.append(lineage[cell].true_state)
        if lineage[cell].true_state == state_1:
            if all_states[lin][cell] == state_1:
                pass
            else:
                wrong += 1
        elif lineage[cell].true_state ==state_2:
            if all_states[lin][cell] == state_2:
                pass
            else:
                wrong += 1           
                        
    accuracy = (len(lineage) - wrong)/len(lineage) #must be fixed for more than 1 lineage    
    
    ''' for cell in range(len(lineage)):
        if cell < len(masterLineage):
            if all_states[lin][cell] == state_1:
                pass
            else:
                wrong += 1
        elif cell >= len(masterLineage) and cell < len(newLineage):
            if all_states[lin][cell] == state_2:
                pass
            else:
                wrong += 1           
    accuracy = (len(lineage) - wrong)/len(lineage) #must be fixed for more than 1 lineage'''
    
    return(T,E,pi,state_1,state_2,accuracy,lineage)