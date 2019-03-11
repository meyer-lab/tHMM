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

def Lin_adam(experimentTime, initCells, locBern, cGom, scaleGom, switchT, bern2, cG2, scaleG2):

    LINEAGE = gpt(experimentTime, initCells, locBern, cGom, scaleGom, switchT, bern2, cG2, scaleG2)
    while len(LINEAGE) == 0:
        LINEAGE = gpt(experimentTime, initCells, locBern, cGom, scaleGom, switchT, bern2, cG2, scaleG2)
        
    X = remove_NaNs(LINEAGE)
    
    master = []
    lin2 = []
    for indx in range(len(X)):
        cell = X[indx]
        if cell.true_state == 0:
            master.append(cell)
        elif cell.true_state == 1:
            lin2.append(cell)
    print('mas',len(master),'lin2',len(lin2))
    
    return(X,lin2)
