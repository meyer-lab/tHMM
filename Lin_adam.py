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

def Lin_adam():
    initCells = [lineage_num]

    LINEAGE = gpt(experimentTime, initCells, locBern, cGom, scaleGom, switchT, bern2, cG2, scaleG2)
    while len(LINEAGE) == 0:
        LINEAGE = gpt(experimentTime, initCells, locBern, cGom, scaleGom, switchT, bern2, cG2, scaleG2)
        
    X = remove_NaNs(LINEAGE)
    print(len(X))
    
    return(X)