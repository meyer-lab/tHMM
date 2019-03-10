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

def Lin_shak(T_MAS, MASinitCells, MASlocBern, MAScGom, MASscaleGom, T_2, initCells2, locBern2, cGom2, scaleGom2):
    'Shakthis lineage where a second state is appended to first'
    MASexperimentTime = T_MAS
    masterLineage = gpt(MASexperimentTime, MASinitCells, MASlocBern, MAScGom, MASscaleGom)
    masterLineage = remove_NaNs(masterLineage)
    while len(masterLineage) == 0:
        masterLineage = gpt(MASexperimentTime, MASinitCells, MASlocBern, MAScGom, MASscaleGom)
        masterLineage = remove_NaNs(masterLineage)
    for cell in masterLineage:
        cell.true_state=0
    experimentTime2 = T_2
    sublineage2 = gpt(experimentTime2, initCells2, locBern2, cGom2, scaleGom2)
    sublineage2 = remove_NaNs(sublineage2)
    while len(sublineage2) == 0:
        sublineage2 = gpt(experimentTime2, initCells2, locBern2, cGom2, scaleGom2)
        sublineage2 = remove_NaNs(sublineage2)
    cell_endT_holder = []
    for cell in masterLineage:
        cell_endT_holder.append(cell.endT)

    master_cell_endT = max(cell_endT_holder) # get the longest tau in the list
    master_cell_endT_idx = np.argmax(cell_endT_holder) # get the idx of the longest tau in the lineage
    master_cell = masterLineage[master_cell_endT_idx] # get the master cell via the longest tau index
    for cell in sublineage2:
        cell.true_state = 1
        cell.linID = master_cell.linID
        cell.gen += master_cell.gen
        cell.startT += master_cell_endT
        cell.endT += master_cell_endT
    master_cell.left = sublineage2[0]
    sublineage2[0].parent = master_cell
    newLineage = masterLineage + sublineage2
    
    #X = newLineage    
    X = remove_NaNs(newLineage)
    print(len(newLineage))
    return(X, masterLineage, newLineage)