'''Generates a lineage with depth model ie one new cell state lineage is added to another state homogeneous lineage'''

import numpy as np
import logging
from .Lineage_utils import remove_singleton_lineages, select_population
from .Lineage_utils import generatePopulationWithTime as gpt


def Depth_Two_State_Lineage(T_MAS, MASinitCells, MASlocBern, T_2, initCells2, locBern2, FOM='E', betaExp=None, betaExp2=None):
    '''X is the complete lineage with removed singletons. newLineage is X wihtout the removed singletons. Master Lineage and sublineage2 are the lineages corresponding to state1 and state2'''
    'Shakthis lineage where a second state is appended to first'
    # Making the first lineage
    MASexperimentTime = T_MAS
    masterLineage = gpt(MASexperimentTime, MASinitCells, MASlocBern, FOM=FOM, betaExp=betaExp)
    masterLineage = remove_singleton_lineages(masterLineage)
    masterLineage, _ = select_population(masterLineage, MASexperimentTime)

    while not masterLineage:
        masterLineage = gpt(MASexperimentTime, MASinitCells, MASlocBern, FOM=FOM, betaExp=betaExp)
        masterLineage = remove_singleton_lineages(masterLineage)
        masterLineage, _ = select_population(masterLineage, MASexperimentTime)
    
    for cell in masterLineage:
        cell.true_state = 0
    # Making the second lineage
    experimentTime2 = T_2
    sublineage2 = gpt(experimentTime2, initCells2, locBern2, FOM=FOM, betaExp=betaExp2)
    sublineage2 = remove_singleton_lineages(sublineage2)
    sublineage2, _ = select_population(sublineage2, experimentTime2)

    while not sublineage2:
        sublineage2 = gpt(experimentTime2, initCells2, locBern2, FOM=FOM, betaExp=betaExp2)
        sublineage2 = remove_singleton_lineages(sublineage2)
        sublineage2, _ = select_population(sublineage2, experimentTime2)
    
    cell_endT_holder = []
    for cell in masterLineage:
        cell_endT_holder.append(cell.endT)

    master_cell_endT = max(cell_endT_holder)  # get the longest tau in the list
    master_cell_endT_idx = np.argmax(cell_endT_holder)  # get the idx of the longest tau in the lineage
    master_cell = masterLineage[master_cell_endT_idx]  # get the master cell via the longest tau index

    for cell in sublineage2:
        cell.true_state = 1
        cell.linID = master_cell.linID
        cell.gen += master_cell.gen
        cell.startT += master_cell_endT
        cell.endT += master_cell_endT
    master_cell.left = sublineage2[0]
    sublineage2[0].parent = master_cell
    newLineage = masterLineage + sublineage2

    logging.info('length of entire lineage:', len(newLineage), 'length master:', len(masterLineage), 'length sublineage:', len(sublineage2))
    return(X, newLineage, masterLineage, sublineage2)
