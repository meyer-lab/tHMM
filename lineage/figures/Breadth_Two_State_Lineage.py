'''Generates a lineage with breadth model ie cells all switch state at a single time point'''

from lineage.Lineage_utils import remove_singleton_lineages, remove_unfinished_cells
from lineage.Lineage_utils import generatePopulationWithTime as gpt
import logging


def Breadth_Two_State_Lineage(experimentTime, initCells, locBern, betaExp, switchT, bern2, betaExp2, FOM, verbose=False):
    '''Creates a lineage at which the entire lineage changes state after a time point'''

    # Generate lineage and remove the cell unfinished and the cells that dont create a lineage
    LINEAGE = gpt(experimentTime, initCells, locBern, betaExp, switchT, bern2, betaExp2, FOM)
    LINEAGE = remove_unfinished_cells(LINEAGE)
    LINEAGE = remove_singleton_lineages(LINEAGE)
    while not LINEAGE:  # determines if lineage is empty, so can regenerate a new one
        LINEAGE = gpt(experimentTime, initCells, locBern, betaExp, switchT, bern2, betaExp2, FOM)
        LINEAGE = remove_unfinished_cells(LINEAGE)
        LINEAGE = remove_singleton_lineages(LINEAGE)
    X = LINEAGE
    
    # Put cells in their respective true lineages based on their true states
    masterLineage = []
    sublineage2 = []
    for ii, cell in enumerate(X):
        if cell.true_state == 0:
            masterLineage.append(cell)
        elif cell.true_state == 1:
            sublineage2.append(cell)
        else:
            logging.info('more than 2 true states error.')
    newLineage = masterLineage + sublineage2
    print('X size: {}, masterLineage size: {}, subLineage2 size: {}'.format(len(X), len(masterLineage), len(sublineage2)))
    return X, newLineage, masterLineage, sublineage2
