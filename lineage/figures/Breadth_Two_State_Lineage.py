'''Generates a lineage with breadth model ie cells all switch state at a single time point'''

from lineage.Lineage_utils import remove_singleton_lineages
from lineage.Lineage_utils import generatePopulationWithTime as gpt


def Breadth_Two_State_Lineage(experimentTime, initCells, locBern, betaExp, switchT, bern2, betaExp2, FOM, verbose=False):
    '''Creates a lineage at which the entire lineage changes state after a time point'''

    LINEAGE = gpt(experimentTime, initCells, locBern, betaExp, switchT, bern2, betaExp2, FOM)

    LINEAGE = remove_unfinished_cells(LINEAGE)
    LINEAGE = remove_singleton_lineages(LINEAGE)
        
    while not LINEAGE:  # determines if lineage is empty, so can regenerate a new one
        LINEAGE = gpt(experimentTime, initCells, locBern, betaExp, switchT, bern2, betaExp2, FOM)
        LINEAGE = remove_unfinished_cells(LINEAGE)
        LINEAGE = remove_singleton_lineages(LINEAGE)



    X = LINEAGE
        
        
    masterLineage = []
    sublineage2 = []
    for ii, cell in enumerate(X):
        if cell.true_state == 0:
            masterLineage.append(cell)
        elif cell.true_state == 1:
            sublineage2.append(cell)
        else:
            raise print('more than 2 true states error')
            

    if verbose:
        print(len(masterLineage), len(sublineage2), len(X))
        
    return X, masterLineage, sublineage2
