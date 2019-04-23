'''Generates a lineage with breadth model ie cells all switch state at a single time point'''

from lineage.Lineage_utils import remove_singleton_lineages
from lineage.Lineage_utils import generatePopulationWithTime as gpt


def Breadth_Two_State_Lineage(experimentTime, initCells, locBern, cGom, scaleGom, bern2, cG2, scaleG2, switchT, verbose=False):
    '''Creates a lineage at which the entire lineage changes state after a time point'''

    LINEAGE = gpt(experimentTime, initCells, locBern, cGom, scaleGom, switchT, bern2, cG2, scaleG2)

    while not LINEAGE:  # determines if lineage is empty, so can regenerate a new one
        LINEAGE = gpt(experimentTime, initCells, locBern, cGom, scaleGom, switchT, bern2, cG2, scaleG2)

    X = remove_singleton_lineages(LINEAGE)
    if verbose:
        print(len(X))

    return X
