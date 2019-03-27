'''Generates a lineage with breadth model ie cells all switch state at a single time point'''

import numpy as np

from lineage.Lineage_utils import remove_NaNs
from lineage.Lineage_utils import generatePopulationWithTime as gpt
from lineage.CellNode import CellNode

def Breadth_Two_State_Lineage(T_MAS, MASinitCells, MASlocBern, MAScGom, MASscaleGom, T_2, initCells2, locBern2, cGom2, scaleGom2, switchT):
    initCells = [lineage_num]

    LINEAGE = gpt(experimentTime, initCells, locBern, cGom, scaleGom, switchT, bern2, cG2, scaleG2)
    while len(LINEAGE) == 0:
        LINEAGE = gpt(experimentTime, initCells, locBern, cGom, scaleGom, switchT, bern2, cG2, scaleG2)
        
    X = remove_NaNs(LINEAGE)
    print(len(X))
    
    return(X)