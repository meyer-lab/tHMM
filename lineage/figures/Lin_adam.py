'''Generates a lineage with breadth model ie cells all switch state at a single time point'''

import unittest
import numpy as np

from lineage.Lineage_utils import remove_NaNs
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