import numpy as np
import scipy.stats as sp
import CellNode as c
import math

def generate(numCells, locBern, cGom, locGom):
    #create first cell
    cell0 = c.CellNode(key=1, startT=0)
    
    # put first cell in list
    out = [cell0]
    
    # have cell divide/die according to distribution
    for cell in out:   # for all cells (cap at numCells)
        if len(out) >= numCells:
            break
        if cell.isUnfinished():
            cell.tau = sp.gompertz.rvs(cGom)
            cell.endT = cell.startT + cell.tau
            cell.fate = sp.bernoulli.rvs(locBern) # assign fate
            if cell.fate == 1:
                temp1, temp2 = cell.divide(cell.endT) # cell divides
                # append to list
                out.append(temp1)
                out.append(temp2)
            else:
                cell.die(cell.endT)
                
    # return the list at end
    return out