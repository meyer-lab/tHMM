import numpy as np
import scipy.stats as sp
import CellNode as c

def generate(numCells, locBern, cGom, locGom):
    #create first cell
    cell0 = c.CellNode(key=0, startT=20)
    
    # put first cell in list
    out = [cell0]
    
    # have cell divide/die according to distribution
    for j in len(out) and j < numCells:   # for all cells (cap at numCells)
        x = out[j]
        if x.isUnfinished():
            x.tau = sp.gompertz.rvs(cGom)
            x.endT = x.startT + x.tau
            x.fate = sp.bernoulli.rvs(locBern) # assign fate
            if x.fate == 1:
                temp1, temp2 = x.divide(x.endT) # cell divides
                # append to list
                out.append(temp1)
                out.append(temp2)
            else:
                x.die(x.endT)
                
    # return the list at end
    return out