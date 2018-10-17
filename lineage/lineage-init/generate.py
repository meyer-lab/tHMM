import numpy as np
import scipy.stats as sp
import CellNode as c

def generate(numCells, locBern, cGom, locGom):
    #create first cell
    cell0 = c.CellNode(key=0, startT=20)
    
    # put first cell in list
    out = [cell0]
    
    # have cell divide/die according to distribution
    for j in range(0, len(out)):   # for all cells (cap at numCells)
        print("j: " + str(j))
        if j == numCells:
            print("breaking from numCells")
            break
        x = out[j]
        print(x.isUnfinished())
        if x.isUnfinished():
            print("unfinished")
            x.tau = sp.gompertz.rvs(cGom)
            x.endT = x.startT + x.tau
            x.fate = sp.bernoulli.rvs(locBern) # assign fate
            print("assigned distributions")
            if x.fate == 1:
                print("dividing")
                temp1, temp2 = x.divide(x.endT) # cell divides
                # append to list
                out.append(temp1)
                out.append(temp2)
            else:
                x.die(x.endT)
                print("dying")
                
    # return the list at end
    return out