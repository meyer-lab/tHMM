# author : shakthi visagan (shak360), adam weiner (adamcweiner)
# description: a file to hold the cell class

import sys
import math
import numpy as np
import scipy.stats as sp

class CellNode:
    def __init__(self, key, startT=0, endT=float('nan'), fate=True, left=None, right=None, parent=None):
        ''' Instantiates a cell node. Only requires a key '''
        self.key = key
        self.startT = startT
        self.endT = endT
        self.tau = self.endT - self.startT # avoiding self.t, since that is a common function (i.e. transposing matrices)
        self.fate = fate
        self.left = left
        self.right = right
        self.parent = parent
    
    def hasLeft(self):
        return self.left

    def hasRight(self):
        return self.right

    def isLeft(self):
        return self.parent and self.parent.left == self

    def isRight(self):
        return self.parent and self.parent.right == self

    def isParent(self):
        return not self.parent

    def isChild(self):
        return not (self.left or self.right)

    def hasAnyChildren(self):
        return self.right or self.left

    def hasBothChildren(self):
        return self.right and self.left

    def calcTau(self):
        self.tau = self.endT - self.startT   # calculate tau here
        if self.tau == float('nan'):
            print("Warning: your cell lifetime {} is a non-positive number".format(self.tau))
    
    def isUnfinished(self):
        return math.isnan(self.endT)   # returns true when cell is still alive
    
    def die(self, endT):
        """ Cell dies without dividing. """
        self.fate = False   # no division
        self.endT = endT    # mark endT
    
    def divide(self, endT):
        """ Cell life ends through division. """
        self.endT = endT
        self.fate = True   # division

        # two daughter cells emerge at this time... not sure about 1st and 3rd arguments bc I don't know what "key" is and don't know what python's statement for "this" is.
        # self is Python's way of denoting this, self can actually be any word, but common practice is to denote it as self
        # key is a binary number, basically if a parent's key is 1, then it's two daughters will have value 10 and 11
        # if a parent has key 11001, then it's two daughter's will have values 110010 and 110011
        
        self.left = CellNode(key=(self.key<<1), startT=endT, parent=self)
        self.right = CellNode(key=((self.key<<1)+1), startT=endT, parent=self)
        
        return (self.left, self.right)

    









def generate(numCells, locBern, cGom):
    #create first cell
    cell0 = CellNode(key=1, startT=0)
    
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