# author : shakthi visagan (shak360), adam weiner (adamcweiner)
# description: a file to hold the cell class

import sys
import math
import scipy.stats as sp
import matplotlib.pyplot as plt

class CellNode:
    def __init__(self, gen=1, startT=0, endT=float('nan'), fate=None, left=None, right=None, parent=None, plotVal=0):
        ''' Instantiates a cell node.'''
        self.gen = gen
        self.startT = startT
        self.endT = endT
        self.tau = self.endT - self.startT # avoiding self.t, since that is a common function (i.e. transposing matrices)
        self.fate = fate
        self.left = left
        self.right = right
        self.parent = parent
        self.plotVal = plotVal

    def isParent(self):
        return self.left.parent is self and self.right.parent is self

    def isChild(self):
        return self.parent.isParent()
    
    def isRootParent(self):
        return (self.gen == 1 and self.parent is None)

    def calcTau(self):
        self.tau = self.endT - self.startT   # calculate tau here
        if math.isnan(self.tau):
            print("Warning: your cell lifetime {} is a nan".format(self.tau))

    def isUnfinished(self):
        return math.isnan(self.endT) and self.fate is None   # returns true when cell is still alive

    def die(self, endT):
        """ Cell dies without dividing. """
        self.fate = False   # no division
        self.endT = endT    # mark endT
        self.calcTau()      # calculate Tau when cell dies

    def divide(self, endT):
        """ Cell life ends through division. """
        self.endT = endT
        self.fate = True    # division
        self.calcTau()      # calculate Tau when cell dies

        if self.isRootParent():
            self.left = CellNode(gen=self.gen+1, startT=endT, parent=self, plotVal=self.plotVal+50)
            self.right = CellNode(gen=self.gen+1, startT=endT, parent=self, plotVal=self.plotVal-50)
        else:
            self.left = CellNode(gen=self.gen+1, startT=endT, parent=self, plotVal=self.plotVal+(0.5**(self.gen))*(1.1**(self.gen))*self.plotVal)
            self.right = CellNode(gen=self.gen+1, startT=endT, parent=self, plotVal=self.plotVal-(0.5**(self.gen))*(1.1**(self.gen))*self.plotVal)

        return (self.left, self.right)


def generateLineage(numCells, locBern, cGom, cScale):
    #TODO: maybe move this elsewhere? (it's not in a class or anything), maybe reconsider naming this to generateTree or generateLineage
    ''' generates list given a maximum number of cells, a Bernoulli parameter for dividing/dying and a Gompertz parameter for cell lifetime'''
    #create first cell
    cell0 = CellNode(startT=0)
    
    # put first cell in list
    lineage = [cell0]
    
    # have cell divide/die according to distribution
    for cell in lineage:   # for all cells (cap at numCells)
        if len(lineage) >= numCells:
            break
        if cell.isUnfinished():
            cell.tau = sp.gompertz.rvs(cGom, scale=cScale)
            cell.endT = cell.startT + cell.tau
            cell.fate = sp.bernoulli.rvs(locBern) # assign fate
            if cell.fate:
                temp1, temp2 = cell.divide(cell.endT) # cell divides
                # append to list
                lineage.append(temp1)
                lineage.append(temp2)
            else:
                cell.die(cell.endT)
                
    # return the list at end
    return lineage