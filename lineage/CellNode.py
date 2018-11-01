# author : shakthi visagan (shak360), adam weiner (adamcweiner)
# description: a file to hold the cell class

import sys
import math
import scipy.stats as sp
import matplotlib.pyplot as plt

class CellNode:
    def __init__(self, gen=1, startT=0, endT=float('nan'), fate=True, left=None, right=None, parent=None, plotVal=0):
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
        return math.isnan(self.endT)   # returns true when cell is still alive

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


def generate(numCells, locBern, cGom, cScale):
    #TODO: maybe move this elsewhere? (it's not in a class or anything), maybe reconsider naming this as well to generateTree or generateLineage
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
            if cell.fate == 1 and len(lineage) < numCells-1:
                temp1, temp2 = cell.divide(cell.endT) # cell divides
                # append to list
                lineage.append(temp1)
                lineage.append(temp2)
            else:
                cell.die(cell.endT)
                
    # return the list at end
    return lineage



class Tree:
    def __init__(self):
        self.tree = list()
    
    def loadTree(self, csv_file):
        #TODO: write function to import a tree from external file
        pass
    
    def plotTree(self):
        '''plots a lineage tree based on list generated by a generate method or from the imported file'''
        lineage = self.tree # assign variable to tree list
        
        plt.figure(figsize=(48,24)) # set up figure for plotting, width and height in inches
        
        for cell in lineage:
            if cell.isRootParent():
                plt.plot(cell.startT, cell.plotVal, 'bo', markersize=10) # plot the root parent cell as a blue dot
            
            plt.plot([cell.startT,cell.endT],[cell.plotVal,cell.plotVal], 'k') # plot the cell lifetime
            
            if cell.fate:
                #TODO: replace the below if statement with the isUnfinished() method for clarity
                if not math.isnan(cell.endT): # check for nan when some cells don't get a chance to be assigned their fate, before the experiment ends
                    plt.plot([cell.endT,cell.endT],[cell.left.plotVal,cell.right.plotVal],'k') # plot a splitting line if the cell divides
            
            else:
                plt.plot(cell.endT, cell.plotVal, 'ro', markersize=10) # plot a red dot if the cell dies

        plt.show()
        #plt.savefig('foo.pdf')
            
def generatePopulation(parameters):
    #TODO: go over how to organize and make various generate() methods
    ''' generates list given a maximum number of lineage trees,'''
    pass
        

class Population:
    def __init__(self):
        self.population = list()
        
    def loadPopulation(self, csv_file):
        #TODO: write function to import a population from external file
        pass
    
    def plotPopulation(self):
        '''plots a population of lineages based on list of lineages'''
        #TODO
        pass
    
    def doublingTime(self):
        # can be moved elsewhere if this isn't the right place for this function
        '''For a given population, calculates the population-level growth rate (i.e. doubling time)'''
        #TODO
        pass
    
    def bernoulliParameterEstimator(self):
        '''Estimates the Bernoulli parameter for a given population using MLE'''
        population = self.population # assign population to a variable
        mle_param_holder = []
        for lineage in population:
            for cell in lineage.tree:
                if not cell.isUnfinished():
                    mle_param_holder.append(cell.fate*1)
    
