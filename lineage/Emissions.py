""" This file contains the class for CellVar which holds the state and observation information in the hidden and observed trees respectively. """
import scipy.stats as sp
import numpy as np

# Boolean functions are in camel-case.
# Functions that return cells or lists of cells will be spaced with underscores.

class CellVar:
    def __init__(self, state, left, right, parent, gen):
        """ Instantiates the cell object. Contains memeber variables that identify daughter cells and parent cells. Also contains the state of the cell. """
        self.state = state
        self.left = left
        self.right = right
        self.parent = parent
        self.gen = gen
        
    def _divide(self, T):
        """ Member function that performs division of a cell. Equivalent to adding another timestep in a Markov process. """
        left_state, right_state = double(self.state, T) # roll a loaded die according to the row in the transtion matrix
        self.left = CellVar(state=left_state, left=None, right=None, parent=self, gen=self.gen+1) # assign the resulting states to new cells
        self.right = CellVar(state=right_state, left=None, right=None, parent=self, gen=self.gen+1) # ensure that those cells are related
        
        return self.left, self.right
    
    def _isParent(self):
        """ Boolean. Returns true if the cell has daughters. """
        return self.left or self.right
    
    def _isChild(self):
        """ Boolean. Returns true if this cell has a known parent. """
        return self.parent.isParent()
    
    def _isRootParent(self):
        """ Boolean. Returns true if this cell is the first cell in a lineage. """
        bool_parent = False
        if not self.parent and self.gen == 1:
            bool_parent = True
        return bool_parent
    
    def _isLeaf(self):
        """ Boolean. Returns true when a cell is a leaf with no children. """
        return self.left
    
    def _get_sister(self):
        """ Member function that gets the sister of the current cell. """
        cell_to_return = None
        if self.parent.left is self:
            cell_to_return = self.parent.right
        elif self.parent.right is self:
            cell_to_return = self.parent.left
        return cell_to_return
        
    def _get_root_cell(self):
        """ Get the first cell in the lineage to which this cell belongs. """
        curr_cell = self
        while curr_cell.gen > 1:
            curr_cell = curr_cell.parent
        assert _isRootParent(curr_cell)
        return curr_cell
        
def _double(parent_state, T):
    """ Function that essentially rolls two of the same loaded dice given a state that determines the row of the transition matrix. The results of the roll of the loaded dice are two new states that are returned. """
    # Checking that the inputs are of the right shape
    assert T.shape[0] == T.shape[1], "Transition numpy array is not square. Ensure that your transition numpy array has the same number of rows and columns."
    num_states = T.shape[0]
    assert 0 <= parent_state <= num_states-1, "The parent state is a state outside of the range states being considered."
    
    # Rolling two of the same loaded dice separate times and assigning where they landed to states
    left_state_results, right_state_results = sp.multinomial.rvs(n=1, p=T[parent_state,:], size=2) # first and second roll are left and right
    left_state = left_state_results.index(1) # the result of the dice toss (where it landed) is the state
    right_state = right_state_results.index(1)

    return left_state, right_state

def generate(T, pi, num_cells):
    """ Generates a single lineage tree given Markov variables. This only generates the hidden variables (i.e., the states). """
    first_state_results = sp.multinomial.rvs(1, pi) # roll the dice and yield the state for the first cell
    first_cell_state = first_state_results.index(1) 
    first_cell = CellVar(state=first_cell_state, left=None, right=None, parent=None, gen=1) # create first cell
    lineage_list = [first_cell]
    
    for cell in lineage_list: # letting the first cell proliferate
        if not cell.left: # if the cell has no daughters...
            left_cell, right_cell = cell._divide(T) # make daughters by dividing and assigning states
            lineage_list.append(left_cell) # add daughters to the list of cells
            lineage_list.append(right_cell)

        if len(lineage_list) >= num_cells:
            break
            
    return lineage_list
