# contains the methods that completes Viterbi decoding algorithm

import numpy as np
from .tHMM_utils import *

def get_delta_leaves(tHMMobj):
    ''' 
        delta matrix and base case at the leaves.
        
        Each element in this N by K matrix is the 

    '''
    numStates = tHMMobj.numStates
    numLineages = tHMM.numLineages
    population = tHMM.population
    EL = tHMMobj.EL
    
    deltas = []
    
    for num in range(numLineages): # for each lineage in our Population
        lineage = population[num] # getting the lineage in the Population by index
        delta_array = np.zeros((len(lineage), numStates)) # instantiating N by K array
        EL_array = EL[num] # geting the EL of the respective lineage
        
        for cell in lineage: # for each cell in the lineage
            if cell.isLeaf(): # if it is a leaf
                leaf_cell_idx = lineage.index(cell) # get the index of the leaf
                delta_array[leaf_cell_idx, :] = EL_array[leaf_cell_idx, :]

        deltas.append(delta_array)
    return deltas

def get_delta_nonleaves(self, deltas):
        """ Calculates the delta values for all non-leaf cells. """
        for num in range(self.numLineages): # for each lineage in our Population
            lineage = self.population[num] # getting the lineage in the Population by index
            EL_array = self.EL[num] # geting the EL of the respective lineage
            params = self.paramlist[num] # getting the respective params by lineage index
            T = params["T"] # getting the transition matrix of the respective lineage
            start = max_gen(lineage) # start at the leafs in the maximum generation
            while start > 1: # move up one generation until the 2nd generation is the children and the root nodes are the parents
                level = get_gen(start, lineage)
                parent_holder = get_parents_for_level(level, lineage)
                for node_parent_m_idx in parent_holder:
                    for state_k in range(self.numStates):
                        fac1 = self.get_delta_parent_child_prod(lineage, self.deltas[num], self.betas[num], T, state_k, node_parent_m_idx)
                        fac2 = EL_array[node_parent_m_idx, state_k]
                        deltas[num][node_parent_m_idx, state_k] = fac1*fac2

                start -= 1
    
def get_delta_parent_child_prod(self, lineage, delta_array, beta_array, T, state_j, node_parent_m_idx):
        """ Calculates the delta coefficient for every parent-child relationship of a given parent cell in a given state. """
        delta_m_n_holder = [] # list to hold the factors in the product
        node_parent_m = lineage[node_parent_m_idx] # get the index of the parent
        children_idx_list = [] # list to hold the children
        if node_parent_m.left: #when you say .left, it means it exists and it will go through
            node_child_n_left_idx = lineage.index(node_parent_m.left)
            children_idx_list.append(node_child_n_left_idx)
        if node_parent_m.right:
            node_child_n_right_idx = lineage.index(node_parent_m.right)
            children_idx_list.append(node_child_n_right_idx)
        for node_child_n_idx in children_idx_list:
            delta_m_n =self.delta_parent_child_func(lineage, delta_array, beta_array, T, state_j, node_parent_m_idx, node_child_n_idx)
            delta_m_n_holder.append(delta_m_n)

        result = reduce((lambda x, y: x * y), delta_m_n_holder) # calculates the product of items in a list
        return result

    
    
def delta_parent_child_func(self, lineage, delta_array, beta_array, T, state_j, node_parent_m_idx, node_child_n_idx):
        """ Calculates the delta coefficient for a single parent-child relationship where the parent is in a given state. """
        assert( lineage[node_child_n_idx].parent is lineage[node_parent_m_idx]) # check the child-parent relationship
        assert( lineage[node_child_n_idx].isChild() ) # if the child-parent relationship is correct, then the child must be either the left daughter or the right daughter
        max_holder=[] # summing over the states
        for state_k in range(self.numStates): # for each state k
            num1 = beta_array[node_child_n_idx, state_k] # get the already calculated beta at node n for state k
            num2 = T[state_j, state_k] # get the transition rate for going from state j to state k
            # P( z_n = k | z_m = j)

            max_holder.append(num1*num2)
        return max(max_holder)



def Viterbi(self):
        """ Runs the viterbi algorithm and returns a list of arrays containing the optimal state of each cell. """
        all_states = []
        for num in range(self.numLineages):
            delta_array = self.deltas[num] # deltas are not being manip. just accessed so this is OK
            lineage = self.population[num]
            params = self.paramlist[num]
            T = params['T']
            pi = params['pi']

            opt_state_tree = np.zeros((len(lineage)), dtype=int)
            possible_first_states = np.multiply(delta_array[0,:], pi)
            opt_state_tree[0] = np.argmax(possible_first_states)
            max_level = max_gen(lineage)
            count = 1 # start at the root nodes
            while count < max_level: # move down until the lowest leaf node is reached
                level = get_gen(count, lineage)
                for cell in level:
                    parent_idx = lineage.index(cell)
                    temp = get_daughters(cell)
                    for n in temp:
                        child_idx = lineage.index(n)
                        parent_state = opt_state_tree[parent_idx]
                        possible_states = np.multiply(delta_array[child_idx,:], T[parent_state,:])
                        opt_state_tree[child_idx] = np.argmax(possible_states)
                count += 1
            all_states.append(opt_state_tree)

        return all_states
