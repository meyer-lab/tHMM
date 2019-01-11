import numpy as np

    #make utils.py that stores all our helper functions (not related to the tree)

    
    
def remove_NaNs(X)
    '''Removes unfinished cells in a population'''
    count = 0
    for cell in X:
        if cell.isUnfinished():
            X.pop(count)
            count-=1
        count+=1
        
  
    
class tHMM:
    def __init__(self, X, numStates=1):
        ''' Instantiates a tHMM. '''
        self.X = X # list containing lineage, should be in correct format (contain no NaNs)
        self.numStates = numStates # number of discrete hidden states 
        self.get_numLineages() # gets the number of lineages in our population
        self.get_Population() # arranges the population into a list of lineages (each lineage might have varying length)
        self.get_paramlist() 
        
    def get_numLineages(self):
        ''' Outputs total number of cell lineages in given Population. '''
        linID_holder = [] # temporary list to hold all the linIDs of the cells in the population
        for cell in self.X: # for each cell in the population
            linID_holder.append(cell.linID) # append the linID of each cell
        self.numLineages = max(linID.holder)+1 # the number of lineages is the maximum linID+1
        return(self.numLineages)
    
    def get_Population(self):
        ''' Creates a full population list of lists which contain each lineage in the population. '''
        self.population = [] # full list to hold all the lineages
        for lineage_num in range(self.numLineages): # iterate over the number of lineages in the population
            temp_lineage = [] # temporary list to hold the cells of a certain lineage with a particular linID
            for cell in self.X: # for each cell in the population
                if cell.linID == lineage_num: # if the cell's linID is the lineage num
                    temp_lineage.append(cell) # append the cell to that certain lineage
            self.Population.append(temp_lineage) # append the lineage to the Population holder
        return(self.Population)
    
    def get_paramlist(self):
        ''' Creates a list of dictionaries holding the tHMM parameters for each lineage. '''
        temp_params = {"pi": np.zeros((self.numStates,1)), # inital state distributions [Kx1]
                       "T": np.zeros((self.numStates, self.numStates)), # state transition matrix [KxK]
                       "E": np.zeros((self.numStates,3))} # sequence of emission likelihood distribution parameters [Kx3]
        self.paramlist = [] # list that is numLineages long of parameters for each lineage tree in our population
        for lineage_num in range(self.numlineages): # for each lineage in our population
            self.paramlist.append(temp_params.copy()) # create a new dictionary holding the parameters and append it
        return(self.paramlist)
    
    '''
    The following are tree manipulating
    functions, that will be used when
    defining more complicated recursions
    when calculating probabilities for
    Downward and Upward recursions.
    '''
    
    def get_leaves(lineage):
        ''' Ouputs a list of leaves in a lineage. '''
        temp_leaves = [] # temporary list to hold the leaves of a lineage
        for cell in lineage: # for each cell in the lineage
            if (cell.left is None and cell.right is None) or (cell.left.isUnfinished() and cell.right.isUnfinished()): 
                # if the cell has no daughters or if the daughters had NaN times
                # why aren't we using isLeaf() here?
                temp_leaves.append(cell) # append those cells
        return(temp_leaves)
                        
    def tree_recursion(cell, subtree):
        ''' Basic recursion method used in all following tree traversal methods. '''
        if cell.Leaf(): # base case: if a leaf, end the recursion
            return
        subtree.append(cell.left)
        subtree.append(cell.right)
        tree_recursion(cell.left, subtree)
        tree_recursion(cell.right, subtree)
        return
    
    def get_subtrees(node,lineage):
        '''Get subtrees for one lineage'''
        subtree_list = [node] 
        tree_recursion(node,subtree)
        not_subtree = []
        for cell in lineage:
            if cell not in subtree:
                not_subtree.append(cell)
        return subtree, not_subtree
    
    def find_two_subtrees(node,lineage):
        '''Gets the left and right subtrees from a cell'''
        left_sub,_ = get_subtrees(cell.left,lineage)
        right_sub,_ = get_subtrees(cell.right,lineage)
        neither_subtree=[]
        for cell in lineage:
            if cell not in left_sub and cell not in right_sub:
                neither_subtree.append(cell)
        return left_sub, right_sub, neither_subtree
    
    def get_mixed_subtrees(node_m,node_n,lineage):
        m_sub,_ = get_subtrees(node_m,lineage)
        n_sub,_ = get_subtrees(node_n,lineage)
        mixed_sub = []
        for cell in m_sub:
            if cell not in n_sub:
                mixed_sub.append(cell)
        not_mixed = []
        for cell in lineage:
            if cell not in mixed_sub:
                not_mixed.append(cell)
        return mixed_sub, not_mixed
    
    '''
    This is the end of the necessary 
    tree manipulation helper functions.
    '''

    def get_Marginal_State_Distribution(self):
        '''
        Marginal State Distribution recursion from Durand et al, 2004
        '''
        self.MSD = [] # temporary Marginal State Distribution holder
        for num in self.numLineages: # for each lineage in our Population
            
            lineage = self.Population[num] # getting the lineage in the Population by index
            params = self.paramlist[num] # getting the respective params by index
            
            MSD_array = np.zeros((len(lineage),self.numStates)) # instantiating N by K array
            for cell in lineage: # for each cell in the lineage
                if cell.isRootParent(): # base case uses pi parameter
                    for states in self.numStates: # for each state
                        MSD_array[0][state] = params["pi"][state] # base case using pi parameter
                else:
                    parent_cell_idx = lineage.index(cell.parent) # get the index of the parent cell
                    current_cell_idx = lineage.index(cell) # get the index of the current cell
                    
                    for state_k in self.numStates: # recursion based on parent cell
                        temp_sum_holder = []
                        for state_j in self.numStates:
                            temp = params["T"][state_j][state_k] * MSD_array[parent_cell_idx][state_j]
                            temp_sum_holder.append(temp)
                        MSD_array[current_cell_idx][state_k] = sum(temp_sum_holder)
                        
            self.MSD.append(MSD_array) # Marginal States Distributions for each lineage in the Population
                        
            
        
    
    
    
