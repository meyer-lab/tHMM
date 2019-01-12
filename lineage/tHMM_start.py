import numpy as np
import scipy.stats as sp

    #make utils.py that stores all our helper functions (not related to the tree)

    
    
def remove_NaNs(X)
    '''Removes unfinished cells in a population'''

    for cell in X:
        unfinished_leaf_cell_idx = lineage.index(cell)
        if cell.isUnfinished():
            if cell.parent.left is cell:
                cell.parent.left = None
            if cell.parent.right is cell:
                cell.parent.right = None
            X.pop(unfinished_leaf_cell_idx)        
  
    
class tHMM:
    def __init__(self, X, numStates=1):
        ''' Instantiates a tHMM. '''
        self.X = X # list containing lineage, should be in correct format (contain no NaNs)
        self.numStates = numStates # number of discrete hidden states 
        self.get_numLineages() # gets the number of lineages in our population
        self.get_Population() # arranges the population into a list of lineages (each lineage might have varying length)
        self.get_paramlist() 
        self.get_Marginal_State_Distributions()
        self.get_Emission_Likelihoods()
        self.get_get_leaf_Norms()
        
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
                       "E": np.zeros((self.numStates, 3))} # sequence of emission likelihood distribution parameters [Kx3]
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
    
    '''
    #Think about deleting
    def get_leaves(lineage):
        ''' Ouputs a list of leaves in a lineage. '''
        temp_leaves = [] # temporary list to hold the leaves of a lineage
        for cell in lineage: # for each cell in the lineage
            if (cell.left is None and cell.right is None) or (cell.left.isUnfinished() and cell.right.isUnfinished()): 
                # if the cell has no daughters or if the daughters had NaN times
                # why aren't we using isLeaf() here?
                temp_leaves.append(cell) # append those cells
        return(temp_leaves)
     '''
                        
    def tree_recursion(cell, subtree):
        ''' Basic recursion method used in all following tree traversal methods. '''
        if cell.isLeaf(): # base case: if a leaf, end the recursion
            return
        if cell.left is not None:
            subtree.append(cell.left)
            tree_recursion(cell.left, subtree)
        if cell.right is not None:
            subtree.append(cell.right)
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

    def get_Marginal_State_Distributions(self):
        '''
            Marginal State Distribution (MSD) recursion from Durand et al, 2004. 
            This is the probability that a hidden state variable z_n is of
            state k, that is, each value in the MSD array for each lineage is
            the probability P(z_n = k) for all z_n in the hidden state tree
            and for all k in the total number of discrete states. Each MSD array is
            an N by K array (an entry for each cell and an entry for each state,
            and each lineage has its own MSD array.
            
            Unit test should be that the addition of all elements in each row 
            for every row is equal to 1.
        '''
        self.MSD = [] # full Marginal State Distribution holder
        for num in self.numLineages: # for each lineage in our Population
            
            lineage = self.Population[num] # getting the lineage in the Population by index
            params = self.paramlist[num] # getting the respective params by index
            
            MSD_array = np.zeros((len(lineage),self.numStates)) # instantiating N by K array
            for cell in lineage: # for each cell in the lineage
                if cell.isRootParent(): # base case uses pi parameter
                    for states in self.numStates: # for each state
                        MSD_array[0,state] = params["pi"][state,:] # base case using pi parameter
                else:
                    parent_cell_idx = lineage.index(cell.parent) # get the index of the parent cell
                    current_cell_idx = lineage.index(cell) # get the index of the current cell
                    
                    for state_k in self.numStates: # recursion based on parent cell
                        temp_sum_holder = []
                        for state_j in self.numStates:
                            temp = params["T"][state_j,state_k] * MSD_array[parent_cell_idx][state_j]
                            temp_sum_holder.append(temp)
                        MSD_array[current_cell_idx,state_k] = sum(temp_sum_holder)
                        
            self.MSD.append(MSD_array) # Marginal States Distributions for each lineage in the Population
        return(self.MSD)

                        
    def get_Emission_Likelihoods(self):
        '''
            Emission Likelihood matrix. Each element in this N by K matrix
            represents the probability P(x_n = x | z_n = k) for x_n and z_n
            in our observed and hidden state tree and for all possible discrete
            states k.
        '''
        self.EL = [] # full Emission Likelihood holder
        for num in self.numLineages: # for each lineage in our Population

            lineage = self.Population[num] # getting the lineage in the Population by index
            params = self.paramlist[num] # getting the respective params by index

            EL_array = np.zeros((len(lineage), self.numStates)) # instantiating N by K array
            E_param_array = params["E"] # K by 3 array 

            for state_k in self.numStates: # for each state 
                E_param_k = E_param_array[state,:] # get the emission parameters for that state
                k_bern = E_param_k[0] # bernoulli rate parameter
                k_gomp_c = E_param_k[1] # gompertz c parameter
                k_gomp_s = E_param_k[2] # gompertz scale parameter

                for cell in lineage: # for eac
                    temp_b = sp.stats.bernoulli.pmf(k=cell.fate, p=k_bern) # bernoulli likelihood
                    temp_g = sp.stats.gompertz.pdf(x=cell.tau, c=k_gomp_c, scale=k_gomp_s) # gompertz likelihood 

                    current_cell_idx = lineage.index(cell) # get the index of the current cell

                    EL_array[current_cell_idx, state_k] = temp_b * temp_g

            self.EL.append(EL_array)
        return(self.EL)

    def get_leaf_Norms(self):
        '''
            Gets the normalizing factor for the downward recursion
            only for the leaves.
            We first calculate the joint following probability
            using the definition of conditional probability:
            
            P(x_n = x | z_n = k) * P(z_n = k) = 
            P(x_n = x , z_n = k), 
            
            and in code,
            
            EL[n,k] * MSD[n,k] = Norms[n].
            
            We can then sum this probability over k, using the
            law of total probability:
            
            sum_k ( P(x_n = x , z_n = k) ) = P(x_n = x).
        '''
        self.Norms = []
        for num in self.numLineages: # for each lineage in our Population
            
            Norm_array = np.zeros((len(lineage), 1)) # instantiating N by 1 array
                
            lineage = self.Population[num] # getting the lineage in the Population by index
            MSD_array = self.MSD[num] # getting the MSD of the respective lineage
            EL_array = self.EL[num] # geting the EL of the respective lineage

            for cell in lineage: # for each cell in the lineage
                if cell.isLeaf(): # if it is a leaf
                    leaf_cell_idx = lineage.index(cell) # get the index of the leaf
                    temp_sum_holder = [] # create a temporary list 
                    
                    for state_k in self.numstates: # for each state
                        joint_prob = MSD_array[leaf_cell_idx, state_k] * EL_array[leaf_cell_idx, state_k] # calculate the product
                        # this product is the joint probability
                        temp_sum_holder.append(joint_prob) # append the joint probability
                        
                    Norm_array[leaf_cell_idx] = sum(temp_sum_holder) # law of total probability
                    # the sum of the joint probabilities is the marginal probability
                    
            self.Norms.append(Norm_array)
        return(self.Norms)
                    
            
            
    
    
    
