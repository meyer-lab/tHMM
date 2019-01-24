import numpy as np
import scipy.stats as sp
from functools import reduce # used to take the product of items in a list

    #make utils.py that stores all our helper functions (not related to the tree)

    
    
def remove_NaNs(X):
    '''Removes unfinished cells in a population'''
    ii = 0 # establish a count outside of the loop
    while ii in range(len(X)): # for each cell in X
        if X[ii].isUnfinished(): # if the cell has NaNs in its times
            if X[ii].parent is None: # do nothing if the parent pointer doesn't point to a cell
                pass
            elif X[ii].parent.left is X[ii]: # if it is the left daughter of the parent cell
                X[ii].parent.left = None # replace the cell with None
            elif X[ii].parent.right is X[ii]: # or if it is the right daughter of the parent cell
                X[ii].parent.right = None # replace the cell with None
            X.pop(ii) # pop the unfinished cell at the current position
        else:
            ii += 1 # only move forward in the list if you don't delete a cell
    return X  

def max_gen(lineage):
        '''finds the max generation in a lineage'''
        gen_holder = 1
        for cell in lineage:
            if cell.gen > gen_holder:
                gen_holder = cell.gen
        return gen_holder
    
def get_gen(gen, lineage):
    '''creates a list with all cells in the given generation'''
    first_set = []
    for cell in lineage:
        if cell.gen == gen:
            first_set.append(cell)
    return first_set

def get_numLineages(X):
    ''' Outputs total number of cell lineages in given Population. '''
    linID_holder = [] # temporary list to hold all the linIDs of the cells in the population
    for cell in X: # for each cell in the population
        linID_holder.append(cell.linID) # append the linID of each cell
    numLineages = max(linID_holder)+1 # the number of lineages is the maximum linID+1
    return numLineages

def init_Population(X, numLineages):
    ''' Creates a full population list of lists which contain each lineage in the population. '''
    population = []
    for lineage_num in range(numLineages): # iterate over the number of lineages in the population
        temp_lineage = [] # temporary list to hold the cells of a certain lineage with a particular linID
        for cell in X: # for each cell in the population
            if cell.linID == lineage_num: # if the cell's linID is the lineage num
                temp_lineage.append(cell) # append the cell to that certain lineage
        population.append(temp_lineage) # append the lineage to the Population holder
    return population

def get_parents_for_level(level, lineage):
    parent_holder = set() #set makes sure only one index is put in and no overlap
    for cell in level:
        parent_cell = cell.parent
        parent_holder.add(lineage.index(parent_cell))
    return parent_holder

def get_daughters(cell):
    temp = []
    if cell.left:
        temp.append(cell.left)
    if cell.right:
        temp.append(cell.right)
    return temp

class tHMM:
    def __init__(self, X, numStates=1):
        ''' Instantiates a tHMM. '''
        self.X = X # list containing lineage, should be in correct format (contain no NaNs)
        self.numStates = numStates # number of discrete hidden states
        self.numLineages = get_numLineages(self.X) # gets the number of lineages in our population
        self.population = init_Population(self.X, self.numLineages) # arranges the population into a list of lineages (each lineage might have varying length)
        self.paramlist = self.init_paramlist() # list that is numLineages long of parameters for each lineage tree in our population
        self.MSD = self.get_Marginal_State_Distributions() # full Marginal State Distribution holder
        self.EL = self.get_Emission_Likelihoods() # full Emission Likelihood holder
        self.NF = self.get_leaf_Normalizing_Factors()
        self.betas = self.get_beta_leaves()
        self.get_beta_and_NF_nonleaves() # this function might cause some problems
        # adam do you still need this comment?
        self.LL = self.calculate_log_likelihood() # calculates the LL after the first pass
        self.deltas = self.get_delta_leaves()
        self.get_delta_nonleaves()

    def init_paramlist(self):
        ''' Creates a list of dictionaries holding the tHMM parameters for each lineage. '''
        paramlist = []
        temp_params = {"pi": np.zeros((self.numStates)), # inital state distributions [K]
                       "T": np.zeros((self.numStates, self.numStates)), # state transition matrix [KxK]
                       "E": np.zeros((self.numStates, 3))} # sequence of emission likelihood distribution parameters [Kx3]
        for lineage_num in range(self.numLineages): # for each lineage in our population
            paramlist.append(temp_params.copy()) # create a new dictionary holding the parameters and append it
        return paramlist

    def get_Marginal_State_Distributions(self):
        '''
            Marginal State Distribution (MSD) matrix and recursion. 
            
            This is the probability that a hidden state variable z_n is of
            state k, that is, each value in the N by K MSD array for each lineage is
            the probability 
            
            P(z_n = k) 
            
            for all z_n in the hidden state tree
            and for all k in the total number of discrete states. Each MSD array is
            an N by K array (an entry for each cell and an entry for each state),
            and each lineage has its own MSD array.
            
            Unit test should be that the addition of all elements in each row 
            for every row is equal to 1.
        '''
        MSD = []
        for num in range(self.numLineages): # for each lineage in our Population
            lineage = self.population[num] # getting the lineage in the Population by lineage index
            params = self.paramlist[num] # getting the respective params by lineage index
            MSD_array = np.zeros((len(lineage),self.numStates)) # instantiating N by K array
            for cell in lineage: # for each cell in the lineage
                if cell.isRootParent(): # base case uses pi parameter at the root cells of the tree
                    for state in range(self.numStates): # for each state
                        MSD_array[0,state] = params["pi"][state] # base case using pi parameter
                else:
                    parent_cell_idx = lineage.index(cell.parent) # get the index of the parent cell
                    current_cell_idx = lineage.index(cell) # get the index of the current cell
                    for state_k in range(self.numStates): # recursion based on parent cell
                        temp_sum_holder = [] # for all states k, calculate the sum of temp
                        for state_j in range(self.numStates): # for all states j, calculate temp
                            temp = params["T"][state_j,state_k] * MSD_array[parent_cell_idx, state_j]
                            # temp = T_jk * P(z_parent(n) = j)
                            temp_sum_holder.append(temp)
                        MSD_array[current_cell_idx,state_k] = sum(temp_sum_holder)

            MSD.append(MSD_array) # Marginal States Distributions for each lineage in the Population

        return MSD

                        
    def get_Emission_Likelihoods(self):
        '''
            Emission Likelihood (EL) matrix. 
            
            Each element in this N by K matrix represents the probability 
            
            P(x_n = x | z_n = k) 
            
            for all x_n and z_n in our observed and hidden state tree 
            and for all possible discrete states k. Since we have a 
            multiple observation model, that is
            
            x_n = {x_B, x_G},
            
            consists of more than one observation, x_B = division(1) or 
            death(0) (which is one of the observations x_B) and the other
            being, x_G = lifetime, lifetime >=0, (which is the other observation x_G)
            we make the assumption that
            
            P(x_n = x | z_n = k) = P(x_n1 = x_B | z_n = k) * P(x_n = x_G | z_n = k).
        '''
        EL = []
        for num in range(self.numLineages): # for each lineage in our Population
            lineage = self.population[num] # getting the lineage in the Population by lineage index
            params = self.paramlist[num] # getting the respective params by lineage index
            EL_array = np.zeros((len(lineage), self.numStates)) # instantiating N by K array for each lineage
            E_param_array = params["E"] # K by 3 array of distribution parameters for each lineage

            for state_k in range(self.numStates): # for each state 
                E_param_k = E_param_array[state_k,:] # get the emission parameters for that state
                k_bern = E_param_k[0] # bernoulli rate parameter
                k_gomp_c = E_param_k[1] # gompertz c parameter
                k_gomp_s = E_param_k[2] # gompertz scale parameter

                for cell in lineage: # for each cell in the lineage
                    temp_b = sp.bernoulli.pmf(k=cell.fate, p=k_bern) # bernoulli likelihood
                    temp_g = sp.gompertz.pdf(x=cell.tau, c=k_gomp_c, scale=k_gomp_s) # gompertz likelihood 

                    current_cell_idx = lineage.index(cell) # get the index of the current cell

                    EL_array[current_cell_idx, state_k] = temp_b * temp_g

            EL.append(EL_array) # append the EL_array for each lineage

        return EL

    def get_leaf_Normalizing_Factors(self):
        '''
            Normalizing factor (NF) matrix and base case at the leaves. 
            
            Each element in this N by 1 matrix is the normalizing 
            factor for each beta value calculation for each node.
            This normalizing factor is essentially the marginal
            observation distribution for a node.
            
            This function gets the normalizing factor for 
            the upward recursion only for the leaves.
            We first calculate the joint probability
            using the definition of conditional probability:
            
            P(x_n = x | z_n = k) * P(z_n = k) = P(x_n = x , z_n = k).  
            
            We can then sum this joint probability over k, 
            which are the possible states z_n can be,
            and through the law of total probability, 
            obtain the marginal observation distribution 
            P(x_n = x):
            
            sum_k ( P(x_n = x , z_n = k) ) = P(x_n = x).
            
        '''
        NF = [] # full Normalizing Factors holder
        for num in range(self.numLineages): # for each lineage in our Population
            lineage = self.population[num] # getting the lineage in the Population by index
            NF_array = np.zeros((len(lineage))) # instantiating N by 1 array
            MSD_array = self.MSD[num] # getting the MSD of the respective lineage
            EL_array = self.EL[num] # geting the EL of the respective lineage

            for cell in lineage: # for each cell in the lineage
                if cell.isLeaf(): # if it is a leaf
                    leaf_cell_idx = lineage.index(cell) # get the index of the leaf
                    temp_sum_holder = [] # create a temporary list 
                    for state_k in range(self.numStates): # for each state
                        joint_prob = MSD_array[leaf_cell_idx, state_k] * EL_array[leaf_cell_idx, state_k] # def of conditional prob
                        # P(x_n = x , z_n = k) = P(x_n = x | z_n = k) * P(z_n = k)
                        # this product is the joint probability
                        
                        # maybe we can consider making this a dot product instead of looping and summing
                        # but I feel like that would be less readable at the sake of speed
                        
                        temp_sum_holder.append(joint_prob) # append the joint probability to be summed
                        
                    marg_prob = sum(temp_sum_holder) # law of total probability
                    # P(x_n = x) = sum_k ( P(x_n = x , z_n = k) )
                    # the sum of the joint probabilities is the marginal probability
                    
                    NF_array[leaf_cell_idx] = marg_prob # each cell gets its own marg prob

            NF.append(NF_array)

        return NF
                    
    def get_beta_leaves(self):
        '''
            beta matrix and base case at the leaves.
            
            Each element in this N by K matrix is the beta value
            for each cell and at each state. In particular, this
            value is derived from the Marginal State Distributions
            (MSD), the Emission Likelihoods (EL), and the 
            Normalizing Factors (NF). Each beta value
            for the leaves is exactly the probability
            
            beta[n,k] = P(z_n = k | x_n = x).
            
            Using Bayes Theorem, we see that the above equals
            
                        P(x_n = x | z_n = k) * P(z_n = k)
            beta[n,k] = _________________________________
                                    P(x_n = x)
            
            The first value in the numerator is the Emission
            Likelihoods. The second value in the numerator is
            the Marginal State Distributions. The value in the
            denominator is the Normalizing Factor.                                
        '''
        betas = [] # full betas holder
        for num in range(self.numLineages): # for each lineage in our Population
            lineage = self.population[num] # getting the lineage in the Population by index
            beta_array = np.zeros((len(lineage), self.numStates)) # instantiating N by K array
            MSD_array = self.MSD[num] # getting the MSD of the respective lineage
            EL_array = self.EL[num] # geting the EL of the respective lineage
            NF_array = self.NF[num]
            for cell in lineage: # for each cell in the lineage
                if cell.isLeaf(): # if it is a leaf
                    leaf_cell_idx = lineage.index(cell) # get the index of the leaf
                    for state_k in range(self.numStates): # for each state 
                        # see expression in docstring
                        num1 = EL_array[leaf_cell_idx, state_k] # Emission Likelihood
                        #  P(x_n = x | z_n = k)
                        num2 = MSD_array[leaf_cell_idx, state_k] # Marginal State Distribution
                        # P(z_n = k)
                        denom = NF_array[leaf_cell_idx] # Normalizing Factor (same regardless of state)
                        # P(x_n = x)
                        beta_array[leaf_cell_idx, state_k] = num1 * num2 / denom

            betas.append(beta_array)
        return betas
    
    def beta_parent_child_func(self, lineage, beta_array, T, MSD_array, state_j, node_parent_m_idx, node_child_n_idx):
        '''
            This "helper" function calculates the probability 
            described as a 'beta-link' between parent and child
            nodes in our tree for some state j. This beta-link
            value is what lets you calculate the values of
            higher (in the direction from the leave
            to the root node) node beta and Normalizing Factor
            values.
        '''
        assert( lineage[node_child_n_idx].parent is lineage[node_parent_m_idx]) # check the child-parent relationship
        assert( lineage[node_child_n_idx].isChild() ) # # if the child-parent relationship
        # is correct, then the child must be either the left daughter or the right daughter
        summand_holder=[] # summing over the states
        for state_k in range(self.numStates): # for each state k
            num1 = beta_array[node_child_n_idx, state_k] # get the already calculated beta at node n for state k
            num2 = T[state_j, state_k] # get the transition rate for going from state j to state k
            # P( z_n = k | z_m = j)
            denom = MSD_array[node_child_n_idx, state_k] # get the MSD for node n at state k
            # P(z_n = k)

            summand_holder.append(num1*num2/denom)
        return sum(summand_holder)
    
    def get_beta_parent_child_prod(self, lineage, beta_array, T, MSD_array, state_j, node_parent_m_idx):
        beta_m_n_holder = [] # list to hold the factors in the product
        node_parent_m = lineage[node_parent_m_idx] # get the index of the parent
        children_idx_list = [] # list to hold the children
        if node_parent_m.left is not None:
            node_child_n_left_idx = lineage.index(node_parent_m.left)
            children_idx_list.append(node_child_n_left_idx)
        if node_parent_m.right is not None:
            node_child_n_right_idx = lineage.index(node_parent_m.right)
            children_idx_list.append(node_child_n_right_idx)
        for node_child_n_idx in children_idx_list:
            beta_m_n = self.beta_parent_child_func(lineage, beta_array, T, MSD_array, state_j, node_parent_m_idx, node_child_n_idx)
            beta_m_n_holder.append(beta_m_n)

        result = reduce((lambda x, y: x * y), beta_m_n_holder) # calculates the product of items in a list
        return result

    def get_beta_and_NF_nonleaves(self):
        for num in range(self.numLineages): # for each lineage in our Population
            lineage = self.population[num] # getting the lineage in the Population by index
            MSD_array = self.MSD[num] # getting the MSD of the respective lineage
            EL_array = self.EL[num] # geting the EL of the respective lineage
            params = self.paramlist[num] # getting the respective params by lineage index
            T = params["T"] # getting the transition matrix of the respective lineage

            start = max_gen(lineage) # start at the lowest level of the lineage
            while start > 1:
                level = get_gen(start, lineage)
                parent_holder = get_parents_for_level(level, lineage)
                for node_parent_m_idx in parent_holder:
                    num_holder = []
                    for state_k in range(self.numStates):
                        fac1 = self.get_beta_parent_child_prod(lineage=lineage,
                                                          beta_array=self.betas[num],
                                                          T=T,
                                                          MSD_array=MSD_array,
                                                          state_j=state_k, 
                                                          node_parent_m_idx = node_parent_m_idx)
                        fac2 = EL_array[node_parent_m_idx, state_k]
                        fac3 = MSD_array[node_parent_m_idx, state_k]
                        num_holder.append(fac1*fac2*fac3)
                    self.NF[num][node_parent_m_idx] = sum(num_holder)
                    for state_k in range(self.numStates):
                        self.betas[num][node_parent_m_idx, state_k] = num_holder[state_k] / self.NF[num][node_parent_m_idx]           

                start -= 1
                
    def calculate_log_likelihood(self):
        """ Calculates log likelihood."""
        LL = []
        for num in range(self.numLineages): # for each lineage in our Population
            lineage = self.population[num] # getting the lineage in the Population by index
            NF_array = self.NF[num] # getting the NF of the respective lineage
            log_NF_array = np.log(NF_array)
            ll_per_num = sum(log_NF_array)
            LL.append(ll_per_num)
        return LL
    
############ VITERBI #############        

    def get_delta_leaves(self):
        ''' creates a deltas list for all cells but only calculates the delta value for the leaves '''
        deltas = []
        for num in range(self.numLineages): # for each lineage in our Population
            lineage = self.population[num] # getting the lineage in the Population by index
            delta_array = np.zeros((len(lineage), self.numStates)) # instantiating N by K array
            EL_array = self.EL[num] # geting the EL of the respective lineage
            for cell in lineage: # for each cell in the lineage
                if cell.isLeaf(): # if it is a leaf
                    leaf_cell_idx = lineage.index(cell) # get the index of the leaf                     
                    delta_array[leaf_cell_idx, :] = EL_array[leaf_cell_idx, :]

            deltas.append(delta_array)
        return deltas

    def delta_parent_child_func(self, lineage, delta_array, beta_array, T, state_j, node_parent_m_idx, node_child_n_idx):
        assert( lineage[node_child_n_idx].parent is lineage[node_parent_m_idx]) # check the child-parent relationship
        assert( lineage[node_child_n_idx].isChild() ) # if the child-parent relationship is correct, then the child must be either the left daughter or the right daughter
        max_holder=[] # summing over the states
        for state_k in range(self.numStates): # for each state k
            num1 = beta_array[node_child_n_idx, state_k] # get the already calculated beta at node n for state k
            num2 = T[state_j, state_k] # get the transition rate for going from state j to state k
            # P( z_n = k | z_m = j)

            max_holder.append(num1*num2)
        return max(max_holder)
        
        
    def get_delta_parent_child_prod(self, lineage, delta_array, beta_array, T, state_j, node_parent_m_idx):
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

    def get_delta_nonleaves(self):
        for num in range(self.numLineages): # for each lineage in our Population
            lineage = self.population[num] # getting the lineage in the Population by index
            EL_array = self.EL[num] # geting the EL of the respective lineage
            params = self.paramlist[num] # getting the respective params by lineage index
            T = params["T"] # getting the transition matrix of the respective lineage
            start = max_gen(lineage)
            while start > 1:
                level = get_gen(start, lineage)
                parent_holder = get_parents_for_level(level, lineage)
                for node_parent_m_idx in parent_holder:
                    for state_k in range(self.numStates):
                        fac1 = self.get_delta_parent_child_prod(lineage, self.deltas[num], self.betas[num], T, state_k, node_parent_m_idx)
                        fac2 = EL_array[node_parent_m_idx, state_k]
                        self.deltas[num][node_parent_m_idx, state_k] = fac1*fac2

                start -= 1

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
            count = 1
            while count < max_level:
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
