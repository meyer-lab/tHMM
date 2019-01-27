""" This file holds the parameters of our tHMM in the tHMM class. """

import numpy as np
import scipy.stats as sp
from .Lineage_utils import get_numLineages, init_Population

class tHMM:
    """ Main tHMM class. """
    def __init__(self, X, numStates=1):
        ''' Instantiates a tHMM. '''
        self.X = X # list containing lineage, should be in correct format (contain no NaNs)
        self.numStates = numStates # number of discrete hidden states
        self.numLineages = get_numLineages(self.X) # gets the number of lineages in our population
        self.population = init_Population(self.X, self.numLineages) # arranges the population into a list of lineages (each lineage might have varying length)
        assert(self.numLineages == len(self.population))
        self.paramlist = self.init_paramlist() # list that is numLineages long of parameters for each lineage tree in our population
        
        self.MSD = self.get_Marginal_State_Distributions() # full Marginal State Distribution holder
        self.EL = self.get_Emission_Likelihoods() # full Emission Likelihood holder

    def init_paramlist(self):
        ''' Creates a list of dictionaries holding the tHMM parameters for each lineage. '''
        paramlist = []
        numStates = self.numStates
        numLineages = self.numLineages
        temp_params = {"pi": np.ones((numStates)) / numStates, # inital state distributions [K] initialized to 1/K
                       "T": np.ones((numStates, numStates)) / numStates, # state transition matrix [KxK] initialized to 1/K
                       "E": np.ones((numStates, 3))} # sequence of emission likelihood distribution parameters [Kx3]
        temp_params["E"][:,0] *= 0.5 # initializing all Bernoulli p parameters to 0.5
        temp_params["E"][:,1] *= 2. # initializing all Gompertz c parameters to 2
        temp_params["E"][:,2] *= 50. # initializing all Gompoertz s(cale) parameters to 50

        for lineage_num in range(numLineages): # for each lineage in our population
            paramlist.append(temp_params.copy()) # create a new dictionary holding the parameters and append it
            assert(len(paramlist) == lineage_num+1)

        return paramlist

    def get_Marginal_State_Distributions(self):
        '''
        Marginal State Distribution (MSD) matrix and recursion.
        This is the probability that a hidden state variable z_n is of
        state k, that is, each value in the N by K MSD array for each lineage is
        the probability
        
        P(z_n = k),

        for all z_n in the hidden state tree
        and for all k in the total number of discrete states. Each MSD array is
        an N by K array (an entry for each cell and an entry for each state),
        and each lineage has its own MSD array.
        '''
        numStates = self.numStates
        numLineages = self.numLineages
        population = self.population
        paramlist = self.paramlist

        MSD = []

        for num in range(numLineages): # for each lineage in our Population
            lineage = population[num] # getting the lineage in the Population by lineage index
            params = paramlist[num] # getting the respective params by lineage index
            MSD_array = np.zeros((len(lineage),numStates)) # instantiating N by K array

            for cell in lineage: # for each cell in the lineage
                if cell.isRootParent(): # base case uses pi parameter at the root cells of the tree

                    for state in range(numStates): # for each state
                        MSD_array[0,state] = params["pi"][state] # base case using pi parameter
                else:
                    parent_cell_idx = lineage.index(cell.parent) # get the index of the parent cell
                    current_cell_idx = lineage.index(cell) # get the index of the current cell

                    for state_k in range(numStates): # recursion based on parent cell
                        temp_sum_holder = [] # for all states k, calculate the sum of temp

                        for state_j in range(numStates): # for all states j, calculate temp
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

        P(x_n = x | z_n = k),

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
        numStates = self.numStates
        numLineages = self.numLineages
        population = self.population
        paramlist = self.paramlist

        EL = []

        for num in range(numLineages): # for each lineage in our Population
            lineage = population[num] # getting the lineage in the Population by lineage index
            params = paramlist[num] # getting the respective params by lineage index
            EL_array = np.zeros((len(lineage), numStates)) # instantiating N by K array for each lineage
            E_param_array = params["E"] # K by 3 array of distribution parameters for each lineage

            for state_k in range(numStates): # for each state
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
