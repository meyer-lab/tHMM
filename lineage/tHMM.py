""" This file holds the parameters of our tHMM in the tHMM class. """

import numpy as np
import scipy.stats as sp
from .Lineage_utils import get_numLineages, init_Population
from .tHMM_utils import max_gen, get_gen


class tHMM:
    """ Main tHMM class. """
    def __init__(self, X, numStates=2, FOM='G', keepBern=True):
        ''' Instantiates a tHMM.

        This function uses the following functions and assings them to the cells
        (objects) in the lineage.

        Args:
            ----------
            X (list of objects): A list of objects (cells) in a lineage in which
            the NaNs have been removed.

            numStates (int): the number of hidden states that we want our model have

            FOM (str): For now, it is either "E": Exponential, or "G": Gompertz
            and it determines the type of distribution for lifetime of the cells

            keepBern (bool): ?
  
        '''
        # list containing lineage, should be in correct format (contain no NaNs)
        self.X = X 
        # number of discrete hidden states
        self.numStates = numStates 
        self.FOM = FOM
        self.keepBern = keepBern
        # gets the number of lineages in our population
        self.numLineages = get_numLineages(self.X) 
        # arranges the population into a list of lineages (each lineage might have varying length)
        self.population = init_Population(self.X, self.numLineages) 


        assert self.numLineages == len(self.population), "Something is wrong with the number of lineages in your population member variable for your tHMM class and the number of lineages member variable for your tHMM class. Check the number of unique root cells and the number of lineages in your data."
        # list that is numLineages long of parameters for each lineage tree in our population
        self.paramlist = self.init_paramlist() 

        # full Marginal State Distribution holder
        self.MSD = self.get_Marginal_State_Distributions() 
        # full Emission Likelihood holder
        self.EL = self.get_Emission_Likelihoods() 


#####-------------------- Create list of parameters -------------------------#####

    def init_paramlist(self):
        ''' Creates a list of dictionaries holding the tHMM parameters for each lineage.
        In this function, the dictionary is initialized.
        
        There are three matrices in this function:
            1. "pi" (The initial probability matrix): The matrix holding the probability of
            being in each state at time = 0. This matrix is a [K x 1] matrix assuming
            we have K hidden states. The matrix is initialized uniformly.
            
            2. "T" (Transition probability matrix): The matrix holding the probability of 
            transitioning between different states. This is a [K x K] matrix assuming 
            we have K hidden states. The matrix in initialized uniformly.
            
            3. "E" (Emission probability matrix): The matrix holding the probability of
            emissions (observations) corresponding to each state. In this case, emissions are
            1. whether a cell dies or divides (Bernoulli distribution with 1 parameter)
            2. how long a cell lives:
                2.1. Gompertz distribution with 2 parameters (loc and scale)
                    loc initialized to 2
                    scale initialized to 62.5
                2.2. Exponential distribution with 1 parameter (beta)
                    beta intialized to 62.5
                
            If the Gompertz is used, on the whole we will have 3 parameters for emissions, so
            the emission matrix will be [K x 3].
            if the Exponential is used, on the whole we will have 2 parameters for emissions, so
            the emission matrix will be [K x 2].

        Returns:
            ----------
            paramlist (dictionary): a dictionary holding three matrices mentioned above.

        '''
        paramlist = []
        numStates = self.numStates
        numLineages = self.numLineages
        
        ##### Initializing the Emission matrix uniformly ######
        
        temp_params = {"pi": np.ones((numStates)) / numStates, # inital state distributions [K] initialized to 1/K
                       "T": np.ones((numStates, numStates)) / numStates} # state transition matrix [KxK] initialized to 1/K
        
        ### If the lifetime determination is using Gompertz:
        if self.FOM == 'G':
            temp_params["E"] = np.ones((numStates, 3)) # sequence of emission likelihood distribution parameters [Kx3]
            for state_j in range(numStates):
                temp_params["E"][state_j, 0] = 1/numStates # initializing all Bernoulli p parameters to 1/numStates
                temp_params["E"][state_j, 1] = 2.0*(1+np.random.uniform()) # initializing all Gompertz c parameters to 2
                temp_params["E"][state_j, 2] = 62.5*(1+np.random.uniform()) # initializing all Gompoertz scale parameters to 62.5
        
        ### If the lifetime determination is using Exponential:
        elif self.FOM == 'E':
            temp_params["E"] = np.ones((numStates, 2)) # sequence of emission likelihood distribution parameters [Kx2]
            for state_j in range(numStates):
                temp_params["E"][state_j, 0] = 1/numStates # initializing all Bernoulli p parameters to 1/numStates
                temp_params["E"][state_j, 1] = 62.5*(1+np.random.uniform()) # initializing all Exponential beta parameters to 62.5

        for lineage_num in range(numLineages): # for each lineage in our population
            paramlist.append(temp_params.copy()) # create a new dictionary holding the parameters and append it
            assert len(paramlist) == lineage_num+1

        return paramlist

#####------------------ Marginal State Distribution -----------------------######

    def get_Marginal_State_Distributions(self):
        '''
        Marginal State Distribution (MSD) matrix and recursion.
        This is the probability that a hidden state variable z_n is of
        state k, that is, each value in the [N x K] MSD array for each lineage is
        the probability

        P(z_n = k),

        for all z_n in the hidden state tree
        and for all k in the total number of discrete states. Each MSD array is
        an [N x K] array (an entry for each cell and an entry for each state),
        and each lineage has its own MSD array.

        Every element in MSD matrix is essentially sum over all transitions from any state to
        state j (from parent to daughter):
            P(z_u = k) = sum_on_all_j(T_jk * P(parent_cell_u) = j)

        '''
        numStates = self.numStates
        numLineages = self.numLineages
        population = self.population
        paramlist = self.paramlist

        MSD = []

        for num in range(numLineages): # for each lineage in our Population
            lineage = population[num] # getting the lineage in the Population by lineage index
            params = paramlist[num] # getting the respective params by lineage index
            MSD_array = np.zeros((len(lineage), numStates), dtype=float) # instantiating N by K array
            for state_k in range(numStates):
                MSD_array[0, state_k] = params["pi"][state_k]
            MSD.append(MSD_array)

        # Making sure the sum of initial probabilities over all states ==1
        for num in range(numLineages):
            MSD_0_row_sum = np.sum(MSD[num][0])
            assert np.isclose(MSD_0_row_sum, 1.), "The Marginal State Distribution for your root cells, P(z_1 = k), for all states k in numStates, are not adding up to 1!"


        for num in range(numLineages):
            lineage = population[num] # getting the lineage in the Population by lineage index
            curr_level = 2 # because we already have filled the first row of the matrix (level1)
            max_level = max_gen(lineage)
            while curr_level <= max_level:
                level = get_gen(curr_level, lineage) #get lineage for the generation
                for cell in level:
                    parent_cell_idx = lineage.index(cell.parent) # get the index of the parent cell
                    current_cell_idx = lineage.index(cell)
                    for state_k in range(numStates): # recursion based on parent cell
                        temp_sum_holder = [] # for all states k, calculate the sum of temp

                        for state_j in range(numStates): # for all states j, calculate temp
                            temp = params["T"][state_j, state_k] * MSD[num][parent_cell_idx, state_j]
                            # temp = T_jk * P(z_parent(n) = j)
                            temp_sum_holder.append(temp)

                        MSD[num][current_cell_idx, state_k] = sum(temp_sum_holder)
                curr_level += 1
            MSD_row_sums = np.sum(MSD[num], axis=1)
            assert np.allclose(MSD_row_sums, 1.0), "The Marginal State Distribution for your cells, P(z_k = k), for all states k in numStates, are not adding up to 1!"
        return MSD

#####--------------Calculating the Emission likelihood ----------------------#####
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
                E_param_k = E_param_array[state_k, :] # get the emission parameters for that state
                k_bern = E_param_k[0] # bernoulli rate parameter
                k_gomp_c = 0
                k_gomp_s = 0
                k_expon_beta = 0
                if self.FOM == 'G':
                    k_gomp_c = E_param_k[1]
                    k_gomp_s = E_param_k[2]
                elif self.FOM == 'E':
                    k_expon_beta = E_param_k[1]

                for cell in lineage: # for each cell in the lineage
                    current_cell_idx = lineage.index(cell) # get the index of the current cell
                    if self.FOM == 'G':
                        temp_b = 1
                        if not cell.isUnfinished(): # only calculate real bernoulli likelihood for finished cells
                            temp_b = sp.bernoulli.pmf(k=cell.fate, p=k_bern)
                        if cell.fateObserved:
                            assert cell.fateObserved
                            temp_g = right_censored_Gomp_pdf(tau_or_tauFake=cell.tau, c=k_gomp_c, scale=k_gomp_s, fateObserved=True) # gompertz likelihood if death is observed
                            assert np.isfinite(temp_g), "You have a Gompertz right-censored likelihood calculation for an observed death returning NaN. Your parameter estimates are likely creating overflow in the likelihood calculations."
                        elif not cell.fateObserved:
                            assert not cell.fateObserved
                            temp_g = right_censored_Gomp_pdf(tau_or_tauFake=cell.tauFake, c=k_gomp_c, scale=k_gomp_s, fateObserved=False) # gompertz likelihood if death is unobserved
                            assert np.isfinite(temp_g), "You have a Gompertz right-censored likelihood calculation for an unobserved death returning NaN. Your parameter estimates are likely creating overflow in the likelihood calculations."
                        EL_array[current_cell_idx, state_k] = temp_g*temp_b

                    elif self.FOM == 'E':
                        temp_b = 1
                        if not cell.isUnfinished():
                            temp_b = sp.bernoulli.pmf(k=cell.fate, p=k_bern) # bernoulli likelihood
                        if cell.fateObserved:
                            temp_beta = sp.expon.pdf(x=cell.tau, scale=k_expon_beta) # exponential likelihood
                        elif not cell.fateObserved:
                            temp_beta = sp.expon.pdf(x=cell.tauFake, scale=k_expon_beta) # exponential likelihood is the same in the cased of an unobserved death
                        assert np.isfinite(temp_beta), "You have a Exponential likelihood calculation returning NaN. Your parameter estimates are likely creating overflow in the likelihood calculations."
                        # the right-censored and uncensored exponential pdfs are the same
                        EL_array[current_cell_idx, state_k] = temp_beta*temp_b
            EL.append(EL_array) # append the EL_array for each lineage
        return EL

def right_censored_Gomp_pdf(tau_or_tauFake, c, scale, deathObserved=True):
    '''
    Gives you the likelihood of a right-censored Gompertz distribution.
    See Pg. 14 of The Gompertz distribution and Maximum Likelihood Estimation of its parameters - a revision
    by Adam Lenart
    November 28, 2011

    This is a replacement for scipy.gompertz function, because at the end of
    the experiment time, there will be some cells that are still alive and
    have not died or divided, so we don't know their end_time, these cells in 
    our data are called right censored. So this distribution is used instead of 
    real gompretz distribution, to make the synthesized data more like the distribution.


    p(tau_i | a, b) = [a * exp(b * tau) ^delta_i] * [exp(-a/b * (exp(b * tau_i -1)))]
    here, `firstCoeff` is [a * exp(b * tau) ^delta_i]  and the `secondCoeff` is [exp(-a/b * (exp(b * tau_i -1)))]

    Args:
        ----------
        tau_or_tauFake (float): the cell's lifetime
        c (float): loc of Gompertz (one of the distribution parameters)
        scale (float): scale parameter of Gompertz disribution
        deathObserved (bool): if the cell has died already, it is True, otherwise 
        it is False
        
    Return:
        ----------
        result (float): the multiplication of two coefficients 
            
    '''
    b = 1. / scale 
    a = c * b

    firstCoeff = a * np.exp(b*tau_or_tauFake)
    if deathObserved:
        pass # this calculation stays as is if the death is observed (delta_i = 1)
    else:
        firstCoeff = 1. # this calculation is raised to the power of delta if the death is unobserved (right-censored) (delta_i = 0)

    secondCoeff = np.exp((-1*a/b)*((np.exp(b*tau_or_tauFake))-1))
    # the observation of the cell death has no bearing on the calculation of the second coefficient in the pdf

    result = firstCoeff*secondCoeff
    assert np.isfinite(result), "Your Gompertz right-censored likelihood calculation is returning NaN. Your parameter estimates are likely creating overflow in the likelihood calculations."

    return result
