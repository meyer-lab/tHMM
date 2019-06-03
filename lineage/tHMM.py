""" This file holds the parameters of our tHMM in the tHMM class. """

import numpy as np
import scipy.stats as sp
from .Lineage_utils import get_numLineages, init_Population
from .tHMM_utils import max_gen, get_gen


class tHMM:
    """ Main tHMM class. """

    def __init__(self, X, numStates=1, FOM='E'):
        """ Instantiates a tHMM.

        This function uses the following functions and assings them to the cells
        (objects) in the lineage.

        Args:
            ----------
            X (list of objects): A list of objects (cells) in a lineage in which
            the NaNs have been removed.
            numStates (int): the number of hidden states that we want our model have
            FOM (str): For now, it is either "E": Exponential, or "G": Gompertz
            and it determines the type of distribution for lifetime of the cells
        """
        self.X = X  # list containing lineage, should be in correct format (contain no NaNs)
        self.numStates = numStates  # number of discrete hidden states
        self.FOM = FOM
        self.numLineages = get_numLineages(self.X)  # gets the number of lineages in our population
        self.population = init_Population(self.X, self.numLineages)  # arranges the population into a list of lineages (each lineage might have varying length)
        assert self.numLineages == len(
            self.population), "Something is wrong with the number of lineages in your population member variable for your tHMM class and the number of lineages member variable for your tHMM class. Check the number of unique root cells and the number of lineages in your data."
        self.paramlist = self.init_paramlist()  # list that is numLineages long of parameters for each lineage tree in our population

        self.MSD = self.get_Marginal_State_Distributions()  # full Marginal State Distribution holder
        self.EL = self.get_Emission_Likelihoods()  # full Emission Likelihood holder


##------------------------ Initializing the parameter list --------------------------##


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
                2.1. Exponential distribution with 1 parameter (beta)
                    beta intialized to 62.5

            If the Exponential is used, on the whole we will have 2 parameters for emissions, so
            the emission matrix will be [K x 2].

        Returns:
            ----------
            paramlist (dictionary): a dictionary holding three matrices mentioned above.

        '''
        paramlist = []
        numStates = self.numStates
        numLineages = self.numLineages
        temp_params = {"pi": np.ones((numStates)) / numStates,  # inital state distributions [K] initialized to 1/K
                       "T": np.ones((numStates, numStates)) / numStates}  # state transition matrix [KxK] initialized to 1/K
        if self.FOM == 'E':
            temp_params["E"] = np.ones((numStates, 2))  # sequence of emission likelihood distribution parameters [Kx2]
            for state_j in range(numStates):
                temp_params["E"][state_j, 0] = 0.5  # initializing all Bernoulli p parameters to 1/numStates
                temp_params["E"][state_j, 1] = 62.5 * (1 + np.random.uniform())  # initializing all Exponential beta parameters to 62.5
        elif self.FOM == 'Ga':
            temp_params["E"] = np.ones((numStates, 3))
            for state_j in range(numStates):
                temp_params["E"][state_j, 0] = 1 / numStates  # initializing all Bernoulli p parameters to 1/numStates
                temp_params["E"][state_j, 1] = 10 * (1 + np.random.uniform())  # Gamma shape parameter
                temp_params["E"][state_j, 2] = 5 * (1 + np.random.uniform())  # Gamma scale parameter
 
        for lineage_num in range(numLineages):  # for each lineage in our population
            paramlist.append(temp_params.copy())  # create a new dictionary holding the parameters and append it
            assert len(paramlist) == lineage_num + 1, "The number of parameters being estimated is mismatched with the number of lineages in your population. Check the number of unique root cells and the number of lineages in your data."

        return paramlist

##---------------------------- Marginal State Distribution ------------------------------##
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

        Every element in MSD matrix is essentially sum over all transitions from any state to
        state j (from parent to daughter):
            P(z_u = k) = sum_on_all_j(Transition(from j to k) * P(parent_cell_u) = j)
        '''
        numStates = self.numStates
        numLineages = self.numLineages
        population = self.population
        paramlist = self.paramlist

        MSD = []

        for num in range(numLineages):  # for each lineage in our Population
            lineage = population[num]  # getting the lineage in the Population by lineage index
            params = paramlist[num]  # getting the respective params by lineage index
            MSD_array = np.zeros((len(lineage), numStates), dtype=float)  # instantiating N by K array
            for state_k in range(numStates):
                MSD_array[0, state_k] = params["pi"][state_k]
            MSD.append(MSD_array)

        for num in range(numLineages):
            MSD_0_row_sum = np.sum(MSD[num][0])
            assert np.isclose(MSD_0_row_sum, 1.), "The Marginal State Distribution for your root cells, P(z_1 = k), for all states k in numStates, are not adding up to 1!"

        for num in range(numLineages):
            lineage = population[num]  # getting the lineage in the Population by lineage index
            curr_level = 2
            max_level = max_gen(lineage)
            while curr_level <= max_level:
                level = get_gen(curr_level, lineage)  # get lineage for the gen
                for cell in level:
                    parent_cell_idx = lineage.index(cell.parent)  # get the index of the parent cell
                    current_cell_idx = lineage.index(cell)
                    for state_k in range(numStates):  # recursion based on parent cell
                        temp_sum_holder = []  # for all states k, calculate the sum of temp

                        for state_j in range(numStates):  # for all states j, calculate temp
                            temp = params["T"][state_j, state_k] * MSD[num][parent_cell_idx, state_j]
                            # temp = T_jk * P(z_parent(n) = j)
                            temp_sum_holder.append(temp)

                        MSD[num][current_cell_idx, state_k] = sum(temp_sum_holder)
                curr_level += 1
            MSD_row_sums = np.sum(MSD[num], axis=1)
            assert np.allclose(MSD_row_sums, 1.0), "The Marginal State Distribution for your cells, P(z_k = k), for all states k in numStates, are not adding up to 1!"
        return MSD

##--------------------------- Emission Likelihood --------------------------------##
    def get_Emission_Likelihoods(self):
        '''
        Emission Likelihood (EL) matrix.

        Each element in this N by K matrix represents the probability

        P(x_n = x | z_n = k),

        for all x_n and z_n in our observed and hidden state tree
        and for all possible discrete states k. Since we have a
        multiple observation model, that is

        x_n = {x_B, x_E}, # in case of Exponential distribution for lifetime

        consists of more than one observation, x_B = division(1) or
        death(0) (which is one of the observations x_B) and the other
        being,x_E = lifetime, lifetime >=0, (which is the other observation x_E)
        we make the assumption that

        In case of Exponential lifetime in stead of Gompertz, we have:
        P(x_n = x | z_n = k) = P(x_n1 = x_B | z_n = k) * P(x_n = x_E | z_n = k).

        '''
        numStates = self.numStates
        numLineages = self.numLineages
        population = self.population
        paramlist = self.paramlist

        EL = []

        for num in range(numLineages):  # for each lineage in our Population
            lineage = population[num]  # getting the lineage in the Population by lineage index
            params = paramlist[num]  # getting the respective params by lineage index
            EL_array = np.zeros((len(lineage), numStates))  # instantiating N by K array for each lineage
            E_param_array = params["E"]  # K by 3 array of distribution parameters for each lineage

            for state_k in range(numStates):  # for each state
                E_param_k = E_param_array[state_k, :]  # get the emission parameters for that state
                k_bern = E_param_k[0]  # bernoulli rate parameter
                k_expon_beta = 0
                k_shape_gamma = 0
                k_scale_gamma = 0

                if self.FOM == 'E':
                    k_expon_beta = E_param_k[1]
                elif self.FOM == 'Ga':
                    k_shape_gamma = E_param_k[1]
                    k_scale_gamma = E_param_k[2]

                for cell in lineage:  # for each cell in the lineage
                    current_cell_idx = lineage.index(cell)  # get the index of the current cell
                    if self.FOM == 'E':
                        temp_b = sp.bernoulli.pmf(k=cell.fate, p=k_bern)  # bernoulli likelihood
                        if cell.fateObserved:
                            temp_beta = sp.expon.pdf(x=cell.tau, scale=k_expon_beta)  # exponential likelihood
                        elif not cell.fateObserved:
                            temp_beta = sp.expon.pdf(x=cell.tauFake, scale=k_expon_beta)  # exponential likelihood is the same in the cased of an unobserved death
                        assert np.isfinite(temp_beta), "You have a Exponential likelihood calculation returning NaN. Your parameter estimates are likely creating overflow in the likelihood calculations."
                        # the right-censored and uncensored exponential pdfs are the same
                        EL_array[current_cell_idx, state_k] = temp_beta * temp_b
                    if self.FOM == 'Ga':
                        temp_b = sp.bernoulli.pmf(k=cell.fate, p=k_bern)  # bernoulli likelihood
                        if cell.fateObserved:
                            temp_g = sp.gamma.pdf(x=cell.tau, a=k_gamma_shape, scale=k_gamma_scale)
                        assert np.isfinite(temp_g),"Gamma likelihood is returning NaN"
                        EL_array[current_cell_idx, state_k] = temp_g * temp_b
            EL.append(EL_array)  # append the EL_array for each lineage
        return EL
