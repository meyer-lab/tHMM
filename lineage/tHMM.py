""" This file holds the parameters of our tHMM in the tHMM class. """

import numpy as np
from .StateDistribution import tHMM_E_init


class estimate:
    def __init__(self, numStates):
        self.numStates = numStates
        self.pi = np.ones((numStates)) / numStates
        self.T = np.ones((numStates, numStates)) / numStates
        self.E = []
        for state in range(self.numStates):
            self.E.append(tHMM_E_init(state))


class tHMM:
    """ Main tHMM class. """

    def __init__(self, X, numStates=1):
        """ Instantiates a tHMM.

        This function uses the following functions and instantials the tHMM with the requirements, such as the population of cells, the number of states, the initial estimates of the parameters, marginal state distribution, and emission likelihood.

        Args:
        -----
        X {obj}: The lineageTree object with its instances and properties.
        numStates {int}: the number of hidden states that we want our model to have.
        """
        self.X = X  # list containing lineages, should be in correct format (contain no NaNs)
        self.numStates = numStates  # number of discrete hidden states
        self.estimate = estimate(self.numStates)
        # full Marginal State Distribution holder
        self.MSD = self.get_Marginal_State_Distributions()
        self.EL = self.get_Emission_Likelihoods()  # full Emission Likelihood holder


##---------------------------- Marginal State Distribution ------------------------------##

    def get_Marginal_State_Distributions(self):
        """
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
        """
        MSD = []

        for num, lineageObj in enumerate(
                self.X):  # for each lineage in our Population
            # getting the lineage in the Population by lineage index
            lineage = lineageObj.output_lineage

            # instantiating N by K array
            MSD_array = np.zeros(
                (len(lineage), self.numStates), dtype=float)
            MSD_array[0, :] = self.estimate.pi
            MSD.append(MSD_array)

        for num, lineageObj in enumerate(
                self.X):  # for each lineage in our Population
            MSD_0_row_sum = np.sum(MSD[num][0])
            assert np.isclose(
                MSD_0_row_sum, 1.), "The Marginal State Distribution for your root cells, P(z_1 = k), for all states k in numStates, are not adding up to 1!"

        for num, lineageObj in enumerate(
                self.X):  # for each lineage in our Population
            # getting the lineage in the Population by lineage index
            lineage = lineageObj.output_lineage

            for level in lineageObj.output_list_of_gens[2:]:
                for cell in level:
                    parent_cell_idx = lineage.index(
                        cell.parent)  # get the index of the parent cell
                    current_cell_idx = lineage.index(cell)
                    for state_k in range(
                            self.numStates):  # recursion based on parent cell
                        temp_sum_holder = 0  # for all states k, calculate the sum of temp

                        for state_j in range(
                                self.numStates):  # for all states j, calculate temp
                            temp_sum_holder += self.estimate.T[state_j,
                                                               state_k] * MSD[num][parent_cell_idx, state_j]

                        MSD[num][current_cell_idx, state_k] = temp_sum_holder

            MSD_row_sums = np.sum(MSD[num], axis=1)

            assert np.allclose(
                MSD_row_sums, 1.0), "The Marginal State Distribution for your cells, P(z_k = k), for all states k in numStates, are not adding up to 1!"
        return MSD


##--------------------------- Emission Likelihood --------------------------------##


    def get_Emission_Likelihoods(self):
        """
        Emission Likelihood (EL) matrix.

        Each element in this N by K matrix represents the probability

        P(x_n = x | z_n = k),

        for all x_n and z_n in our observed and hidden state tree
        and for all possible discrete states k.
        """
        numStates = self.numStates

        EL = []

        for lineageObj in self.X:  # for each lineage in our Population
            # getting the lineage in the Population by lineage index
            lineage = lineageObj.output_lineage
            # instantiating N by K array for each lineage
            EL_array = np.zeros((len(lineage), numStates))

            for state_k in range(numStates):  # for each state
                for cell in lineage:  # for each cell in the lineage
                    # get the index of the current cell
                    current_cell_idx = lineage.index(cell)
                    EL_array[current_cell_idx,
                             state_k] = self.estimate.E[state_k].pdf(cell.obs)

            EL.append(EL_array)  # append the EL_array for each lineage
        return EL
