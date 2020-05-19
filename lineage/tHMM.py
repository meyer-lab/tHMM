""" This file holds the parameters of our tHMM in the tHMM class. """

import numpy as np


class estimate:
    def __init__(self, X, num_states, fpi=None, fT=None, fE=None):
        self.fpi = fpi
        self.fT = fT
        self.fE = fE
        self.num_states = num_states
        self.pi = np.squeeze(np.random.dirichlet(np.random.rand(num_states), 1).T)
        if self.fpi is not None:
            self.pi = self.fpi
        self.T = np.random.dirichlet(np.random.rand(num_states), num_states)
        if self.fT is not None:
            self.T = self.fT
        self.E = []
        for _ in range(self.num_states):
            self.E.append(X[0].E[0].tHMM_E_init())
        if self.fE is not None:
            self.E = self.fE


class tHMM:
    """ Main tHMM class. """

    def __init__(self, X, num_states, fpi=None, fT=None, fE=None):
        """ Instantiates a tHMM.

        This function uses the following functions and assings them to the cells
        (objects) in the lineage.

        Args:
            ----------
            X (list of objects): A list of objects (cells) in a lineage in which
            the NaNs have been removed.
            num_states (int): the number of hidden states that we want our model have
        """
        self.fpi = fpi
        self.fT = fT
        self.fE = fE
        self.X = X  # list containing lineages, should be in correct format (contain no NaNs)
        self.num_states = num_states  # number of discrete hidden states
        self.estimate = estimate(self.X, self.num_states, fpi=self.fpi, fT=self.fT, fE=self.fE)
        self.MSD = self.get_Marginal_State_Distributions()  # full Marginal State Distribution holder
        self.EL = self.get_Emission_Likelihoods()  # full Emission Likelihood holder

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

        for num, lineageObj in enumerate(self.X):  # for each lineage in our Population
            lineage = lineageObj.output_lineage  # getting the lineage in the Population by lineage index

            MSD_array = np.zeros((len(lineage), self.num_states))  # instantiating N by K array
            MSD_array[0, :] = self.estimate.pi

            np.isclose(np.sum(MSD_array[0]), 1.0)
            MSD.append(MSD_array)

        for num, lineageObj in enumerate(self.X):  # for each lineage in our Population
            lineage = lineageObj.output_lineage  # getting the lineage in the Population by lineage index

            for level in lineageObj.output_list_of_gens[2:]:
                for cell in level:
                    parent_cell_idx = lineage.index(cell.parent)  # get the index of the parent cell
                    current_cell_idx = lineage.index(cell)

                    # recursion based on parent cell
                    MSD[num][current_cell_idx, :] = np.matmul(MSD[num][parent_cell_idx, :], self.estimate.T)

            assert np.allclose(np.sum(MSD[num], axis=1), 1.0)
        return MSD

    def get_Emission_Likelihoods(self):
        """
        Emission Likelihood (EL) matrix.

        Each element in this N by K matrix represents the probability

        P(x_n = x | z_n = k),

        for all x_n and z_n in our observed and hidden state tree
        and for all possible discrete states k.
        """
        EL = []

        for lineageObj in self.X:  # for each lineage in our Population
            lineage = lineageObj.output_lineage  # getting the lineage in the Population by lineage index
            EL_array = np.zeros((len(lineage), self.num_states))  # instantiating N by K array for each lineage

            for current_cell_idx, cell in enumerate(lineage):  # for each cell in the lineage
                for state_k in range(self.num_states):  # for each state
                    EL_array[current_cell_idx, state_k] = self.estimate.E[state_k].pdf(cell.obs)

            EL.append(EL_array)  # append the EL_array for each lineage
        return EL
