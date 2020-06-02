""" This file holds the parameters of our tHMM in the tHMM class. """

import numpy as np
from .BaumWelch import do_E_step, do_M_step


class estimate:
    def __init__(self, X, nState: int, fpi=None, fT=None, fE=None):
        self.fpi = fpi
        self.fT = fT
        self.fE = fE
        self.num_states = nState

        if self.fpi is None:
            self.pi = np.random.rand(nState)
            self.pi /= np.sum(self.pi)
        else:
            self.pi = self.fpi

        if self.fT is None:
            self.T = np.random.dirichlet(np.random.rand(nState), nState)
        else:
            self.T = self.fT

        if self.fE is None:
            self.E = [X[0].E[0].__class__() for _ in range(self.num_states)]
        else:
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

    def fit(self, tolerance=np.spacing(1), max_iter=100):
        """Runs the tHMM function through Baum Welch fitting"""

        # first E step
        MSD, EL, NF, betas, gammas = do_E_step(self)
        new_LL = calculate_log_likelihood(NF)

        # first stopping condition check
        for iter_number in range(max_iter):
            old_LL = new_LL

            est_pi, est_T, est_E = do_M_step(self, MSD, betas, gammas)
            MSD, EL, NF, betas, gammas = do_E_step(self)
            new_LL = calculate_log_likelihood(NF)

            diff = np.linalg.norm(old_LL - new_LL)

            if diff < tolerance:
                break

        return self, MSD, EL, NF, betas, gammas, new_LL


    