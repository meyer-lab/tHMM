""" This file holds the parameters of our tHMM in the tHMM class. """

from copy import deepcopy
import numpy as np
from typing import Tuple
import scipy.stats as sp

from .UpwardRecursion import get_Emission_Likelihoods
from .BaumWelch import do_E_step, calculate_log_likelihood, do_M_step, do_M_E_step, do_M_E_step_atonce
from .Viterbi import get_leaf_deltas, get_nonleaf_deltas, Viterbi


class estimate:
    """Estimation class.
    """

    def __init__(self, X: list, nState: int, fpi=None, fT=None, fE=None):
        """
        Instantiating the estimation class.
        The initial probability array (pi), transition probability matrix (T), and the emission likelihood (E) are initialized.
        If these parameters are already estimated, then they are assigned as an instance to the estimate class.

        :param X: A list of objects (cells) in one lineage
        :param nStates: The number of hidden states
        """
        self.fpi = fpi
        self.fT = fT
        self.fE = fE
        self.num_states = nState

        if self.fpi is None or self.fpi is True:
            self.pi = np.random.rand(nState)
            self.pi /= np.sum(self.pi)
        else:
            self.pi = self.fpi

        if self.fT is None:
            self.T = np.random.dirichlet(np.random.rand(nState), nState)
        else:
            self.T = self.fT

        if self.fE is None:
            self.E = [deepcopy(X[0].E[0]) for _ in range(self.num_states)]
        else:
            self.E = self.fE


class tHMM:
    """Main tHMM class.
    """

    def __init__(self, X: list, num_states: int, fpi=None, fT=None, fE=None):
        """Instantiates a tHMM.
        This function uses the following functions and assings them to the cells
        (objects) in the lineage.

        :param X: A list of objects (cells) in one lineage
        :param nStates: The number of hidden states
        """
        self.fpi = fpi
        self.fT = fT
        self.fE = fE
        self.X = X  # list containing lineages
        self.num_states = num_states  # number of discrete hidden states, should be integral
        self.estimate = estimate(
            self.X, self.num_states, fpi=self.fpi, fT=self.fT, fE=self.fE)
        self.EL = get_Emission_Likelihoods(self)

    def fit(self, tolerance=1e-9, max_iter=1000):
        """
        Runs the tHMM function through Baum Welch fitting.

        :param tolerance: the tolerance for change of likelihood between two steps
        :param max_iter: the maximum number of iterations for fitting
        """
        MSD_list, NF_list, betas_list, gammas_list, new_LL = fit_list([self], tolerance=tolerance, max_iter=max_iter)

        return self, MSD_list[0], NF_list[0], betas_list[0], gammas_list[0], new_LL

    def predict(self) -> list:
        """
        Given a fit model, the model predicts an optimal
        state assignment using the Viterbi algorithm.

        :return: assigned states to each cell in each lineage. It is organized in the form of list of arrays, each array shows the state of cells in one lineage.
        """
        deltas, state_ptrs = get_leaf_deltas(self)
        get_nonleaf_deltas(self, deltas, state_ptrs)
        pred_states_by_lineage = Viterbi(self, deltas, state_ptrs)
        return pred_states_by_lineage

    def get_BIC(self, LL: float, num_cells: int, atonce=False) -> Tuple[float, float]:
        """
        Gets the BIC values. Akaike Information Criterion, used for model selection and deals with the trade off
        between over-fitting and under-fitting.
        :math:`BIC = - 2 * log(LL) + log(number_of_cells) * DoF` in which k is the number of free parameters and LL is the maximum of likelihood function.
        Minimum of BIC detremines the relatively better model.
        """
        degrees_of_freedom = 0.0
        # initial prob. matrix
        if self.fpi is None:
            degrees_of_freedom += self.num_states - 1

        # transition matrix
        if self.fT is None:
            degrees_of_freedom += self.num_states * (self.num_states - 1)

        if atonce:  # assuming we are fitting all 4 concentrations at once and we have cell cycle phase specific distributions.
            degrees_of_freedom += self.num_states * 4.5
        else:
            for ii in range(self.num_states):
                degrees_of_freedom += self.estimate.E[ii].dof()

        # the whole population has one BIC value.
        BIC_value = -2 * np.sum(LL) + np.log(num_cells) * degrees_of_freedom

        return BIC_value, degrees_of_freedom


def fit_list(tHMMobj_list: list, tolerance: float = 1e-9, max_iter: int = 1000) -> Tuple[list, list, list, list, float]:
    """
    Runs the tHMM function through Baum Welch fitting for a list containing a set of data for different concentrations.

    :param tHMMobj_list: all lineage trees we want to fit at once
    :param tolerance: the stopping criteria for fitting. when the likelihood does not change more than tolerance from one step to the other, the fitting converges.
    :param max_iter: the maximum number of iterations for fitting
    :return MSD_list: marginal state distributions for all populations we fit at once
    :return NF: normalizing factor
    :return betas: beta values (conditional probability of cell states given cell observations)
    :return gammas: gamma values (used to calculate the downward reursion)
    :return new_LL: the log-likelihood of the optimized solution
    """

    # Step 0: initialize with random assignments and do an M step
    # when there are no fixed emissions, we need to randomize the start
    init_all_gammas = [[sp.multinomial.rvs(n=1, p=[1. / tHMMobj.num_states] * tHMMobj.num_states, size=len(lineage))
                        for lineage in tHMMobj.X] for tHMMobj in tHMMobj_list]

    if len(tHMMobj_list) > 1:  # it means we are fitting several concentrations at once.
        do_M_E_step_atonce(tHMMobj_list, init_all_gammas)
    else:  # means we are fitting one condition at a time.
        do_M_E_step(tHMMobj_list[0], init_all_gammas[0])

    # Step 1: first E step
    MSD_list, NF_list, betas_list, gammas_list = map(list, zip(*[do_E_step(tHMM) for tHMM in tHMMobj_list]))
    old_LL = np.sum([np.sum(calculate_log_likelihood(NF)) for NF in NF_list])

    # first stopping condition check
    for _ in range(max_iter):
        do_M_step(tHMMobj_list, MSD_list, betas_list, gammas_list)
        MSD_list, NF_list, betas_list, gammas_list = map(list, zip(*[do_E_step(tHMM) for tHMM in tHMMobj_list]))
        new_LL = np.sum([np.sum(calculate_log_likelihood(NF)) for NF in NF_list])
        if new_LL - old_LL < tolerance:
            break

        old_LL = new_LL

    return MSD_list, NF_list, betas_list, gammas_list, new_LL
