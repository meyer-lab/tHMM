""" This file holds the parameters of our tHMM in the tHMM class. """

from copy import deepcopy
import numpy as np
from typing import Tuple
from .Viterbi import Viterbi
from .LineageTree import LineageTree, get_Emission_Likelihoods


class estimate:
    """Estimation class."""

    def __init__(
        self, X: list[LineageTree], nState: int, fpi=None, fT=None, fE=None, rng=None
    ):
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
        rng = np.random.default_rng(rng)

        if self.fpi is None or self.fpi is True:
            self.pi = rng.dirichlet(np.ones(nState))
        else:
            self.pi = self.fpi

        if self.fT is None:
            self.T = rng.dirichlet(np.ones(nState), size=nState)
        else:
            self.T = self.fT

        if self.fE is None:
            self.E = [deepcopy(X[0].E[0]) for _ in range(self.num_states)]
        else:
            self.E = self.fE


class tHMM:
    """Main tHMM class."""

    def __init__(
        self,
        X: list[LineageTree],
        num_states: int,
        fpi=None,
        fT=None,
        fE=None,
        rng=None,
    ):
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
        self.num_states = (
            num_states  # number of discrete hidden states, should be integral
        )
        self.estimate = estimate(
            self.X, self.num_states, fpi=self.fpi, fT=self.fT, fE=self.fE, rng=rng
        )

    def predict(self) -> list:
        """
        Given a fit model, the model predicts an optimal
        state assignment using the Viterbi algorithm.

        :return: assigned states to each cell in each lineage. It is organized in the form of list of arrays, each array shows the state of cells in one lineage.
        """
        return Viterbi(self)

    def get_BIC(
        self, LL: float, num_cells: int, atonce: bool=False, mcf10a: bool=False
    ) -> Tuple[float, float]:
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

        if (
            atonce
        ):  # assuming we are fitting all 4 concentrations at once and we have cell cycle phase specific distributions.
            if mcf10a:
                degrees_of_freedom += self.num_states * 2.25
            else:
                degrees_of_freedom += self.num_states * 4.5
        else:
            for ii in range(self.num_states):
                degrees_of_freedom += self.estimate.E[ii].dof()

        # the whole population has one BIC value.
        BIC_value = -2 * np.sum(LL) + np.log(num_cells) * degrees_of_freedom

        return BIC_value, degrees_of_freedom

    def log_score(self, X_state_tree_sequence: list, pi=None, T=None, E=None) -> list[float]:
        """
        This function returns the log-likelihood of a possible state assignment
        given the estimated model parameters.
        The user can also provide their own pi, T, or E matrices instead to score
        a possible state assignment.
        :math:`P(x_1,...,x_N,z_1,...,z_N) = P(z_1) * prod_{n=2:N}(P(z_n | z_pn)) * prod_{n=1:N}(P(x_n|z_n))`
        :math:`log{P(x_1,...,x_N,z_1,...,z_N)} = log{P(z_1)} + sum_{n=2:N}(log{P(z_n | z_pn)}) + sum_{n=1:N}(log{P(x_n|z_n)})`
        :param X_state_tree_sequence: the assigned states to cells at each lineage object
        :return: the log-likelihood of states assigned to single cells, based on the pi, T, and E, separate for each lineage tree
        """
        if pi is None:
            pi = self.estimate.pi
        if T is None:
            T = self.estimate.T
        if E is None:
            E = self.estimate.E

        log_scores = []
        for idx, lineageObj in enumerate(self.X):
            state_tree_sequence = X_state_tree_sequence[idx]

            # the first term is the value of pi for the state of the first cell
            log_score = np.log(pi[state_tree_sequence[0]])
            log_score += log_T_score(T, state_tree_sequence, lineageObj)

            # Calculate the joint probability of state and observations
            log_EL_array = np.log(get_Emission_Likelihoods(self.X, E)[idx])
            log_score += np.sum(
                log_EL_array[np.arange(log_EL_array.shape[0]), state_tree_sequence]
            )

            assert np.all(np.isfinite(log_score))
            log_scores.append(float(log_score))

        return log_scores


def log_T_score(
    T: np.ndarray, state_tree_sequence: list[np.ndarray], lineageObj: LineageTree
) -> float:
    """
    To calculate the joint probability of state and observations.
    This function calculates the second term.
    :math:`P(x_1,...,x_N,z_1,...,z_N) = P(z_1) * prod_{n=2:N}(P(z_n | z_pn)) * prod_{n=1:N}(P(x_n|z_n))`
    :math:`log{P(x_1,...,x_N,z_1,...,z_N)} = log{P(z_1)} + sum_{n=2:N}(log{P(z_n | z_pn)}) + sum_{n=1:N}(log{P(x_n|z_n)})`
    :param T: transition probability matrix
    :type T: ndarray
    :param state_tree_sequence: the assigned states to cells at each lineage object
    :param lineageObj: the lineage trees
    :type lineageObj: object
    :return: the log-likelihood of the transition probability matrix
    """
    log_T_score_holder = 0.0
    log_T = np.log(T)

    # we start with the first transition, from the root cell
    for cIDX, cell in enumerate(lineageObj.output_lineage):
        if cell.gen > 0 and not cell.isLeaf():
            cell_state = state_tree_sequence[cIDX]
            for dIDX in [cIDX * 2 + 1, cIDX * 2 + 2]:
                daughter_state = state_tree_sequence[dIDX]
                log_T_score_holder += float(log_T[cell_state, daughter_state])

    return float(log_T_score_holder)
