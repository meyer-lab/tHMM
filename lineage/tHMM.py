""" This file holds the parameters of our tHMM in the tHMM class. """

from copy import deepcopy
import numpy as np
from typing import Tuple
from .Viterbi import Viterbi
from .LineageTree import LineageTree


class estimate:
    """Estimation class."""

    def __init__(self, X: list[LineageTree], nState: int, fpi=None, fT=None, fE=None):
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
    """Main tHMM class."""

    def __init__(self, X: list[LineageTree], num_states: int, fpi=None, fT=None, fE=None):
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
            self.X, self.num_states, fpi=self.fpi, fT=self.fT, fE=self.fE
        )

    def predict(self) -> list:
        """
        Given a fit model, the model predicts an optimal
        state assignment using the Viterbi algorithm.

        :return: assigned states to each cell in each lineage. It is organized in the form of list of arrays, each array shows the state of cells in one lineage.
        """
        return Viterbi(self)

    def get_BIC(
        self, LL: float, num_cells: int, atonce=False, mcf10a=False
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

    def get_Emission_Likelihoods(self):
        return get_Emission_Likelihoods(self)

    def log_score(self, X_state_tree_sequence: list, pi=None, T=None, E=None) -> list:
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
            log_EL_array = np.log(get_Emission_Likelihoods(self, E)[idx])
            log_score += np.sum(
                log_EL_array[np.arange(log_EL_array.shape[0]), state_tree_sequence]
            )

            assert np.all(np.isfinite(log_score))
            log_scores.append(log_score)

        return log_scores


def log_T_score(T: np.ndarray, state_tree_sequence: list, lineageObj: LineageTree) -> float:
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
    log_T_score_holder = 0
    log_T = np.log(T)
    # we start with the first transition, from the root cell
    for level in lineageObj.idx_by_gen:
        for cell_idx in level:
            cell = lineageObj.output_lineage[cell_idx]
            if not cell.isLeaf():
                cell_state = state_tree_sequence[cell_idx]
                for daughter in cell.get_daughters():
                    child_idx = lineageObj.output_lineage.index(daughter)
                    daughter_state = state_tree_sequence[child_idx]
                    log_T_score_holder += log_T[cell_state, daughter_state]
    return log_T_score_holder


def get_Emission_Likelihoods(tHMMobj: tHMM, E: list = None) -> list:
    """
    Emission Likelihood (EL) matrix.

    Each element in this N by K matrix represents the probability

    :math:`P(x_n = x | z_n = k)`,

    for all :math:`x_n` and :math:`z_n` in our observed and hidden state tree
    and for all possible discrete states k.
    :param tHMMobj: A class object with properties of the lineages of cells
    :param E: The emissions likelihood
    :return: The marginal state distribution
    """
    if E is None:
        E = tHMMobj.estimate.E

    all_cells = np.array([cell.obs for lineage in tHMMobj.X for cell in lineage.output_lineage])
    ELstack = np.zeros((len(all_cells), tHMMobj.num_states))

    for k in range(tHMMobj.num_states):  # for each state
        ELstack[:, k] = np.exp(E[k].logpdf(all_cells))
        assert np.all(np.isfinite(ELstack[:, k]))
    EL = []
    ii = 0
    for lineageObj in tHMMobj.X:  # for each lineage in our Population
        nl = len(lineageObj.output_lineage)  # getting the lineage length
        EL.append(ELstack[ii:(ii + nl), :])  # append the EL_array for each lineage

        ii += nl

    return EL
