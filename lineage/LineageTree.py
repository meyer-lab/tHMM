""" This file contains the LineageTree class. """
import numpy as np
import numpy.typing as npt
import operator
from .CellVar import CellVar


class LineageTree:
    """A class for lineage trees. This class also handles algorithms for walking
    the tree to calculate various properties.
    """

    pi: np.ndarray
    T: np.ndarray
    leaves_idx: np.ndarray
    output_list_of_gens: list

    def __init__(self, list_of_cells: list, E: list):
        self.E = E
        self.output_lineage = sorted(list_of_cells, key=operator.attrgetter("gen"))
        self.output_list_of_gens = max_gen(self.output_lineage)
        # assign times using the state distribution specific time model
        E[0].assign_times(self.output_list_of_gens)
        self.leaves_idx = get_leaves_idx(self.output_lineage)

    @classmethod
    def init_from_parameters(
        cls,
        pi: np.ndarray,
        T: np.ndarray,
        E: list,
        desired_num_cells: int,
        censor_condition=0,
        **kwargs,
    ):
        r"""
        Constructor method

        :param :math:`\pi`: The initial probability matrix; its shape must be the same as the number of states and all of them must sum up to 1.
        :param T: The transition probability matrix; every row must sum up to 1.
        :param E: A list containing state distribution objects, the length of it is the same as the number of states.
        :param desired_num_cells: The desired number of cells we want the lineage to end up with.
        :param censor_condition: An integer :math:`\in` \{0, 1, 2, 3\} that decides the type of censoring.

        Censoring guide
        - 0 means no pruning
        - 1 means censor based on the fate of the cell
        - 2 means censor based on the length of the experiment
        - 3 means censor based on both the 'fate' and 'time' conditions
        """
        assert pi.size == T.shape[0]
        assert T.shape[0] == T.shape[1]

        full_lineage = generate_lineage_list(
            pi=pi, T=T, desired_num_cells=desired_num_cells
        )
        for i_state in range(pi.size):
            output_assign_obs(i_state, full_lineage, E)

        full_list_of_gens = max_gen(full_lineage)

        # assign times using the state distribution specific time model
        E[0].assign_times(full_list_of_gens)

        output_lineage = E[0].censor_lineage(
            censor_condition, full_list_of_gens, full_lineage, **kwargs
        )

        lineageObj = cls(output_lineage, E)
        lineageObj.pi = pi
        lineageObj.T = T
        return lineageObj

    def get_parent_idxs(self, level: list) -> set[int]:
        """
        Given a list of cells, return the indices of the parents.
        :param level: The list of cells.
        :return parent_holder: Indices of the parents.
        """
        parent_holder = set()  # set makes sure only one index is put in and no overlap
        for cell in level:
            parent_holder.add(self.output_lineage.index(cell.parent))
        return parent_holder

    def get_Marginal_State_Distributions(
        self, pi: np.ndarray, T: np.ndarray
    ) -> np.ndarray:
        r"""Marginal State Distribution (MSD) matrix by upward recursion.
        This is the probability that a hidden state variable :math:`z_n` is of
        state k, that is, each value in the N by K MSD array for each lineage is
        the probability

        :math:`P(z_n = k)`,

        for all :math:`z_n` in the hidden state tree
        and for all k in the total number of discrete states. Each MSD array is
        an N by K array (an entry for each cell and an entry for each state),
        and each lineage has its own MSD array.

        Every element in MSD matrix is essentially sum over all transitions from any state to
        state j (from parent to daughter):

        :math:`P(z_u = k) = \sum_j(Transition(j -> k) * P(parent_{cell_u}) = j)`

        :param pi: Initial probabilities vector
        :param T: State transitions matrix
        :return: The marginal state distribution
        """
        m = np.zeros((len(self.output_lineage), pi.size))
        m[0, :] = pi

        for level in self.output_list_of_gens[2:]:
            for cell in level:
                pCellIDX = self.output_lineage.index(
                    cell.parent
                )  # get the index of the parent cell
                cCellIDX = self.output_lineage.index(cell)

                # recursion based on parent cell
                m[cCellIDX, :] = m[pCellIDX, :] @ T

        np.testing.assert_allclose(np.sum(m, axis=1), 1.0)
        return m

    def get_beta(
        self, T: np.ndarray, MSD: np.ndarray, EL: np.ndarray, NF: np.ndarray
    ) -> np.ndarray:
        r"""Beta matrix and base case at the leaves.

        Each element in this N by K matrix is the beta value
        for each cell and at each state. In particular, this
        value is derived from the Marginal State Distributions
        (MSD), the Emission Likelihoods (EL), and the
        Normalizing Factors (NF). Each beta value
        for the leaves is exactly the probability

        :math:`beta[n,k] = P(z_n = k | x_n = x)`.

        Using Bayes Theorem, we see that the above equals

        numerator = :math:`P(x_n = x | z_n = k) * P(z_n = k)`
        denominator = :math:`P(x_n = x)`
        :math:`beta[n,k] = numerator / denominator`

        For non-leaf cells, the first value in the numerator is the Emission
        Likelihoods. The second value in the numerator is
        the Marginal State Distributions. The value in the
        denominator is the Normalizing Factor.

        Traverses upward through each tree and calculates the
        beta value for each non-leaf cell. The normalizing factors (NFs)
        are also calculated as an intermediate for determining each
        beta term. Helper functions are called to determine one of
        the terms in the NF equation. This term is also used in the calculation
        of the betas.

        :param tHMMobj: A class object with properties of the lineages of cells
        :param MSD: The marginal state distribution P(z_n = k)
        :param EL: The emissions likelihood
        :param NF: normalizing factor. The marginal observation distribution P(x_n = x)
        :return: beta values. The conditional probability of states, given observations of the sub-tree rooted in cell_n
        """
        beta = np.zeros((len(self.output_lineage), MSD.shape[1]))

        # Emission Likelihood, Marginal State Distribution, Normalizing Factor (same regardless of state)
        # P(x_n = x | z_n = k), P(z_n = k), P(x_n = x)
        ii = self.leaves_idx
        beta[ii, :] = EL[ii, :] * MSD[ii, :] / NF[ii, np.newaxis]
        assert np.isclose(np.sum(beta[-1]), 1.0)

        lineage = self.output_lineage  # lineage in the population
        MSD_array = np.clip(
            MSD, np.finfo(float).eps, np.inf
        )  # MSD of the respective lineage
        ELMSD = EL * MSD

        for level in self.output_list_of_gens[2:][
            ::-1
        ]:  # a reversed list of generations
            for pii in self.get_parent_idxs(level):
                ch_ii = [lineage.index(d) for d in lineage[pii].get_daughters()]
                ratt = beta[ch_ii, :] / MSD_array[ch_ii, :]
                fac1 = np.prod(ratt @ T.T, axis=0) * ELMSD[pii, :]

                NF[pii] = sum(fac1)
                beta[pii, :] = fac1 / NF[pii]

        return beta

    def get_all_zetas(
        self,
        beta_array: np.ndarray,
        MSD_array: np.ndarray,
        gamma_array: np.ndarray,
        T: np.ndarray,
    ) -> np.ndarray:
        """
        Sum of the list of all the zeta parent child for all the parent cells for a given state transition pair.
        This is an inner component in calculating the overall transition probability matrix.

        :param lineageObj: the lineage tree of cells
        :param beta_array: beta values. The conditional probability of states, given observations of the sub-tree rooted in cell_n
        :param MSD_array: marginal state distribution
        :param gamma_array: gamma values. The conditional probability of states, given the observation of the whole tree
        :param T: transition probability matrix
        :return: numerator for calculating the transition probabilities
        """
        betaMSD = beta_array / np.clip(MSD_array, np.finfo(float).eps, np.inf)
        TbetaMSD = np.clip(betaMSD @ T.T, np.finfo(float).eps, np.inf)
        lineage = self.output_lineage
        holder = np.zeros(T.shape)

        for level in self.output_list_of_gens[1:]:
            for cell in level:  # get lineage for the generation
                gamma_parent = gamma_array[lineage.index(cell), :]  # x by j

                if not cell.isLeaf():
                    for daughter_idx in cell.get_daughters():
                        d_idx = lineage.index(daughter_idx)
                        js = gamma_parent / TbetaMSD[d_idx, :]
                        holder += np.outer(js, betaMSD[d_idx, :])
        return holder * T

    def get_leaf_Normalizing_Factors(
        self, MSD: npt.NDArray[np.float64], EL: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        Normalizing factor (NF) matrix and base case at the leaves.

        Each element in this N by 1 matrix is the normalizing
        factor for each beta value calculation for each node.
        This normalizing factor is essentially the marginal
        observation distribution for a node.

        This function gets the normalizing factor for
        the upward recursion only for the leaves.
        We first calculate the joint probability
        using the definition of conditional probability:

        :math:`P(x_n = x | z_n = k) * P(z_n = k) = P(x_n = x , z_n = k)`,
        where n are the leaf nodes.

        We can then sum this joint probability over k,
        which are the possible states z_n can be,
        and through the law of total probability,
        obtain the marginal observation distribution
        :math:`P(x_n = x) = sum_k ( P(x_n = x , z_n = k) ) = P(x_n = x)`.

        :param EL: The emissions likelihood
        :param MSD: The marginal state distribution P(z_n = k)
        :return: normalizing factor. The marginal observation distribution P(x_n = x)
        """
        NF_array = np.zeros(MSD.shape[0], dtype=float)  # instantiating N by 1 array

        # P(x_n = x , z_n = k) = P(x_n = x | z_n = k) * P(z_n = k)
        # this product is the joint probability
        # P(x_n = x) = sum_k ( P(x_n = x , z_n = k) )
        # the sum of the joint probabilities is the marginal probability
        NF_array[self.leaves_idx] = np.einsum("ij,ij->i", MSD[self.leaves_idx, :], EL[self.leaves_idx, :])
        assert np.all(np.isfinite(NF_array))
        return NF_array

    def sum_nonleaf_gammas(self, gamma_arr: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Sum of the gammas of the cells that are able to divide, that is,
        sum the of the gammas of all the nonleaf cells. It is used in estimating the transition probability matrix.
        This is an inner component in calculating the overall transition probability matrix.

        This is downward recursion.

        :param lO: the object of lineage tree
        :param gamma_arr: the gamma values for each lineage
        :return: the sum of gamma values for each state for non-leaf cells.
        """
        sum_wo_leaf = np.zeros(gamma_arr.shape[1])
        for level in self.output_list_of_gens[
            1:
        ]:  # sum the gammas for cells that are transitioning
            for cell in level:
                if not cell.isLeaf():
                    cell_idx = self.output_lineage.index(cell)
                    sum_wo_leaf += gamma_arr[cell_idx, :]
        assert np.all(np.isfinite(sum_wo_leaf))

        return sum_wo_leaf

    def get_gamma(self, T: npt.NDArray[np.float64], MSD: npt.NDArray[np.float64], beta: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Get the gammas using downward recursion from the root nodes.
        The conditional probability of states, given observation of the whole tree P(z_n = k | X_bar = x_bar)
        x_bar is the observations for the whole tree.
        gamma_1 (k) = P(z_1 = k | X_bar = x_bar)
        gamma_n (k) = P(z_n = k | X_bar = x_bar)

        :param MSD: The marginal state distribution P(z_n = k)
        :param betas: beta values. The conditional probability of states, given observations of the sub-tree rooted in cell_n
        """
        gamma = np.zeros((len(self.output_lineage), T.shape[0]))
        gamma[0, :] = beta[0, :]

        lineage = self.output_lineage
        coeffs = beta / np.clip(MSD, np.finfo(float).eps, np.inf)
        coeffs = np.clip(coeffs, np.finfo(float).eps, np.inf)
        beta_parents = np.einsum("ij,kj->ik", T, coeffs)

        for level in self.output_list_of_gens[1:]:
            for cell in level:
                parent_idx = lineage.index(cell)
                gam = gamma[parent_idx, :]

                for d in cell.get_daughters():
                    ci = lineage.index(d)
                    gamma[ci, :] = coeffs[ci, :] * np.matmul(
                        gam / beta_parents[:, ci], T
                    )

        assert np.all(np.isfinite(gamma))
        return gamma

    def __len__(self):
        """Defines the length of a lineage by returning the number of cells
        it contains.
        """
        return len(self.output_lineage)


def generate_lineage_list(
    pi: np.ndarray, T: np.ndarray, desired_num_cells: int
) -> list:
    """
    Generates a single lineage tree given Markov variables.
    This only generates the hidden variables (i.e., the states) in a output binary tree manner.
    It keeps generating cells in the tree until it reaches the desired number of cells in the lineage.
    :param pi: An array of the initial probability of a cell being a certain state.
    :param T: An array of the probability of a cell switching states or remaining in the same state.
    :param desired_num_cells: The desired number of cells in a lineage.
    :return full_lineage: A list of the generated cell lineage.
    """
    first_state = np.random.choice(
        pi.size, p=pi
    )  # roll the dice and yield the state for the first cell
    first_cell = CellVar(parent=None, state=first_state)  # create first cell
    full_lineage = [first_cell]  # instantiate lineage with first cell

    for cell in full_lineage:  # letting the first cell proliferate
        if cell.isLeaf():  # if the cell has no daughters...
            # make daughters by dividing and assigning states
            full_lineage.extend(cell.divide(T))

        if len(full_lineage) >= desired_num_cells:
            break
    return full_lineage


def output_assign_obs(state: int, full_lineage: list[CellVar], E: list):
    """
    Observation assignment give a state.
    Given the lineageTree object and the intended state, this function assigns the corresponding observations
    coming from specific distributions for that state.
    :param state: The integer value of the state that is being observed.
    :param full_lineage: The list of cells within the lineageTree object.
    :param E: The list of observations assignments.
    """
    cells_in_state = [cell for cell in full_lineage if cell.state == state]
    list_of_tuples_of_obs = E[state].rvs(size=len(cells_in_state))
    list_of_tuples_of_obs = list(map(list, zip(*list_of_tuples_of_obs)))

    assert len(cells_in_state) == len(list_of_tuples_of_obs)
    for i, cell in enumerate(cells_in_state):
        cell.obs = list_of_tuples_of_obs[i]


def max_gen(lineage: list[CellVar]) -> list[list[CellVar]]:
    """
    Finds the maximal generation in the tree, and cells organized by their generations.
    This walks through the cells in a given lineage, finds the maximal generation, and the group of cells belonging to a same generation and
    creates a list of them, appends the lists leading to have a list of the lists of cells in specific generations.
    :param lineage: The list of cells in a lineageTree object.
    :return max: The maximal generation in the tree.
    :return cells_by_gen: The list of lists of cells belonging to the same generation separated by specific generations.
    """
    gens = sorted(
        {cell.gen for cell in lineage}
    )  # appending the generation of cells in the lineage
    cells_by_gen: list[list[CellVar]] = [[]]
    for gen in gens:
        level = [cell for cell in lineage if (cell.gen == gen and cell.observed)]
        cells_by_gen.append(level)
    return cells_by_gen


def get_leaves_idx(lineage: list[CellVar]) -> np.ndarray:
    """
    A function to find the leaves and their indexes in the lineage list.
    :param lineage: The list of cells in a lineageTree object.
    :return leaf_indices: The list of cell indexes.
    :return leaves: The last cells in the lineage branch.
    """
    leaf_indices = []
    for index, cell in enumerate(lineage):
        if cell.isLeaf():
            assert cell.observed
            leaf_indices.append(index)  # appending the index of the cells
    return np.array(leaf_indices)
