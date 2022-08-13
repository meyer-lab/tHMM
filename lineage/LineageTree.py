""" This file contains the LineageTree class. """
import numpy as np
import operator
from .CellVar import CellVar


class LineageTree:
    """A class for lineage trees.
    Every lineage object from this class is a binary tree built based on initial probabilities,
    transition probabilities, and emissions defined by state distributions given by the user.
    Lineages are generated in full (no pruning) by creating cells of different states in a
    binary fashion utilizing the pi and the transtion probabilities. Cells are then filled with
    observations based on their states by sampling observations from their emission distributions.
    The lineage tree is then censord based on the censor condition.
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
        barcode: int = 0,
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
            pi=pi, T=T, desired_num_cells=desired_num_cells, barcode=barcode
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

    def get_parents_for_level(self, level: list) -> set:
        """
        Get the parents's index of a generation in the population list.
        Given the generation level, this function returns the index of parent cells of the cells being in that generation level.
        :param level: The number of the generation for a particular parent cell.
        :return parent_holder: The index of parent cells for the cells in a given generation level.
        """
        parent_holder = set()  # set makes sure only one index is put in and no overlap
        for cell in level:
            parent_holder.add(self.output_lineage.index(cell.parent))
        return parent_holder

    def get_Marginal_State_Distributions(
        self, pi: np.ndarray, T: np.ndarray
    ) -> np.ndarray:
        r"""Marginal State Distribution (MSD) matrix and recursion.
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

        This is part of upward recursion.

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

    def get_leaf_Normalizing_Factors(
        self, MSD: np.ndarray, EL: np.ndarray
    ) -> np.ndarray:
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

        This is part of upward recursion.

        :param EL: The emissions likelihood
        :param MSD: The marginal state distribution P(z_n = k)
        :return: normalizing factor. The marginal observation distribution P(x_n = x)
        """
        MSD_array = MSD[
            self.leaves_idx, :
        ]  # getting the MSD of the lineage leaves
        EL_array = EL[self.leaves_idx, :]  # geting the EL of the lineage leaves
        NF_array = np.zeros(MSD.shape[0], dtype=float)  # instantiating N by 1 array

        # P(x_n = x , z_n = k) = P(x_n = x | z_n = k) * P(z_n = k)
        # this product is the joint probability
        # P(x_n = x) = sum_k ( P(x_n = x , z_n = k) )
        # the sum of the joint probabilities is the marginal probability
        NF_array[self.leaves_idx] = np.einsum("ij,ij->i", MSD_array, EL_array)
        assert np.all(np.isfinite(NF_array))
        return NF_array

    def __len__(self):
        """Defines the length of a lineage by returning the number of cells
        it contains.
        """
        return len(self.output_lineage)


def generate_lineage_list(
    pi: np.ndarray, T: np.ndarray, desired_num_cells: int, barcode: int
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
    first_cell_state = np.random.choice(pi.size, size=1, p=pi)[
        0
    ]  # roll the dice and yield the state for the first cell
    first_cell = CellVar(
        parent=None, gen=1, state=first_cell_state, barcode=barcode
    )  # create first cell
    full_lineage = [first_cell]  # instantiate lineage with first cell

    for cell in full_lineage:  # letting the first cell proliferate
        if cell.isLeaf():  # if the cell has no daughters...
            # make daughters by dividing and assigning states
            left_cell, right_cell = cell.divide(T)
            # add daughters to the list of cells
            full_lineage.append(left_cell)
            full_lineage.append(right_cell)

        if len(full_lineage) >= desired_num_cells:
            break
    return full_lineage


def output_assign_obs(state: int, full_lineage: list, E: list):
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


# tools for analyzing trees


def max_gen(lineage: list) -> list:
    """
    Finds the maximal generation in the tree, and cells organized by their generations.
    This walks through the cells in a given lineage, finds the maximal generation, and the group of cells belonging to a same generation and
    creates a list of them, appends the lists leading to have a list of the lists of cells in specific generations.
    :param lineage: The list of cells in a lineageTree object.
    :return max: The maximal generation in the tree.
    :return list_of_lists_of_cells_by_gen: The list of lists of cells belonging to the same generation separated by specific generations.
    """
    gens = sorted(
        {cell.gen for cell in lineage}
    )  # appending the generation of cells in the lineage
    list_of_lists_of_cells_by_gen = [[None]]
    for gen in gens:
        level = [cell for cell in lineage if (cell.gen == gen and cell.observed)]
        list_of_lists_of_cells_by_gen.append(level)
    return list_of_lists_of_cells_by_gen


def get_leaves_idx(lineage: list) -> np.ndarray:
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
