"""This file contains the LineageTree class."""

import operator
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_array

from .CellVar import CellVar
from .states.StateDistributionGamma import StateDistribution as StA
from .states.StateDistributionGaPhs import StateDistribution as StB


class LineageTree:
    """A class for lineage trees. This class also handles algorithms for walking
    the tree to calculate various properties.
    """

    pi: npt.NDArray[np.float64]
    T: npt.NDArray[np.float64]
    leaves_idx: np.ndarray
    output_lineage: list[CellVar]
    obs: np.ndarray
    tree: csr_array
    states: np.ndarray
    E: Sequence[StA | StB]

    def __init__(self, list_of_cells: list, E: Sequence[StA | StB]):
        self.E = E
        # output_lineage must be sorted according to generation
        self.output_lineage = sorted(list_of_cells, key=operator.attrgetter("gen"))

        self.tree = lineage_to_tree(self.output_lineage)

        # Leaves have no daughters
        self.leaves_idx = np.nonzero(np.diff(self.tree.indptr) == 0)[0]

        self.states = np.array([cell.state for cell in self.output_lineage], dtype=int)
        self.obs = np.array([cell.obs for cell in self.output_lineage])

    @property
    def non_leaves_idx(self) -> np.ndarray:
        """Return array of non-leaf cell indices."""
        return np.nonzero(np.diff(self.tree.indptr) > 0)[0]

    @property
    def edges(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (parents, daughters) edge arrays."""
        parents = np.repeat(np.arange(self.tree.shape[0]), np.diff(self.tree.indptr))
        return parents, self.tree.indices

    @property
    def cell_to_daughters(self) -> np.ndarray:
        """Compatibility helper returning (N, 2) array of daughter indices."""
        output = np.full((len(self.output_lineage), 2), -1, dtype=int)
        for i in range(len(self.output_lineage)):
            children = self.tree.indices[self.tree.indptr[i] : self.tree.indptr[i + 1]]
            for j, c in enumerate(children):
                if j < 2:
                    output[i, j] = c
        return output

    @classmethod
    def rand_init(
        cls,
        pi: np.ndarray,
        T: np.ndarray,
        E: Sequence[StA | StB],
        desired_num_cells: int,
        censor_condition=0,
        desired_experiment_time=2e12,
        rng=None,
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
        rng = np.random.default_rng(rng)

        # Generate lineage list
        first_state = rng.choice(pi.size, p=pi)  # roll the dice and yield the state for the first cell
        first_cell = CellVar(parent=None, state=first_state)  # create first cell
        full_lineage = [first_cell]  # instantiate lineage with first cell

        for cell in full_lineage:  # letting the first cell proliferate
            if cell.isLeaf():  # if the cell has no daughters...
                # make daughters by dividing and assigning states
                full_lineage.extend(cell.divide(T, rng=rng))

            if len(full_lineage) >= desired_num_cells:
                break

        # Assign observations
        for i_state in range(pi.size):
            cells_in_state = [cell for cell in full_lineage if cell.state == i_state]
            list_of_tuples_of_obs = E[i_state].rvs(size=len(cells_in_state), rng=rng)
            list_of_tuples_of_obs = list(map(list, zip(*list_of_tuples_of_obs, strict=False)))

            assert len(cells_in_state) == len(list_of_tuples_of_obs)
            for i, cell in enumerate(cells_in_state):
                cell.obs = list_of_tuples_of_obs[i]

        output_lineage = E[0].censor_lineage(censor_condition, full_lineage, desired_experiment_time)

        lineageObj = cls(output_lineage, E)
        lineageObj.pi = pi
        lineageObj.T = T
        return lineageObj

    def __len__(self):
        """Defines the length of a lineage by returning the number of cells
        it contains.
        """
        return len(self.output_lineage)


def get_Emission_Likelihoods(X: list[LineageTree], E: list) -> list[np.ndarray]:
    """
    Emission Likelihood (EL) matrix.

    Each element in this N by K matrix represents the probability

    :math:`P(x_n = x | z_n = k)`,

    for all :math:`x_n` and :math:`z_n` in our observed and hidden state tree
    and for all possible discrete states k.
    :param X: list of lineage trees
    :param E: The emissions likelihood
    :return: The marginal state distribution
    """
    all_cells = np.vstack([lineage.obs for lineage in X])
    ELstack = np.zeros((len(all_cells), len(E)))

    for k in range(len(E)):  # for each state
        ELstack[:, k] = np.exp(E[k].logpdf(all_cells))
        assert np.all(np.isfinite(ELstack[:, k]))
    EL = []
    ii = 0
    for lineageObj in X:  # for each lineage in our Population
        nl = len(lineageObj)  # getting the lineage length
        EL.append(ELstack[ii : (ii + nl), :])  # append the EL_array for each lineage

        ii += nl

    return EL


def lineage_to_tree(lineage: list[CellVar]) -> csr_array:
    """Build a directed adjacency CSR array (parent -> daughter) from a lineage."""
    n = len(lineage)
    if n == 0:
        return csr_array((0, 0), dtype=bool)

    cell_indices = {cell: idx for idx, cell in enumerate(lineage)}
    indptr = np.zeros(n + 1, dtype=np.int32)
    indices_list: list[int] = []

    for i, cell in enumerate(lineage):
        count = 0
        if cell.left in cell_indices:
            indices_list.append(cell_indices[cell.left])
            count += 1
        if cell.right in cell_indices:
            indices_list.append(cell_indices[cell.right])
            count += 1
        indptr[i + 1] = indptr[i] + count

    indices = np.array(indices_list, dtype=np.int32)
    data = np.ones(len(indices), dtype=bool)
    return csr_array((data, indices, indptr), shape=(n, n))


def cell_to_daughters(lineage: list[CellVar]) -> np.ndarray:
    """Compatibility helper converting a lineage list to an (N, 2) daughter index array."""
    tree = lineage_to_tree(lineage)
    output = np.full((len(lineage), 2), -1, dtype=int)
    for i in range(len(lineage)):
        children = tree.indices[tree.indptr[i] : tree.indptr[i + 1]]
        for j, c in enumerate(children):
            if j < 2:
                output[i, j] = c
    return output
