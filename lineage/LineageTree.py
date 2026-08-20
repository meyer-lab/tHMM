"""This file contains the LineageTree class."""

import operator
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_array

from .CellVar import CellVar
from .states.stateCommon import censor_lineage_gamma
from .states.StateDistributionGamma import StateDistribution as StA
from .states.StateDistributionGaPhs import StateDistribution as StB


class LineageTree:
    """A class for lineage trees. This class also handles algorithms for walking
    the tree to calculate various properties.
    """

    pi: npt.NDArray[np.float64]
    T: npt.NDArray[np.float64]
    leaves_idx: np.ndarray
    _output_lineage: list[CellVar] | None
    obs: np.ndarray
    tree: csr_array
    states: np.ndarray
    E: Sequence[StA | StB]

    def __init__(
        self,
        list_of_cells: list | csr_array,
        E: Sequence[StA | StB],
        obs: np.ndarray | None = None,
        states: np.ndarray | None = None,
    ):
        self.E = E
        if isinstance(list_of_cells, list):
            self._output_lineage = sorted(list_of_cells, key=operator.attrgetter("gen"))
            self.tree = lineage_to_tree(self._output_lineage)
            self.states = np.array([cell.state for cell in self._output_lineage], dtype=int)
            self.obs = np.array([cell.obs for cell in self._output_lineage])
        else:
            self.tree = list_of_cells
            self.obs = obs if obs is not None else np.empty((self.tree.shape[0], 0))
            self.states = states if states is not None else np.full(self.tree.shape[0], -1, dtype=int)
            self._output_lineage = None

        # Leaves have no daughters
        self.leaves_idx = np.nonzero(np.diff(self.tree.indptr) == 0)[0]

    @property
    def output_lineage(self) -> list[CellVar]:
        """Backwards compatibility property constructing CellVar list from arrays."""
        if self._output_lineage is not None:
            return self._output_lineage
        return self._build_cellvar_list()

    def _build_cellvar_list(self) -> list[CellVar]:
        n = self.tree.shape[0]
        cells = [CellVar(state=int(self.states[i])) for i in range(n)]
        for i in range(n):
            cells[i].obs = self.obs[i].tolist() if hasattr(self.obs[i], "tolist") else self.obs[i]
            children = self.tree.indices[self.tree.indptr[i] : self.tree.indptr[i + 1]]
            if len(children) > 0:
                cells[i].left = cells[children[0]]
                cells[children[0]].parent = cells[i]
                cells[children[0]].gen = cells[i].gen + 1
            if len(children) > 1:
                cells[i].right = cells[children[1]]
                cells[children[1]].parent = cells[i]
                cells[children[1]].gen = cells[i].gen + 1
        self._output_lineage = cells
        return cells

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
        n = self.tree.shape[0]
        output = np.full((n, 2), -1, dtype=int)
        for i in range(n):
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
        Constructor method generating pure array representation.

        :param :math:`\pi`: The initial probability matrix.
        :param T: The transition probability matrix.
        :param E: A list containing state distribution objects.
        :param desired_num_cells: The desired number of cells.
        :param censor_condition: An integer in {0, 1, 2, 3} deciding censoring type.
        """
        assert pi.size == T.shape[0]
        assert T.shape[0] == T.shape[1]
        rng = np.random.default_rng(rng)

        # Generate tree connectivity and states
        states = [int(rng.choice(pi.size, p=pi))]
        rows: list[int] = []
        cols: list[int] = []

        curr = 0
        while len(states) < desired_num_cells:
            parent_state = states[curr]
            left_s, right_s = rng.choice(T.shape[0], size=2, p=T[parent_state, :])
            c1 = len(states)
            c2 = len(states) + 1
            states.extend([int(left_s), int(right_s)])
            rows.extend([curr, curr])
            cols.extend([c1, c2])
            curr += 1

        n = len(states)
        states_arr = np.array(states, dtype=int)
        full_tree = csr_array((np.ones(len(rows), dtype=bool), (rows, cols)), shape=(n, n))

        # Sample observations from state distributions
        obs_by_state = {}
        obs_dim = None
        for s in range(pi.size):
            s_idx = np.where(states_arr == s)[0]
            if len(s_idx) > 0:
                rvs_out = E[s].rvs(size=len(s_idx), rng=rng)
                stacked = np.column_stack(rvs_out)
                obs_by_state[s] = (s_idx, stacked)
                if obs_dim is None:
                    obs_dim = stacked.shape[1]

        obs_arr = np.zeros((n, obs_dim or 0), dtype=float)
        for s_idx, s_obs in obs_by_state.values():
            obs_arr[s_idx, :] = s_obs

        # Apply censoring directly on arrays
        if hasattr(E[0], "censor_lineage_array"):
            pruned_tree, pruned_obs, pruned_states = E[0].censor_lineage_array(
                censor_condition, full_tree, obs_arr, states_arr, desired_experiment_time
            )
        else:
            pruned_tree, pruned_obs, pruned_states = censor_lineage_gamma(
                full_tree, obs_arr, states_arr, censor_condition, desired_experiment_time
            )

        lineageObj = cls(pruned_tree, E, obs=pruned_obs, states=pruned_states)
        lineageObj.pi = pi
        lineageObj.T = T
        return lineageObj

    def __len__(self):
        """Defines the length of a lineage by returning the number of cells
        it contains.
        """
        return self.tree.shape[0]


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
