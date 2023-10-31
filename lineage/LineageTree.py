""" This file contains the LineageTree class. """
import numpy as np
import numpy.typing as npt
import operator
from .CellVar import CellVar


class LineageTree:
    """A class for lineage trees. This class also handles algorithms for walking
    the tree to calculate various properties.
    """

    pi: npt.NDArray[np.float64]
    T: npt.NDArray[np.float64]
    leaves_idx: npt.NDArray[np.uintp]
    idx_by_gen: list[np.ndarray]
    output_lineage: list[CellVar]
    cell_to_parent: np.ndarray
    cell_to_daughters: np.ndarray

    def __init__(self, list_of_cells: list, E: list):
        self.E = E
        # output_lineage must be sorted according to generation
        self.output_lineage = sorted(list_of_cells, key=operator.attrgetter("gen"))
        self.idx_by_gen = max_gen(self.output_lineage)
        # assign times using the state distribution specific time model
        E[0].assign_times(self.output_lineage)

        self.leaves_idx = get_leaves_idx(self.output_lineage)
        self.cell_to_parent = cell_to_parent(self.output_lineage)
        self.cell_to_daughters = cell_to_daughters(self.output_lineage)

    @classmethod
    def rand_init(
        cls,
        pi: np.ndarray,
        T: np.ndarray,
        E: list,
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
        first_state = rng.choice(
            pi.size, p=pi
        )  # roll the dice and yield the state for the first cell
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
            list_of_tuples_of_obs = list(map(list, zip(*list_of_tuples_of_obs)))

            assert len(cells_in_state) == len(list_of_tuples_of_obs)
            for i, cell in enumerate(cells_in_state):
                cell.obs = list_of_tuples_of_obs[i]

        # assign times using the state distribution specific time model
        E[0].assign_times(full_lineage)

        output_lineage = E[0].censor_lineage(
            censor_condition, full_lineage, desired_experiment_time
        )

        lineageObj = cls(output_lineage, E)
        lineageObj.pi = pi
        lineageObj.T = T
        return lineageObj

    def __len__(self):
        """Defines the length of a lineage by returning the number of cells
        it contains.
        """
        return len(self.output_lineage)


def get_Emission_Likelihoods(X: list[LineageTree], E: list) -> list:
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
    all_cells = np.array([cell.obs for lineage in X for cell in lineage.output_lineage])
    ELstack = np.zeros((len(all_cells), len(E)))

    for k in range(len(E)):  # for each state
        ELstack[:, k] = np.exp(E[k].logpdf(all_cells))
        assert np.all(np.isfinite(ELstack[:, k]))
    EL = []
    ii = 0
    for lineageObj in X:  # for each lineage in our Population
        nl = len(lineageObj.output_lineage)  # getting the lineage length
        EL.append(ELstack[ii : (ii + nl), :])  # append the EL_array for each lineage

        ii += nl

    return EL


def generate_lineage_list(
    pi: npt.NDArray[np.float64], T: npt.NDArray[np.float64], desired_num_cells: int
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


def max_gen(lineage: list[CellVar]) -> list[np.ndarray]:
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
    cells_by_gen: list[np.ndarray] = []
    for gen in gens:
        level = np.array(
            [
                lineage.index(cell)
                for cell in lineage
                if (cell.gen == gen and cell.observed)
            ],
            dtype=int,
        )
        cells_by_gen.append(level)
    return cells_by_gen


def cell_to_parent(lineage: list[CellVar]) -> np.ndarray:
    output = np.full(len(lineage), -1, dtype=int)
    for ii, cell in enumerate(lineage):
        parent = cell.parent
        if parent is not None:
            output[ii] = lineage.index(parent)

    return output


def cell_to_daughters(lineage: list[CellVar]) -> np.ndarray:
    output = np.full((len(lineage), 2), -1, dtype=int)
    for ii, cell in enumerate(lineage):
        if cell.left is not None and cell.left in lineage:
            output[ii, 0] = lineage.index(cell.left)

        if cell.right is not None and cell.right in lineage:
            output[ii, 1] = lineage.index(cell.right)

    return output


def get_leaves_idx(lineage: list[CellVar]) -> npt.NDArray[np.uintp]:
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
    return np.array(leaf_indices, dtype=np.uintp)
