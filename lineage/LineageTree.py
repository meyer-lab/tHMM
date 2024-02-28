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
    output_lineage: list[CellVar]

    def __init__(self, list_of_cells: list, E: list):
        self.E = E
        # sort according to generation
        sorted_cells = sorted(list_of_cells, key=operator.attrgetter("gen"))

        # assign times using the state distribution specific time model
        E[0].assign_times(sorted_cells)

        # add root
        self.output_lineage = [sorted_cells[0]]

        # build remaining tree
        for parent_idx in range(2**(sorted_cells[-1].gen - 1) - 1):
            parent = self.output_lineage[parent_idx]

            if parent is not None:
                self.output_lineage.append(parent.left)
                self.output_lineage.append(parent.right)

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
        pIDX = 0

        # fill in the cells
        while len(full_lineage) < desired_num_cells:  # letting the first cell proliferate
            parent = full_lineage[pIDX]
            states = rng.choice(T.shape[0], size=2, p=T[parent.state, :])

            full_lineage.append(CellVar(parent=parent, state=states[0]))
            full_lineage.append(CellVar(parent=parent, state=states[1]))
            parent.left = full_lineage[-2]
            parent.right = full_lineage[-1]

            pIDX += 1

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
    all_cells: list[np.ndarray] = []

    for lineage in X:
        for cell in lineage.output_lineage:
            if cell is None:
                all_cells.append(-1 * np.ones(all_cells[0].size))
            else:
                all_cells.append(np.array(cell.obs))

    all_cells = np.array(all_cells) # type: ignore
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
