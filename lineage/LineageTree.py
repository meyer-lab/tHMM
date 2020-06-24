""" This file contains the LineageTree class. """
from copy import deepcopy
import scipy.stats as sp
import operator

from .CellVar import CellVar
from .states.stateCommon import assign_times, basic_censor, fate_censor, time_censor


class LineageTree:
    """A class for lineage trees.
    Every lineage object from this class is a binary tree built based on initial probabilities,
    transition probabilities, and emissions defined by state distributions given by the user.
    Lineages are generated in full (no pruning) by creating cells of different states in a
    binary fashion utilizing the pi and the transtion probabilities. Cells are then filled with
    observations based on their states by sampling observations from their emission distributions.
    The lineage tree is then censord based on the censor condition.
    """

    def __init__(self, list_of_cells, E):
        self.E = E
        self.output_lineage = sorted(list_of_cells, key=operator.attrgetter('gen'))
        self.output_max_gen, self.output_list_of_gens = max_gen(self.output_lineage)
        self.output_leaves_idx, self.output_leaves = get_leaves(self.output_lineage)
        
    @classmethod
    def init_from_parameters(cls, pi, T, E, desired_num_cells, censor_condition=0, **kwargs):
        """
        Constructor method

        :param :math:`\pi`: The initial probability matrix; its shape must be the same as the number of states and all of them must sum up to 1.
        :type :math:`\pi`: Array
        :param T: The transition probability matrix; every row must sum up to 1.
        :type T: square Matrix
        :param E: A list containing state distribution objects, the length of it is the same as the number of states.
        :type E: list
        :param desired_num_cells: The desired number of cells we want the lineage to end up with.
        :type desired_num_cells: Int
        :param censor_condition: An integer :math:`\in` \{0, 1, 2, 3\} that decides the type of censoring.
        :type censor_condition: Int

        Censoring guide
        - 0 means no pruning
        - 1 means censor based on the fate of the cell
        - 2 means censor based on the length of the experiment
        - 3 means censor based on both the 'fate' and 'time' conditions
        """
        pi_num_states = len(pi)
        T_shape = T.shape
        assert (
            T_shape[0] == T_shape[1]
        ), "Transition numpy array is not square. Ensure that your transition numpy array has the same number of rows and columns."
        T_num_states = T.shape[0]
        E_num_states = len(E)
        assert pi_num_states == T_num_states == E_num_states, \
            f"The number of states in your input Markov probability parameters are mistmatched. \
        \nPlease check that the dimensions and states match. \npi {pi} \nT {T} \nE {E}"

        num_states = pi_num_states

        full_lineage = generate_lineage_list(pi=pi, T=T, desired_num_cells=desired_num_cells)
        for i_state in range(num_states):
            output_assign_obs(i_state, full_lineage, E)

        full_max_gen, full_list_of_gens = max_gen(full_lineage)
        full_leaves_idx, full_leaves = get_leaves(full_lineage)
        # TODO: assign_times needs to be moved
        if len(E[0].rvs(1)) > 1:
            assign_times(full_lineage)

        if kwargs:
            desired_experiment_time = kwargs.get("desired_experiment_time", 2e12)

        output_lineage = censor_lineage(censor_condition, desired_experiment_time)

        lineageObj = cls(output_lineage)
        
        lineageObj.pi = pi
        lineageObj.T = T
        lineageObj.E = E
        lineageObj.num_states = num_states
        lineageObj.full_lineage = full_lineage
        lineageObj.full_max_gen = full_max_gen
        lineageObj.full_list_of_gens = full_list_of_gens
        lineageObj.full_leaves_idx = full_leaves_idx
        lineageObj.full_leaves = full_leaves
        
        return lineageObj
        
    def get_parents_for_level(self, level):
        """Get the parents's index of a generation in the population list.
        Given the generation level, this function returns the index of parent cells of the cells being in that generation level.

        :param level: A list containing cells in a specific generation level.
        :type level: list
        :return: A set holding the parents' indexes of cells in a given generation.
        :rtype: set
        """
        parent_holder = set()  # set makes sure only one index is put in and no overlap
        for cell in level:
            parent_holder.add(self.output_lineage.index(cell.parent))
        return parent_holder

    def __repr__(self):
        """This function is used to get string representation of an object, used for debugging and development.
        Represents the information about the lineage that the user has created,
        like whether the tree is censord or is a output tree;
        and for both of the options it prints the number of states,
        the number of cells in the states, the total number of cells.
        """
        s = ""
        for cell in self.output_lineage:
            s += cell.__repr__()
        return s

    def __str__(self):
        """This function is used to get string representation of an object,
        used for showing the results to the user.
        """
        return self.__repr__()

    def __len__(self):
        """Defines the length of a lineage by returning the number of cells
        it contains.
        """
        return len(self.output_lineage)

    def is_heterogeneous(self):
        """Checks whether a lineage is heterogeneous by ensuring that the true states
        of the cells contained within it create a set that has more than one state.
        """
        true_states_set_len = len({cell.state for cell in self.output_lineage})
        if true_states_set_len > 1:
            return True
        return False
        

def generate_lineage_list(pi, T, desired_num_cells):
    """
    Generates a single lineage tree given Markov variables.
    This only generates the hidden variables (i.e., the states) in a output binary tree manner.
    It keeps generating cells in the tree until it reaches the desired number of cells in the lineage.
    """
    first_state_results = sp.multinomial.rvs(1, pi)  # roll the dice and yield the state for the first cell
    first_cell_state = first_state_results.tolist().index(1)
    first_cell = CellVar(parent=None, gen=1, state=first_cell_state, synthetic=True)  # create first cell
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


def output_assign_obs(state, full_lineage, E):
    """
    Observation assignment give a state.
    Given the lineageTree object and the intended state, this function assigns the corresponding observations
    comming from specific distributions for that state.

    :param state: The number assigned to a state.
    :type state: Int
    """
    cells_in_state = [cell for cell in full_lineage if cell.state == state]
    list_of_tuples_of_obs = E[state].rvs(size=len(cells_in_state))
    list_of_tuples_of_obs = list(map(list, zip(*list_of_tuples_of_obs)))

    assert len(cells_in_state) == len(list_of_tuples_of_obs)
    for i, cell in enumerate(cells_in_state):
        cell.obs = list_of_tuples_of_obs[i]

def censor_lineage(censor_condition):
    """
    This function removes those cells that are intended to be remove
    from the output binary tree based on emissions.
    It takes in LineageTree object, walks through all the cells in the output binary tree,
    applies the pruning to each cell that is supposed to be removed,
    and returns the censord list of cells.
    """
    if censor_condition == 0:
        output_lineage = full_lineage
        return output_lineage

    self.output_lineage = []
    for cell in self.full_lineage:
        basic_censor(cell)
        if self.censor_condition == 1:
            fate_censor(cell)
        elif self.censor_condition == 2:
            time_censor(cell, self.desired_experiment_time)
        elif self.censor_condition == 3:
            fate_censor(cell)
            time_censor(cell, self.desired_experiment_time)
        if not cell.censored:
            self.output_lineage.append(cell)
    return output_lineage



# tools for analyzing trees


def max_gen(lineage):
    """Finds the maximal generation in the tree, and cells organized by their generations.
    This walks through the cells in a given lineage, finds the maximal generation, and the group of cells belonging to a same generation and
    creates a list of them, appends the lists leading to have a list of the lists of cells in specific generations.

    :param lineage: A list of cells (objects) with known state, generation, ect.
    :type lineage: list
    :return: The maximal generation in the given lineage.
    :rtype: Int
    :return: A list of lists of cells, organized by their generations.
    :rtype: list
    """
    gens = sorted({cell.gen for cell in lineage})  # appending the generation of cells in the lineage
    list_of_lists_of_cells_by_gen = [[None]]
    for gen in gens:
        level = [cell for cell in lineage if (cell.gen == gen and not cell.censored)]
        list_of_lists_of_cells_by_gen.append(level)
    return max(gens), list_of_lists_of_cells_by_gen


def get_leaves(lineage):
    """
    A function to find the leaves and their indexes in the lineage list.

    :param lineage: A list of cells in the lineage.
    :type lineage: list
    :return: A list of indexes to the leaf cells in the lineage list.
    :rtype: list
    :return: A list holding the leaf cells in the lineage given.
    :rtype: list
    """
    leaf_indices = []
    leaves = []
    for index, cell in enumerate(lineage):
        if cell.isLeaf():
            if not cell.isRootParent:
                assert not cell.parent.censored
            leaves.append(cell)  # appending the leaf cells to a list
            leaf_indices.append(index)  # appending the index of the cells
    return leaf_indices, leaves
