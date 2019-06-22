'''Utility and helper functions for recursions and other needs in the tHMM class. This also contains the methods for AIC and accuracy.'''

import itertools
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score
from scipy.special import gamma, gammaincc


##------------------- Find maximum generation in a lineage -----------------------##


def max_gen(lineage):
    """
    finds the max generation in a lineage tree, in a given experiment time;
    i.e., the generation of the leaf cells.

    Args:
        ----------
        lineage (list): a list of objects (cells) in a lineage.

    Returns:
        ----------
        gen_holder (int): the maximum generation in a lineage.
    """
    gen_holder = 1
    for cell in lineage:
        if cell.gen > gen_holder:
            gen_holder = cell.gen
    return gen_holder

##---------------------- Finding the cells in a generation -------------------------##


def get_gen(gen, lineage):
    """
    Creates a list with all cells in the given generation
    Args:
        ----------
        gen (int): the generation number that we want to separate from the rest.
        lineage (list of objects): a list holding the objects (cells) in a lineage.

    Returns:
        ----------
        first_set (list of objects): a list that holds the cells with the same given
        generation.
    """
    first_set = []
    for cell in lineage:
        if cell.gen == gen:
            first_set.append(cell)
    return first_set

##----------------------finding parents of cells in a generation------------------##


def get_parents_for_level(level, lineage):
    """
    Returns a set of all the parents of all the cells in a
    given level/generation. For example this would give you
    all the non-leaf cells in the generation above the one given.

    Args:
        ----------
        level (list of objects): a list that holds objects (cells) in a given level
        (or generation).
        lineage (list of objects): a list hodling objects (cells) in a lineage

    Returns:
        ----------
        parent_holder (set): a list that holds objects (cells) which
        are the parents of the cells in a given generation
    """
    parent_holder = set()  # set makes sure only one index is put in and no overlap
    for cell in level:
        parent_cell = cell.parent
        parent_holder.add(lineage.index(parent_cell))
    return parent_holder

##---------------------- finding daughter of a given cell -------------------------##


def get_daughters(cell):
    """
    Returns a list of the daughters of a given cell.
    Args:
        ----------
        cell (obj): an object (the cell) with different instances, including
        the cell's right daughter and cell's left daughter.

    Returns:
        ----------
        temp (list): a list of two objects, i.e., two daughter cells of a given cell.
    """
    temp = []
    if cell.left:
        temp.append(cell.left)
    if cell.right:
        temp.append(cell.right)
    return temp

##------------------------ Akaike Information Criterion -------------------------##


def getAIC(tHMMobj, LL):
    '''
    Gets the AIC values. Akaike Information Criterion, used for model selection and deals with the trade off
    between over-fitting and under-fitting.
    AIC = 2*k - 2 * log(LL) in which k is the number of free parameters and LL is the maximum of likelihood function.
    Minimum of AIC detremines the relatively better model.

    Args:
        ----------
        tHMMobj (obj): the tHMM class which has been built.
        LL (list): a list containing log-likelihood values of Normalizing Factors for each lineage.

    Returns:
        ----------
        AIC_ls_rel_0 (list): containing AIC values relative to 0 for each lineage.
        LL_ls_rel_0 (list): containing LL values relative to 0 for each lineage.
        AIC_degrees_of_freedom : the degrees of freedom in AIC calculation (numStates**2 + numStates * number_of_parameters - 1) - same for each lineage


    Example Usage:

        from matplotlib.ticker import MaxNLocator

        x1val = []
        x2val = []
        yval = []
        for numState in range(3):
            tHMMobj = tHMM(X, numStates=numState, FOM='G') # build the tHMM class with X
            tHMMobj, NF, betas, gammas, LL = fit(tHMMobj, max_iter=100, verbose=False)
            AIC_value, numStates, deg = getAIC(tHMMobj, LL)
            x1val.append(numStates)
            x2val.append(deg)
            yval.append(AIC_value)

        fig = plt.figure(figsize=(10,10))
        ax1 = fig.add_subplot(111)
        ax1.scatter(xval, yval, marker='*', c='b', s=500, label='One state data/model')
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax1.grid(True, linestyle='--')
        ax1.set_xlabel('Number of States')
        ax1.set_ylabel('AIC Cost')
        title = ax1.set_title('Akaike Information Criterion')
        title.set_y(1.1)
        fig.subplots_adjust(top=1.3)

        ax2 = ax1.twiny()
        ax2.set_xticks([1]+ax1.get_xticks())
        ax2.set_xbound(ax1.get_xbound())
        ax2.set_xticklabels(x2val)
        ax2.set_xlabel('Number of parameters')

        ax1.legend()
        plt.rcParams.update({'font.size': 28})
        plt.show()
    '''
    numStates = tHMMobj.numStates
    AIC_ls = []
    LL_ls = []
    number_of_parameters = 0
    for param in range(tHMMobj.paramlist[0]['E'].shape[1]): #obtain the paramlist for one lineage which serves as same for all lineages because they all have the same E values
        number_of_parameters += 1
    AIC_degrees_of_freedom = numStates**2 + numStates * number_of_parameters - 1
    for num in range(tHMMobj.numLineages):
        AIC_value = -2 * LL[num] + 2 * AIC_degrees_of_freedom
        AIC_ls.append(AIC_value)
        LL_ls.append(LL[num])

    return(AIC_ls, LL_ls, AIC_degrees_of_freedom) # no longer returning relative to zero

##------------------------- Calculate accuracy ----------------------------------##


def getAccuracy(tHMMobj, all_states, verbose=False):
    '''
    Gets the accuracy for state assignment per lineage.

    This function takes in the tree-HMM model as an object and the state matrix assigned by Viterbi,
    and by permuting state assignments, it will find the best label assignment based on the
    higheset accuracy, and then changes the labels in Viterbi state assignment and the accuracy.

    Args:
        ----------
        tHMMobj (obj): tree-HMM model as an object
        all_states (matrix): a matrix holding the states assigned by viterbi algorithm as the most likely states.

    Returns:
        ----------
        tHMMobj.Accuracy (list): accuracy of state assignment
        tHMMobj.states (list): the correct order of states
        tHMMobj.stateAssignment (list): the correct states assigned by Viterbi

    Example usage:

        tHMMobj = tHMM(X, numStates=2, FOM='G') # build the tHMM class with X
        fit(tHMMobj, max_iter=500, verbose=True)
        deltas, state_ptrs = get_leaf_deltas(tHMMobj) # gets the deltas matrix
        get_nonleaf_deltas(tHMMobj, deltas, state_ptrs)
        all_states = Viterbi(tHMMobj, deltas, state_ptrs)

        accuracy, states, stateAssignment = getAccuracy(tHMMobj, all_states, verbose = True)

    '''
    numStates = tHMMobj.numStates
    tHMMobj.Accuracy = []
    tHMMobj.stateAssignment = []
    tHMMobj.states = []

    for lin in range(tHMMobj.numLineages):
        lineage = tHMMobj.population[lin]

        true_state_holder = np.zeros((len(lineage)), dtype=int)
        viterbi_est_holder = np.zeros((len(lineage)), dtype=int)

        for ii, cell in enumerate(lineage):
            true_state_holder[ii] = cell.true_state
            viterbi_est_holder[ii] = all_states[lin][ii]

        permutation_of_states = list(itertools.permutations(range(numStates)))
        temp_acc_holder = []
        for possible_state_assignment in permutation_of_states:
            # gets a list of lists of permutations of state assignments
            temp_all_states = all_states[lin].copy()
            for ii, temp_state in enumerate(temp_all_states):
                for state in range(numStates):
                    if temp_state == state:
                        temp_all_states[ii] = possible_state_assignment[state]

            common_state_counter = [true_state == temp_vit_state for (true_state, temp_vit_state) in zip(true_state_holder, temp_all_states)]
            accuracy = sum(common_state_counter) / len(lineage)  # gets the accuracies per possible state assignment
            temp_acc_holder.append(accuracy)

        idx_of_max_acc = np.argmax(temp_acc_holder)
        tHMMobj.Accuracy.append(temp_acc_holder[idx_of_max_acc])

        tHMMobj.stateAssignment.append(permutation_of_states[idx_of_max_acc])  # the correct state assignment

        for ii, cell_viterbi_state in enumerate(viterbi_est_holder):
            for state in range(numStates):
                if cell_viterbi_state == state:
                    viterbi_est_holder[ii] = tHMMobj.stateAssignment[lin][state]

        tHMMobj.states.append(viterbi_est_holder)  # the correct ordering of the states

        if verbose:
            printAssessment(tHMMobj, lin)
            print("True states: ")
            print(true_state_holder)
            print("Viterbi estimated raw states (before state assignment switch): ")
            print(all_states[lin])
            print("State assignment after analysis: ")
            print(tHMMobj.stateAssignment[lin])
            print("Viterbi estimated relative states (after state switch): ")
            print(viterbi_est_holder)
            print("Accuracy: ")
            print(tHMMobj.Accuracy[lin])

    return(tHMMobj.Accuracy, tHMMobj.states, tHMMobj.stateAssignment)


##--------------------getting the accuracy using mutual information ----------------##

def get_mutual_info(tHMMobj, all_states, verbose=True):
    """This fuction calculates the nutual information score between the sequence of
    true states and the sequence that the Viterbi estimates.

    Here a normalized_mutual_info_score function from sklearn.metrics.cluster has been used
    which is commonly used for evaluating clustering accuracy. Using this function helps with
    calculating accuracy regardless of the order and name of labels that the true states and the
    Viterbi outcome have.

    Agrs:
        ---------
        tHMMobj (obj): tree-HMM model as an object
        all_states (matrix): a matrix holding the states assigned by viterbi algorithm as the most likely states.

    Returns:
        ----------
        tHMMobj.Accuracy2 (list): an atribute to tHMMobj which holds the accuracy for each lineage.

    Example usage:

        tHMMobj = tHMM(X, numStates=2, FOM='G') # build the tHMM class with X
        fit(tHMMobj, max_iter=500, verbose=True)
        deltas, state_ptrs = get_leaf_deltas(tHMMobj) # gets the deltas matrix
        get_nonleaf_deltas(tHMMobj, deltas, state_ptrs)
        all_states = Viterbi(tHMMobj, deltas, state_ptrs)

        mutual_info = get_mutual_info(tHMMobj, all_states, verbose = True)
    """

    numStates = tHMMobj.numStates
    tHMMobj.Accuracy2 = []
    tHMMobj.stateAssignment = []
    tHMMobj.states = []

    for lin in range(tHMMobj.numLineages):
        lineage = tHMMobj.population[lin]

        true_state_holder = np.zeros((len(lineage)), dtype=int)
        viterbi_est_holder = np.zeros((len(lineage)), dtype=int)

        for ii, cell in enumerate(lineage):
            true_state_holder[ii] = cell.true_state
            viterbi_est_holder[ii] = all_states[lin][ii]

        tHMMobj.Accuracy2.append(normalized_mutual_info_score(true_state_holder, viterbi_est_holder))

        if verbose:
            printAssessment(tHMMobj, lin)
            print("True states: ")
            print(true_state_holder)
            print("Viterbi estimated states: ")
            print(viterbi_est_holder)
            print("Accuracy: ")
            print(tHMMobj.Accuracy2)
    return tHMMobj.Accuracy2


##-------------------- printing probability matrices of a model ---------------------##

def printAssessment(tHMMobj, lin):
    """This function takes in the tree-HMM model as an object and lineage index, and returns three
    probability matrices of a given model for every lineage including intial probabilities (pi),
    transition probabilities (T), emission probabilities (E).
    """
    print("\n")
    print("Lineage Index: {}".format(lin))
    print("Initial Proabablities: ")
    print(tHMMobj.paramlist[lin]["pi"])
    print("Transition State Matrix: ")
    print(tHMMobj.paramlist[lin]["T"])
    print("Emission Parameters: ")
    print(tHMMobj.paramlist[lin]["E"])
