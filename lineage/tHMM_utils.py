'''Utility and helper functions for recursions and other needs in the tHMM class. This also contains the methods for AIC and accuracy.'''

import itertools
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score


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


def right_censored_Gomp_pdf(tau_or_tauFake, c, scale, fateObserved=True):
    """
    Gives you the likelihood of a right-censored Gompertz distribution.
    See Pg. 14 of The Gompertz distribution and Maximum Likelihood Estimation of its parameters - a revision
    by Adam Lenart
    November 28, 2011

    This is a replacement for scipy.gompertz function to find the Emission Likelihood,
    because at the end of the experiment time,there will be some cells that are still
    alive and have not died or divided, so we don't know their end_time, these cells in
    our data are called right censored. So this distribution is used instead of
    real gompretz distribution, to make the synthesized data more like the distribution.


    p(tau_i | a, b) = [a * exp(b * tau) ^delta_i] * [exp(-a/b * (exp(b * tau_i -1)))]
    here, `firstCoeff` is [a * exp(b * tau) ^delta_i]  and the `secondCoeff` is [exp(-a/b * (exp(b * tau_i -1)))]

    Args:
        ----------
        tau_or_tauFake (float): the cell's lifetime
        c (float): loc of Gompertz (one of the distribution parameters)
        scale (float): scale parameter of Gompertz disribution
        deathObserved (bool): if the cell has died already, it is True, otherwise
        it is False

    Return:
        ----------
        result (float): the multiplication of two coefficients

    """
    b = 1. / scale
    a = c * b

    firstCoeff = a * np.exp(b * tau_or_tauFake)
    if fateObserved:
        pass  # this calculation stays as is if the death is observed (delta_i = 1)
    else:
        firstCoeff = 1.  # this calculation is raised to the power of delta if the death is unobserved (right-censored) (delta_i = 0)

    secondCoeff = np.exp((-1 * a / b) * (np.expm1(b * tau_or_tauFake)))
    # the observation of the cell death has no bearing on the calculation of the second coefficient in the pdf

    result = firstCoeff * secondCoeff
    assert np.isfinite(result), "Your Gompertz right-censored likelihood calculation is returning NaN. Your parameter estimates are likely creating overflow in the likelihood calculations with the following values: \n tau_or_tauFake: {} \n c: {} \n scale: {} b: {} \n a: {} \n fateObserved: {}.".format(tau_or_tauFake, c, scale, b, a, fateObserved)

    return result

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
        AIC_value_holder_rel_0 (list): containing AIC values relative to 0 for each lineage.
        [numStates]*len(AIC_value_holder) (list): a list containing the # of states, repeated # of lineage times.
        AIC_degrees_of_freedom_holder (list): Contains all of the parameters for each of the lineages.


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
    AIC_value_holder = []
    AIC_degrees_of_freedom_holder = []
    for num in range(tHMMobj.numLineages):
        number_of_parameters = 0
        if tHMMobj.keepBern:
            number_of_parameters += 1
        if tHMMobj.FOM == 'G':
            number_of_parameters += 2
        elif tHMMobj.FOM == 'E':
            number_of_parameters += 1

        AIC_degrees_of_freedom = numStates**2 + numStates * number_of_parameters - 1
        AIC_degrees_of_freedom_holder.append(AIC_degrees_of_freedom)
        AIC_value = -2 * LL[num] + 2 * AIC_degrees_of_freedom
        AIC_value_holder.append(AIC_value)

    AIC_value_holder_rel_0 = AIC_value_holder - min(AIC_value_holder)  # this line is to make it so the minimum value is 0
    return(AIC_value_holder_rel_0, [numStates] * len(AIC_value_holder), AIC_degrees_of_freedom_holder)

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

def get_mutual_info(tHMMobj, all_states, verbose = True):
    
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
