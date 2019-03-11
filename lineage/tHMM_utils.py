'''utility and helper functions for recursions and other needs in the tHMM class'''

import numpy as np
import itertools

def max_gen(lineage):
    '''finds the max generation in a lineage'''
    gen_holder = 1
    for cell in lineage:
        if cell.gen > gen_holder:
            gen_holder = cell.gen
    return gen_holder

def get_gen(gen, lineage):
    '''creates a list with all cells in the given generation'''
    first_set = []
    for cell in lineage:
        if cell.gen == gen:
            first_set.append(cell)
    return first_set

def get_parents_for_level(level, lineage):
    """
        Returns a set of all the parents of all the cells in a
        given level/generation. For example this would give you
        all the non-leaf cells in the generation above the one given.
    """
    parent_holder = set() #set makes sure only one index is put in and no overlap
    for cell in level:
        parent_cell = cell.parent
        parent_holder.add(lineage.index(parent_cell))
    return parent_holder

def get_daughters(cell):
    """ Returns a list of the daughters of a given cell. """
    temp = []
    if cell.left:
        temp.append(cell.left)
    if cell.right:
        temp.append(cell.right)
    return temp

def right_censored_Gomp_pdf(tau_or_tauFake, c, scale, deathObserved=True):
    '''
    Gives you the likelihood of a right-censored Gompertz distribution.
    See Pg. 14 of The Gompertz distribution and Maximum Likelihood Estimation of its parameters - a revision
    by Adam Lenart
    November 28, 2011
    '''
    b = 1. / scale
    a = c * b

    firstCoeff = a * np.exp(b*tau_or_tauFake)
    if deathObserved:
        pass # this calculation stays as is if the death is observed (delta_i = 1)
    else:
        firstCoeff = 1. # this calculation is raised to the power of delta if the death is unobserved (right-censored) (delta_i = 0)

    secondCoeff = np.exp((-1*a/b)*((np.exp(b*tau_or_tauFake))-1))
    # the observation of the cell death has no bearing on the calculation of the second coefficient in the pdf

    result = firstCoeff*secondCoeff
    assert np.isfinite(result), "Your Gompertz right-censored likelihood calculation is returning NaN. Your parameter estimates are likely creating overflow in the likelihood calculations."

    return result

def printAssessment(tHMMobj, lin):
    '''Prints the parameters.'''
    print("\n")
    print("Lineage Index: {}".format(lin))
    print("Initial Proabablities: ")
    print(tHMMobj.paramlist[lin]["pi"])
    print("Transition State Matrix: ")
    print(tHMMobj.paramlist[lin]["T"])
    print("Emission Parameters: ")
    print(tHMMobj.paramlist[lin]["E"])
        
def getAccuracy(tHMMobj, all_states, verbose=False):
    '''Gets the accuracy for state assignment per lineage.'''
    numStates = tHMMobj.numStates
    tHMMobj.Accuracy = []
    tHMMobj.stateAssignment = []
    
    for lin in range(tHMMobj.numLineages):
        lineage = tHMMobj.population[lin]
        pi = tHMMobj.paramlist[lin]["pi"] 
        T = tHMMobj.paramlist[lin]["T"]
        E = tHMMobj.paramlist[lin]["E"] 
        
        true_state_holder = np.zeros((len(lineage)), dtype=int)
        viterbi_est_holder = np.zeros((len(lineage)), dtype=int)

        for ii, cell in enumerate(lineage):
            true_state_holder[ii] = cell.true_state
            viterbi_est_holder[ii] = all_states[lin][ii]
            
        permutation_of_states = list(itertools.permutations(range(numStates)))
        temp_acc_holder = []
        for possible_state_assignment in permutation_of_states:
            # gets a list of lists of permutations of state assignments
            print(possible_state_assignment)
            temp_all_states = all_states[lin]
            for state in range(numStates):
                for ii, temp_state in enumerate(temp_all_states):
                    if temp_state == state:
                        temp_all_states[ii] = possible_state_assignment[state]
                    
            common_state_counter = [true_state == temp_vit_state for (true_state,temp_vit_state) in zip(true_state_holder,temp_all_states)]
            print(sum(common_state_counter))
            accuracy = sum(common_state_counter)/len(lineage) # gets the accuracies per possible state assignment
            print(accuracy)
            temp_acc_holder.append(accuracy)
            
        idx_of_max_acc = np.argmax(temp_acc_holder)
        tHMMobj.Accuracy.append(temp_acc_holder[idx_of_max_acc])
        
        tHMMobj.stateAssignment = permutation_of_states[idx_of_max_acc]

        for state in range(numStates):
            for ii,cell_viterbi_state in enumerate(viterbi_est_holder):
                if cell_viterbi_state==state:
                    viterbi_est_holder[ii]= tHMMobj.stateAssignment[state]
            
        if verbose:
            printAssessment(tHMMobj, lin)
            print("True states: ")
            print(true_state_holder)
            print("State assignment after analysis: ")
            print(tHMMobj.stateAssignment)
            print("Viterbi estimated states (after state switch): ")
            print(viterbi_est_holder)
            print("Accuracy: ")
            print(tHMMobj.Accuracy)
