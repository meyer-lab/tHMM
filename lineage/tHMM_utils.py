'''utility and helper functions for recursions and other needs in the tHMM class'''

import numpy as np

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

def printAssessment(tHMMobj):
    for num in range(tHMMobj.numLineages):
        print("\n")
        print("Lineage Index: {}".format(num))
        print("Initial Proabablities: ")
        print(tHMMobj.paramlist[num]["pi"])
        print("Transition State Matrix: ")
        print(tHMMobj.paramlist[num]["T"])
        print("Emission Parameters: ")
        print(tHMMobj.paramlist[num]["E"])
        
def getAccuracy(tHMMobj, lin, all_states):
    lineage = tHMMobj.population[lin]
    numStates = tHMMobj.numStates
    pi = tHMMobj.paramlist[lin]["pi"] 
    T = tHMMobj.paramlist[lin]["T"]
    E = tHMMobj.paramlist[lin]["E"] 
    
    #assign state 1 and state 2   
    if E[0,0] > E[1,0]:
        state_1 = 0
        state_0 = 1
    elif E[1,0] > E[0,0]:
        state_1 = 1
        state_0 = 0
        
    wrong_counter = 0  
    
    trues = []
    viterbi_est = []
    for cell in range(len(lineage)):
        cell_state = lineage[cell].true_state
        viterbi_state = all_states[lin][cell]
        trues.append(cell_state)
        viterbi_est.append(viterbi_state)
        
        if cell_state == 0:
            if viterbi_state == state_0:
                pass
            else:
                wrong_counter += 1
        elif cell_state == 1:
            if viterbi_state == state_1:
                pass
            else:
                wrong_counter += 1           
                        
    accuracy = (len(lineage) - wrong)/len(lineage) #must be fixed for more than 1 lineage   
    print('trues', trues)
    print('viterbi',viterbi_est)
    return(T,E,pi,state_0,state_1,accuracy,lineage)