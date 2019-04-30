'''Re-calculates the tHMM parameters of pi, T, and emissions using Baum Welch'''
import numpy as np

from .tHMM_utils import max_gen, get_gen, get_daughters
from .DownwardRecursion import get_root_gammas, get_nonroot_gammas
from .UpwardRecursion import get_leaf_Normalizing_Factors, get_leaf_betas, get_nonleaf_NF_and_betas, calculate_log_likelihood, beta_parent_child_func
from .Lineage_utils import bernoulliParameterEstimatorAnalytical, gompertzAnalytical, exponentialAnalytical


def zeta_parent_child_func(node_parent_m_idx, node_child_n_idx, state_j, state_k, lineage, beta_array, MSD_array, gamma_array, T):
    '''calculates the zeta value that will be used to fill the transition matrix in baum welch'''

    # check the child-parent relationship
    assert lineage[node_child_n_idx].parent is lineage[node_parent_m_idx], "Something wrong with your parent-daughter linkage when trying to use the zeta-related functions... Check again that your lineage is constructed clearly."
    # if the child-parent relationship is correct, then the child must
    assert lineage[node_child_n_idx].isChild(), "Something wrong with your parent-daughter linkage when trying to use the zeta-related functions... Check again that your lineage is constructed clearly."
    # either be the left daughter or the right daughter

    beta_child_state_k = beta_array[node_child_n_idx, state_k]
    gamma_parent_state_j = gamma_array[node_parent_m_idx, state_j]
    MSD_child_state_k = MSD_array[node_child_n_idx, state_k]
    numStates = MSD_array.shape[1]
    also_numStates = gamma_array.shape[1]
    also_also_numStates = beta_array.shape[1]
    assert numStates == also_numStates == also_also_numStates, "Your numStates (number of states) and the dimensions of your arrays are mistmatched! Please check your instantiation of the tHMM object."
    beta_parent_child_state_j = beta_parent_child_func(numStates=numStates,
                                                       lineage=lineage,
                                                       beta_array=beta_array,
                                                       T=T,
                                                       MSD_array=MSD_array,
                                                       state_j=state_j,
                                                       node_parent_m_idx=node_parent_m_idx,
                                                       node_child_n_idx=node_child_n_idx)

    zeta = beta_child_state_k * T[state_j, state_k] * gamma_parent_state_j / (MSD_child_state_k * beta_parent_child_state_j)
    return zeta


def get_all_gammas(lineage, gamma_array_at_state_j):
    '''sum of the list of all the gamma parent child for all the parent child relationships'''
    curr_level = 1
    max_level = max_gen(lineage)
    holder = []
    while curr_level < max_level:  # get all the gammas but not the ones at the last level
        level = get_gen(curr_level, lineage)  # get lineage for the gen
        for cell in level:
            if not cell.isLeaf():
                cell_idx = lineage.index(cell)
                holder.append(gamma_array_at_state_j[cell_idx])

        curr_level += 1
    return sum(holder)


def get_all_zetas(parent_state_j, child_state_k, lineage, beta_array, MSD_array, gamma_array, T):
    '''sum of the list of all the zeta parent child for all the parent cells for a given state transition pair'''
    curr_level = 1
    max_level = max_gen(lineage)
    holder = []
    while curr_level < max_level:
        level = get_gen(curr_level, lineage)  # get lineage for the gen

        for cell in level:
            parent_idx = lineage.index(cell)
            daughter_idxs_list = get_daughters(cell)

            for daughter_idx in daughter_idxs_list:
                child_idx = lineage.index(daughter_idx)
                holder.append(zeta_parent_child_func(node_parent_m_idx=parent_idx,
                                                     node_child_n_idx=child_idx,
                                                     state_j=parent_state_j,
                                                     state_k=child_state_k,
                                                     lineage=lineage,
                                                     beta_array=beta_array,
                                                     MSD_array=MSD_array,
                                                     gamma_array=gamma_array,
                                                     T=T))
        curr_level += 1
    return sum(holder)


def fit(tHMMobj, tolerance=1e-10, max_iter=100, verbose=False):
    '''Runs the tHMM function through Baum Welch fitting'''
    numLineages = tHMMobj.numLineages
    numStates = tHMMobj.numStates
    population = tHMMobj.population

    # first E step

    NF = get_leaf_Normalizing_Factors(tHMMobj)
    betas = get_leaf_betas(tHMMobj, NF)
    get_nonleaf_NF_and_betas(tHMMobj, NF, betas)
    gammas = get_root_gammas(tHMMobj, betas)
    get_nonroot_gammas(tHMMobj, gammas, betas)

    # first stopping condition check

    old_LL_list = [-np.inf] * numLineages
    new_LL_list = calculate_log_likelihood(tHMMobj, NF)
    truth_list = []
    for lineage_iter in range(len(new_LL_list)):
        truth_list.append(abs(new_LL_list[lineage_iter] - old_LL_list[lineage_iter]) > tolerance)
    go = any(truth_list)

    count = 0
    while go:  # exit the loop

        count += 1

        old_LL_list = new_LL_list

        # code for grouping all states in cell lineages
        cell_groups = {}
        for state in range(numStates):
            cell_groups[str(state)] = []
        
        for num in range(numLineages):
            if not truth_list[num]:
                break
            lineage = population[num]
            beta_array = betas[num]
            MSD_array = tHMMobj.MSD[num]
            gamma_array = gammas[num]
            tHMMobj.paramlist[num]["pi"] = gamma_array[0, :]
            T_holder = np.zeros((numStates, numStates), dtype=float)
            for state_j in range(numStates):
                gamma_array_at_state_j = gamma_array[:, state_j]
                denom = get_all_gammas(lineage, gamma_array_at_state_j)
                for state_k in range(numStates):
                    numer = get_all_zetas(parent_state_j=state_j,
                                          child_state_k=state_k,
                                          lineage=lineage,
                                          beta_array=beta_array,
                                          MSD_array=MSD_array,
                                          gamma_array=gamma_array,
                                          T=tHMMobj.paramlist[num]["T"])
                    entry = numer / denom
                    T_holder[state_j, state_k] = entry

            row_sums = T_holder.sum(axis=1)
            T_new = T_holder / row_sums[:, np.newaxis]
            tHMMobj.paramlist[num]["T"] = T_new

            max_state_holder = [] #a list the size of lineage, that contains max state for each cell
            for ii, cell in enumerate(lineage):
                assert lineage[ii] is cell
                max_state_holder.append(np.argmax(gammas[num][ii, :])) #says which state is maximal

            #this bins the cells by lineage to the population cell lists
            for ii, state in enumerate(max_state_holder):
                cell_groups[str(state)].append(lineage[ii])

        #after iterating through each lineage, do the population wide E calculation
        global_params = {}
        for state_j in range(numStates):
            cells = cell_groups[str(state_j)] #this array has the correct cells classified per group
            global_params['B' + str(state_j)] = bernoulliParameterEstimatorAnalytical(cells) #list of cells
            global_params['E' + str(state_j)] = exponentialAnalytical(cells)
            
        #now go through each lineage and replace with the new E
        for num in range(numLineages):
            for state in range(numStates):
                #assigns the global state to the lineage-specific state assignment
                tHMMobj.paramlist[num]["E"][state, 0] = global_params['B' + str(state)]
                if tHMMobj.FOM == 'E':
                    tHMMobj.paramlist[num]["E"][state, 1] = global_params['E' + str(state)]

        tHMMobj.MSD = tHMMobj.get_Marginal_State_Distributions()
        tHMMobj.EL = tHMMobj.get_Emission_Likelihoods()

        NF = get_leaf_Normalizing_Factors(tHMMobj)
        betas = get_leaf_betas(tHMMobj, NF)
        get_nonleaf_NF_and_betas(tHMMobj, NF, betas)
        gammas = get_root_gammas(tHMMobj, betas)
        get_nonroot_gammas(tHMMobj, gammas, betas)

        # tolerance checking
        new_LL_list = calculate_log_likelihood(tHMMobj, NF)

        # if verbose:
        #print("Average Log-Likelihood across all lineages: {}".format(np.mean(new_LL_list)))

        for lineage_iter in range(len(new_LL_list)):
            calculation = abs(new_LL_list[lineage_iter] - old_LL_list[lineage_iter])
            truth_list[lineage_iter] = (calculation > tolerance)
        go = any(truth_list)

        if count > max_iter:
            if verbose:
                print("Max iteration of {} steps achieved. Exiting Baum-Welch EM while loop.".format(max_iter))
            break
    return(tHMMobj, NF, betas, gammas, new_LL_list)
