#to do : make sure everything is aligned correctly
# self should become tHMMobj
# add docstring to the document
# fix linting

from .UpwardRecursion import get_leaf_Normalizing_Factors, get_leaf_betas, get_nonleaf_NF_and_betas, calculate_log_likelihood


def zeta_parent_child_func(node_parent_m_idx, node_child_n_idx, state_j, state_k, lineage, beta_array, MSD_array, gamma_array, T):
    '''calculates the zeta value that will be used to fill the transition matrix in baum welch'''
    child = lineage.index(node_child_n_idx)
    parent = lineage.index(node_parent_m_idx)

    assert(child.parent is parent)
    assert(parent.isLeft is child or parent.isRight is child)

    beta_child_state_k = beta_array[child, state_k]
    gamma_parent_state_j = gamma_array[parent_state_j]
    MSD_child_state_k = MSD_array[child, state_k]
    beta_parent_child = get_beta_parent_child_prod(self, lineage, beta_array, T, MSD_array, state_j, node_parent_m_idx)
    zeta = beta_child_state_k*T[state_j,state_k]*gamma_parent_state_j/(MSD_child_state_k*beta_parent_child)
    return(zeta)

def get_all_zetas(parent_state_j, child_state_k,, lineage, beta_array, MSD_array, gamma_array, T):
    '''sum of the list of all the zeta parent child for all the parent cells for a given state transition pair'''
    curr_level = 1
    max_level = max_gen(lineage)
    
    while curr_level < max_level:
        level = get_gen(curr_level, lineage) #get lineage for the gen
        holder = []
        
        for cell in level:
            parent_idx = lineage.index(cell)
            daughter_idxs_list = get_daughters(cell)
            
            for daughter_idx in daughter_idxs_list:
                child_idx = lineage.index(daughter_idx) 
                holder.append(get_zeta(node_parent_m_idx=parent_idx,
                                       node_child_n_idx=child_idx,
                                       state_j=parent_state_j,
                                       state_k=child_state_k,
                                       lineage=lineage,
                                       beta_array=beta_array
                                       MSD_array=MSD_array,
                                       gamma_array=gamma_array,
                                       T=T))
        curr_level += 1
    return sum(holder)

def fit(tHMMobj, tolerance = 0.1, verbose = False):
    
    old_LL_list = [-inf] * numLineages
    new_LL_list = calculate_log_likelihood(tHMMobj, NF)
    truth_list = [new_LL_list[lineage] - old_LL_list[lineage] > tolerance for lineage in zip(new_LL_list, old_LL_list)]
    
    while any(truth_list): # exit the loop 
        old_LL_list = new_LL_list
        
        numLineages = tHMMobj.numLineages
        numStates = tHMMobj.numStates
        population = tHMMobj.population
        
        # calculation loop
        NF = get_leaf_Normalizing_Factors(tHMMobj)
        betas = get_leaf_betas(tHMMobj, NF)
        get_nonleaf_NF_and_betas(tHMMobj, NF, betas)
        gammas = get_root_gammas(tHMMobj, betas)
        get_nonleaf_gammas(tHMMobj, gammas, betas)
        
        # update loop        
        for num in range(numLineages):
            lineage = population[num]
            beta_array = betas[num]
            MSD_array = tHMMobj.MSD[num]
            gamma_array = gammas[num]
            tHMMobj.paramlist[num]["pi"] = gamma_array[0,:]
            for state_j in numStates:
                denom = sum(gamma_array[:-1,state_j]) # gammas [NxK]
                for state_k in numStates:
                    numer = get_all_zetas(parent_state_j=state_j,
                                             child_state_k=state_k,
                                             lineage=lineage, 
                                             beta_array=beta_array, 
                                             MSD_array=MSD_array,
                                             gamma_array=gamma_array,
                                             T= tHMMobj.paramlist[num]["T"] )
                    tHMMobj.paramlist[num]["T"][state_j,state_k] = numer/denom
                    
        # tolerance checking
        new_LL_list = []
        NF = get_leaf_Normalizing_Factors(tHMMobj)
        for num in range(numLineages):
            NF_array = NF[num]
            log_NF_array = np.log(NF_array)
            ll_per_num = sum(log_NF_array)
            new_LL.append(ll_per_num) 
            
        truth_list = [new_LL_list[lineage] - old_LL_list[lineage] > tolerance for lineage in zip(new_LL_list, old_LL_list)]

                    
            
            

    