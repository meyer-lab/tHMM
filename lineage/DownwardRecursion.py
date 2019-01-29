#to do : make sure everything is aligned correctly
# self should become tHMMobj
# add docstring to the document
# fix linting

from .tHMM_utils import max_gen, get_gen, get_parents_for_level


def get_root_gammas(tHMMobj, betas):
    '''need the first gamma terms in the baum welch, which are just the beta values of the root nodes.'''
    numStates = tHMMobj.numStates
    numLineages = tHMMobj.numLineages

    gammas = []

    for num in range(len(numLineages)): # for each lineage in our Population
        gamma_array = np.zeros( (len(numLineages),numstates) )
        gamma_array[0,:] = betas[num][0,:]
        gammas.append(gamma_roots)

    return(gammas)

def get_gamma_non_leaves(tHMMobj, gammas, betas):
    '''get the gammas for all other nodes using recursion from the root nodes'''
    numStates = tHMMobj.numStates
    numLineages = tHMMobj.numLineages
    population = tHMMobj.population
    paramlist = tHMM.paramlist
    MSD = tHMMobj.MSD
    
    for num in range(numLineages): # for each lineage in our Population
        lineage = population[num] # getting the lineage in the Population by index
        MSD_array = MSD[num] # getting the MSD of the respective lineage
        params = paramlist[num]
        beta_array = betas[num] # instantiating N by K array
        T = params['T']
        
        curr_level = 1
        max_level = max_gen(lineage)
        
        while curr_level < max_level:
            level = get_gen(curr_level, lineage) #get lineage for the gen
            for cell in level:
                parent_idx = lineage.index(cell)
                daughter_idxs_list = get_daughters(cell) 

                for daughter_idx in daughter_idxs_list:
                    child_idx = lineage.index(daughter_idx) 

                    for state_k in range(numStates):
                        beta_child = beta_array[child_idx, state_k]
                        MSD_array[child_idx, state_k]
                        sum_holder = []

                        for state_j in range(numstates):
                            gamma_parent = gammas[num][parent_idx, state_j]
                            beta_parent = beta_array[parent_idx, state_j]
                            sum_holder.append(T*gamma_parent/beta_parent)
                            
                        gamma_state_k = beta_child/MSD_child*sum(sum_holder)
                    gammas[num][child_idx, gamma_state_k]       
            curr_level += 1