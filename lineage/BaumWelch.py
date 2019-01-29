#to do : make sure everything is aligned correctly
# self should become tHMMobj
# add docstring to the document
# fix linting

from .UpwardRecursion import get_leaf_Normalizing_Factors, get_leaf_betas, get_nonleaf_NF_and_betas, calculate_log_likelihood


def get_zeta(node_parent_m_idx, node_child_n_idx, state_j, state_k, lineage, beta_array, MSD_array, gamma_array, T):
    '''calculates the zeta value that will be used to fill the transition matrix in baum welch'''
    child = lineage.index(node_child_n_idx)
    parent = lineage.index(node_parent_m_idx)

    assert(child.parent is parent)
    assert(parent.isLeft is child or parent.isRight is child)

    beta_child_state_k = beta_array[child, state_k]
    gamma_parent_state_j = gamma_array[parent_state_j]
    MSD_child_state_k = MSD_array[child, state_k]
    beta_parent_child = get_beta_parent_child_prod(self, lineage, beta_array, T, MSD_array, state_j, node_parent_m_idx)
    zeta = beta_child_state_k*T*gamma_parent_state_j/(MSD_child_state_k*beta_parent_child)
    return(zeta)

def fit(tHMMobj, tolerance = 0.1, verbose = false):

        numLineages = tHMMobj.numLineages
        numStates = tHMMobj.numStates
        population = tHMMobj.population
        NF = get_leaf_Normalizing_Factors(tHMMobj)
        betas = get_leaf_betas(tHMMobj, NF)
        get_nonleaf_NF_and_betas(tHMMobj, NF, betas)
        new_LL_list = calculate_log_likelihood(tHMMobj, NF)
        gammas = get_root_gammas(tHMMobj, betas)
        get_nonleaf_gammas(tHMMobj, gammas, betas)
        
        # pi,  updates
        for num in range(numLineages):
            tHMMobj.paramlist[num]["pi"] = gammas[num][0,:]
            for state_j in numStates:
                denom = sum(gammas[num][:,state_j]) # gammas [NxK]
                for state_k in numStates:
                    numer = []
                    for cell in population[num]:
                        temp_zeta = get_zeta # helper function to get all zetas
                    
            
            
        
        
        
        #old_LL_list = [inf] * numLineages
        
        truth_list = [new_LL_list[lineage] - old_LL_list[lineage] > tolerance for lineage in zip(new_LL_list, old_LL_list)]
        
        while any(truth_list): # exit the loop 
            old_LL_list = new_LL_list
            # re run recursions
            # with new parameters
            
            MSD = get_Marginal_State_Distributions()
            EL = self.get_Emission_Likelihoods()
            NF = self.get_leaf_Normalizing_Factors()
            betas = self.get_beta_leaves()
            self.get_beta_and_NF_nonleaves(betas, NF)
            gammas = self.get_gamma_roots()
            self.get_gamma_non_leaves
           

            # update
            pi = gammas[0,:]
            for state_j in self.numstates:
                for state_k in self.numstates:
                    denom = sum(gammas[:, state_j])
                    for cell in range(len(lineage))
                    node_parent_m_idx = 
                    T[state_j, state_k] = sum(get_zeta(self, node_parent_m_idx, node_child_n_idx, state_j, state_k, lineage, beta_array, MSD_array, gamma_array, T)/sum(gammas[:, state_j])
            
            
            
            
            
            
            
            
        
        
        
        
        
        
        
        
        pi_update = []
        for num in range(self.numLineages): # for each lineage in our Population
            old_LL = inf #caclulates starting log likelihood before while loop
            new_LL = 0
            while new_LL - old_LL > tolerance:
                old_LL = new_LL
                
                get_Marginal_State_Distributions
                
                lineage = self.population[num] # getting the lineage in the Population by index
                betas = self.betas[num] # instantiating N by K array
                MSD_array = self.MSD[num] # getting the MSD of the respective lineage
                params = self.paramlist[num]
                T = params['T']
                pi_update[num] = gammas[0,:]
                
                
                
                ##new _LL
                new_LL = []
                 # for each lineage in our Population
                    lineage = self.population # getting the lineage in the Population by index
                    NF_array = self.NF # getting the NF of the respective lineage
                    log_NF_array = np.log(NF_array)
                    ll_per_num = sum(log_NF_array)
                    new_LL.append(ll_per_num) 
            
    
    
    