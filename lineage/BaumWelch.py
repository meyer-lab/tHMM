def get_gamma_roots(self):
    '''need the first gamma terms in the baum welch, which are just the beta values of the root nodes.'''
        gammas = []
        betas = self.betas
        for num in range(len(self.numLineages)): # for each lineage in our Population
            gamma_array = np.zeros( (len(self.numLineages),self.numstates) )
            gamma_array[0,:] = betas[num][0,:]
        gammas.append(gamma_roots)
        return(gammas)
    
    def get_gamma_non_leaves(self):
        '''get the gammas for all other nodes using recursion from the root nodes'''
        for num in range(self.numLineages): # for each lineage in our Population
            lineage = self.population[num] # getting the lineage in the Population by index
            betas = self.betas[num] # instantiating N by K array
            MSD_array = self.MSD[num] # getting the MSD of the respective lineage
            params = self.paramlist[num]
            T = params['T']
            max_level = max_gen(lineage)
            count = 1
            while count < max_level:
                level = get_gen(count, lineage) #get lineage for the gen
                for cell in level:
                    parent_idx = lineage.index(cell)
                    temp = get_daughters(cell) 
                    for n in temp:
                        child_idx = lineage.index(n) 
                        for state_k in range(numstates):
                            beta_child = betas[child_idx, state_k]
                            MSD_array[child_idx, state_k]
                            sum_holder = []
                            for state_j in range(numstates):
                                gamma_parent = gammas[parent_idx, state_j]
                                beta_parent = betas[parent_idx, state_j]
                                sum_holder.append(T*gamma_parent/beta_parent)
                            gamma_state_k = beta_child/MSD_child*sum(sum_holder)
                        gammas[child_idx, gamma_state_k]       
                count += 1
       
    def get_zeta(self, node_parent_m_idx, node_child_n_idx, state_j, state_k, lineage, beta_array, MSD_array, gamma_array, T):
        '''calculates the zeta value that will be used to fill the transition matrix in baum welch'''
        child = lineage.index(node_child_n_idx)
        parent = lineage.index(node_parent_m_idx)
        assert[child.parent = parent]
        assert[parent.isLeft = child or parent.isRight = child]
        beta_child_state_k = beta_array[child, state_k]
        gamma_parent_state_j = gamma_array[parent_state_j]
        MSD_child_state_k = MSD_array[child, state_k]
        beta_parent_child = get_beta_parent_child_prod(self, lineage, beta_array, T, MSD_array, state_j, node_parent_m_idx)
        zeta = beta_child_state_k*T*gamma_parent_state_j/(MSD_child_state_k*beta_parent_child)
        return(zeta)
    
    def fit(self, tolerance = 0.1, verbose = false):
        truth_list = [True] # self.numLineages # create a list with only True for each lineage
        # the following loop will only exit when the entire list is False
        # a value in the list only turns to false when the change in the LL is <= the tolerance
        old_LL_list = [inf] * self.numLineages
        new_LL_list = self.calculate_log_likelihood()
        truth_list = [new_LL_list[lineage] - old_LL_list[lineage] > tolerance for lineage in zip(new_LL_list, old_LL_list)]
        while any(truth_list): # exit the loop 
            old_LL_list = new_LL_list
            MSD = self.get_Marginal_State_Distributions()
            EL = self.get_Emission_Likelihoods()
            NF = self.get_leaf_Normalizing_Factors()
            betas = self.get_beta_leaves()
            self.get_beta_and_NF_nonleaves(betas, NF)
            gammas = self.get_gamma_roots()
            self.get_gamma_non_leaves
            #update
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
            
    
    
    