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
    