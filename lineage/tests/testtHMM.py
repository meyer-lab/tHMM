    
    
    
    #######################
    # tHMM.py tests below #
    #######################

    def test_init_paramlist(self):
        '''
        Make sure paramlist has proper
        labels and sizes.
        '''
        X = remove_unfinished_cells(self.X)
        X = remove_singleton_lineages(X)
        t = tHMM(X, numStates=2)  # build the tHMM class with X
        self.assertEqual(t.paramlist[0]["pi"].shape[0], 2)  # make sure shape is numStates
        self.assertEqual(t.paramlist[0]["T"].shape[0], 2)  # make sure shape is numStates
        self.assertEqual(t.paramlist[0]["T"].shape[1], 2)  # make sure shape is numStates
        self.assertEqual(t.paramlist[0]["E"].shape[0], 2)  # make sure shape is numStates

    def test_get_MSD(self):
        '''
        Calls get_Marginal_State_Distributions and
        ensures the output is of correct data type and
        structure.
        '''
        X = remove_unfinished_cells(self.X)
        X = remove_singleton_lineages(X)
        t = tHMM(X, numStates=2)  # build the tHMM class with X
        MSD = t.get_Marginal_State_Distributions()
        self.assertLessEqual(len(MSD), 50)  # there are <=50 lineages in the population
        for _, MSDlin in enumerate(MSD):
            self.assertGreaterEqual(MSDlin.shape[0], 0)  # at least zero cells in each lineage
            self.assertEqual(MSDlin.shape[1], 2)  # there are 2 states for each cell
            for node_n in range(MSDlin.shape[0]):
                self.assertEqual(sum(MSDlin[node_n, :]), 1)  # the rows should sum to 1

    def test_get_EL(self):
        '''
        Calls get_Emission_Likelihoods and ensures
        the output is of correct data type and structure.
        '''
        X = remove_unfinished_cells(self.X)
        X = remove_singleton_lineages(X)
        t = tHMM(X, numStates=2)  # build the tHMM class with X
        EL = t.get_Emission_Likelihoods()
        self.assertLessEqual(len(EL), 50)  # there are <=50 lineages in the population
        for _, ELlin in enumerate(EL):
            self.assertGreaterEqual(ELlin.shape[0], 0)  # at least zero cells in each lineage
            self.assertEqual(ELlin.shape[1], 2)  # there are 2 states for each cell

    ##################################
    # UpwardRecursion.py tests below #
    ##################################

    def test_get_leaf_NF(self):
        '''
        Calls get_leaf_Normalizing_Factors and
        ensures the output is of correct data type and
        structure.
        '''
        X = remove_unfinished_cells(self.X)
        X = remove_singleton_lineages(X)
        t = tHMM(X, numStates=2)  # build the tHMM class with X
        NF = get_leaf_Normalizing_Factors(t)
        self.assertLessEqual(len(NF), 50)  # there are <=50 lineages in the population
        for _, NFlin in enumerate(NF):
            self.assertGreaterEqual(NFlin.shape[0], 0)  # at least zero cells in each lineage
