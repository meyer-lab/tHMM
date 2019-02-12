''' Unit test file. '''
import unittest
import numpy as np

# from ..BaumWelch import fit
from ..DownwardRecursion import get_root_gammas, get_nonroot_gammas
from ..Viterbi import get_leaf_deltas, get_nonleaf_deltas, Viterbi
from ..UpwardRecursion import get_leaf_Normalizing_Factors, get_leaf_betas, get_nonleaf_NF_and_betas
from ..tHMM import tHMM
from ..tHMM_utils import max_gen, get_gen, get_parents_for_level
from ..Lineage_utils import remove_NaNs, get_numLineages, init_Population
from ..Lineage import generatePopulationWithTime as gpt
from ..CellNode import CellNode

class TestModel(unittest.TestCase):
    ''' Here are the unit tests.'''
    def setUp(self):
        '''
        Small trees that can be used to run unit tests and
        check for edge cases. Some cells are labeled as self
        so they can be accessed in unit tests for comparisons.
        '''
        # 3 level tree with no gaps
        cell1 = CellNode(startT=0, linID=1)
        cell2, cell3 = cell1.divide(10)
        self.cell4, self.cell5 = cell2.divide(15)
        self.cell6, self.cell7 = cell3.divide(20)
        self.lineage1 = [cell1, cell2, cell3, self.cell4, self.cell5, self.cell6, self.cell7]

        # 3 level tree where only one of the 2nd generation cells divides
        cell10 = CellNode(startT=0, linID=2)
        self.cell11, self.cell12 = cell10.divide(10)
        self.cell13, self.cell14 = self.cell11.divide(15)
        self.lineage2 = [cell10, self.cell11, self.cell12, self.cell13, self.cell14]

        # 3 level tree where one of the 2nd generation cells doesn't divide and the other only has one daughter cell
        cell20 = CellNode(startT=0, linID=3)
        cell21, cell22 = cell20.divide(10)
        self.cell23, _ = cell21.divide(15)
        cell21.right = None # reset the right pointer to None to effectively delete the cell from the lineage
        self.lineage3 = [cell20, cell21, cell22, self.cell23]

        # example where lineage is just one cell
        self.cell30 = CellNode(startT=0, linID=4)
        self.lineage4 = [self.cell30]

        # create a common population to use in all tests
        experimentTime = 50
        initCells = [50] # there should be 50 lineages b/c there are 50 initial cells
        locBern = [0.8]
        cGom = [2]
        scaleGom = [40]
        self.X = gpt(experimentTime, initCells, locBern, cGom, scaleGom) # generate a population

        initCells = [40, 10] # there should be around 50 lineages b/c there are 50 initial cells
        locBern = [0.999, 0.8]
        cGom = [2, 3]
        scaleGom = [40, 50]
        self.X2 = gpt(experimentTime, initCells, locBern, cGom, scaleGom)
        
        
        #initCells = [1] # there should be around 50 lineages b/c there are 50 initial cells
        #locBern = [0.999]
        #cGom = [2]
        #scaleGom = [40]
        #self.X3 = gpt(experimentTime, initCells, locBern, cGom, scaleGom)
        

    ################################
    # Lineage_utils.py tests below #
    ################################
    def test_remove_NaNs(self):
        '''
        Checks to see that cells with a NaN of tau
        are eliminated from a population list.
        '''
        experimentTime = 100.
        initCells = [50, 50]
        locBern = [0.6, 0.8]
        cGom = [2, 0.5]
        scaleGom = [40, 50]
        X = gpt(experimentTime, initCells, locBern, cGom, scaleGom) # generate a population
        X = remove_NaNs(X) # remove unfinished cells
        num_NAN = 0
        for cell in X:
            if cell.isUnfinished():
                num_NAN += 1

        self.assertEqual(num_NAN, 0) # there should be no unfinished cells left

    def test_get_numLineages(self):
        '''
        Checks to see that the initial number
        of cells created is the number of lineages.
        '''
        numLin = get_numLineages(self.X)
        self.assertLessEqual(numLin, 50) # call func

        # case where the lineages follow different parameter sets
        experimentTime = 50.
        initCells = [50, 42, 8] # there should be 100 lineages b/c there are 100 initial cells
        locBern = [0.6, 0.8, 0.7]
        cGom = [2, 0.5, 1]
        scaleGom = [40, 50, 45]
        X = gpt(experimentTime, initCells, locBern, cGom, scaleGom) # generate a population
        numLin = get_numLineages(X)
        self.assertLessEqual(numLin, 100) # call func

    def test_init_Population(self):
        '''
        Tests that populations are lists
        of lineages and each cell in a
        lineage has the correct linID.
        '''
        numLin = get_numLineages(self.X)
        pop = init_Population(self.X, numLin)
        self.assertEqual(len(pop), numLin) # len(pop) corresponds to the number of lineages

        # check that all cells in a lineage have same linID
        for lineage in pop: # for each lineage
            prev_cell_linID = lineage[0].linID
            for cell in lineage: # for each cell in said lineage
                self.assertEqual(prev_cell_linID, cell.linID) # linID should correspond with i
                prev_cell_linID = cell.linID
                # sometimes lineages have one cell and those are removed
                # possibly a better way to test

    ############################
    # tHMM_utils.pytests below #
    ############################

    def test_max_gen(self):
        '''
        Calls lineages 1 through 4 and ensures
        that the maximimum number of generations
        is correct in each case.
        '''
        # lineages 1-3 have 3 levels/generations
        self.assertEqual(max_gen(self.lineage1), 3)
        self.assertEqual(max_gen(self.lineage2), 3)
        self.assertEqual(max_gen(self.lineage3), 3)
        self.assertEqual(max_gen(self.lineage4), 1) # lineage 4 is just one cell

    def test_get_gen(self):
        '''
        Checks to make sure get_gen of a
        certain lineage returns the proper
        cells.
        '''
        temp1 = get_gen(3, self.lineage1) # bottom row of lineage row
        self.assertIn(self.cell4, temp1)
        self.assertIn(self.cell5, temp1)
        self.assertIn(self.cell6, temp1)
        self.assertIn(self.cell7, temp1)
        self.assertEqual(len(temp1), 4)

        temp2 = get_gen(2, self.lineage2) # 2nd generation of lineage 2
        self.assertIn(self.cell11, temp2)
        self.assertIn(self.cell12, temp2)
        self.assertEqual(len(temp2), 2)

        temp3 = get_gen(3, self.lineage2) # 3rd generation of lineage 2
        self.assertIn(self.cell13, temp3)
        self.assertIn(self.cell14, temp3)
        self.assertEqual(len(temp3), 2)

        temp4 = get_gen(3, self.lineage3)
        self.assertIn(self.cell23, temp4)
        self.assertEqual(len(temp4), 1)

        temp5 = get_gen(1, self.lineage4)
        self.assertIn(self.cell30, temp5)
        self.assertEqual(len(temp5), 1)

    def test_get_parents_for_level(self):
        '''
        Make sure the proper parents
        are returned of a specific level.
        '''
        level = get_gen(3, self.lineage1)
        temp1 = get_parents_for_level(level, self.lineage1)
        self.assertEqual(temp1, {1, 2})

    #######################
    # tHMM.py tests below #
    #######################

    def test_init_paramlist(self):
        '''
        Make sure paramlist has proper
        labels and sizes.
        '''
        X = remove_NaNs(self.X)
        t = tHMM(X, numStates=2) # build the tHMM class with X
        self.assertEqual(t.paramlist[0]["pi"].shape[0], 2) # make sure shape is numStates
        self.assertEqual(t.paramlist[0]["T"].shape[0], 2) # make sure shape is numStates
        self.assertEqual(t.paramlist[0]["T"].shape[1], 2) # make sure shape is numStates
        self.assertEqual(t.paramlist[0]["E"].shape[0], 2) # make sure shape is numStates

    def test_get_MSD(self):
        '''
        Calls get_Marginal_State_Distributions and
        ensures the output is of correct data type and
        structure.
        '''
        X = remove_NaNs(self.X)
        t = tHMM(X, numStates=2) # build the tHMM class with X
        MSD = t.get_Marginal_State_Distributions()
        self.assertLessEqual(len(MSD), 50) # there are <=50 lineages in the population
        for ii in range(len(MSD)):
            self.assertGreaterEqual(MSD[ii].shape[0], 0) # at least zero cells in each lineage
            self.assertEqual(MSD[ii].shape[1], 2) # there are 2 states for each cell
            for node_n in range(MSD[ii].shape[0]):
                self.assertEqual(sum(MSD[ii][node_n,:]), 1) # the rows should sum to 1

    def test_get_EL(self):
        '''
        Calls get_Emission_Likelihoods and ensures
        the output is of correct data type and structure.
        '''
        X = remove_NaNs(self.X)
        t = tHMM(X, numStates=2) # build the tHMM class with X
        EL = t.get_Emission_Likelihoods()
        self.assertLessEqual(len(EL), 50) # there are <=50 lineages in the population
        for ii in range(len(EL)):
            self.assertGreaterEqual(EL[ii].shape[0], 0) # at least zero cells in each lineage
            self.assertEqual(EL[ii].shape[1], 2) # there are 2 states for each cell

    ##################################
    # UpwardRecursion.py tests below #
    ##################################

    def test_get_leaf_NF(self):
        '''
        Calls get_leaf_Normalizing_Factors and
        ensures the output is of correct data type and
        structure.
        '''
        X = remove_NaNs(self.X)
        t = tHMM(X, numStates=2) # build the tHMM class with X
        NF = get_leaf_Normalizing_Factors(t)
        self.assertLessEqual(len(NF), 50) # there are <=50 lineages in the population
        for ii in range(len(NF)):
            self.assertGreaterEqual(NF[ii].shape[0], 0) # at least zero cells in each lineage

    ##########################
    # Viterbi.py tests below #
    ##########################

    def test_viterbi(self):
        '''
        Builds the tHMM class and calls
        the Viterbi function to find
        the optimal hidden states.
        '''
        X = remove_NaNs(self.X)
        t = tHMM(X, numStates=2) # build the tHMM class with X
        deltas, state_ptrs = get_leaf_deltas(t) # gets the deltas matrix
        self.assertLessEqual(len(deltas), 50) # there are <=50 lineages in X
        self.assertLessEqual(len(state_ptrs), 50) # there are <=50 lineages in X
        get_nonleaf_deltas(t, deltas, state_ptrs)
        self.assertLessEqual(len(deltas), 50) # there are <=50 lineages in X
        self.assertLessEqual(len(state_ptrs), 50) # there are <=50 lineages in X
        all_states = Viterbi(t, deltas, state_ptrs)
        self.assertLessEqual(len(all_states), 50) # there are <=50 lineages in X

    def test_viterbi2(self):
        '''
        Builds the tHMM class and calls
        the Viterbi function to find
        the optimal hidden states.
        Now trying to see if altering the parameters
        gives one different optimal state
        trees.
        '''
        X = remove_NaNs(self.X)
        numStates=2
        t = tHMM(X, numStates=numStates) # build the tHMM class with X
        fake_param_list = []
        numLineages = t.numLineages
        temp_params = {"pi": np.ones((numStates), dtype=int)/(numStates), # inital state distributions [K] initialized to 1/K
                       "T": np.ones((numStates, numStates), dtype=int)/(numStates), # state transition matrix [KxK] initialized to 1/K
                       "E": np.ones((numStates, 3))} # sequence of emission likelihood distribution parameters [Kx3]
        temp_params["pi"][1] = 0 # the hidden state for the second node should always be 1
        to_state_one = np.zeros((numStates, numStates), dtype=int)
        to_state_one[:,1] = np.ones((numStates), dtype=int)
        temp_params["T"] = to_state_one # should always end up in state 1 regardless of previous state
        # since transition matrix is a dependent matrix (0 is now a trivial state)
        temp_params["E"][:,0] *= 0.5 # initializing all Bernoulli p parameters to 0.5
        temp_params["E"][:,1] *= 2 # initializing all Gompertz c parameters to 2
        temp_params["E"][:,2] *= 50 # initializing all Gompoertz s(cale) parameters to 50

        for lineage_num in range(numLineages): # for each lineage in our population
            fake_param_list.append(temp_params.copy()) # create a new dictionary holding the parameters and append it
            assert(len(fake_param_list) == lineage_num+1)
        t.paramlist = fake_param_list
        t.MSD = t.get_Marginal_State_Distributions() # rerun these with new parameters
        t.EL = t.get_Emission_Likelihoods() # rerun these with new parameters
        # run Viterbi with new parameter list
        deltas, state_ptrs = get_leaf_deltas(t) # gets the deltas matrix
        self.assertLessEqual(len(deltas), 50) # there are <=50 lineages in X
        self.assertLessEqual(len(state_ptrs), 50) # there are <=50 lineages in X
        get_nonleaf_deltas(t, deltas, state_ptrs)
        self.assertLessEqual(len(deltas), 50) # there are <=50 lineages in X
        self.assertLessEqual(len(state_ptrs), 50) # there are <=50 lineages in X
        all_states = Viterbi(t, deltas, state_ptrs)
        self.assertLessEqual(len(all_states), 50) # there are <=50 lineages in X
        for num in range(numLineages):
            curr_all_states = all_states[num]
            self.assertEqual(curr_all_states[0], 0) # the first state should always be 0
            all_ones = curr_all_states[1:] # get all the items in the list except the first item
            # this list should now be all ones since everything will always transition to 1
            self.assertTrue(all(all_ones))

    def test_viterbi3(self):
        '''
        Builds the tHMM class and calls
        the Viterbi function to find
        the optimal hidden states.
        Now trying to see if altering the parameters
        gives one different optimal state
        trees. For this example, we have two
        homogeneous populations. Using parameter sets that
        describe those homogenous populations.
        '''
        X = remove_NaNs(self.X2)
        numStates=2
        t = tHMM(X, numStates=numStates) # build the tHMM class with X

        fake_param_list = []
        numLineages = t.numLineages
        temp_params = {"pi": np.ones((numStates), dtype=float)/(numStates), # inital state distributions [K] initialized to 1/K
                       "T": np.eye(2, dtype=int), # state transition matrix [KxK] initialized to identity (no transitions)
                       # should always end up in state 1 regardless of previous state
                       "E": np.ones((numStates, 3))} # sequence of emission likelihood distribution parameters [Kx3]

        temp_params["pi"][0] = 2/5 # the population is distributed as such 2/5 is of state 0
        temp_params["pi"][1] = 3/5 # state 1 occurs 3/5 of the time

        temp_params["E"][0,0] *= 0.999 # initializing all Bernoulli p parameters to 0.5
        temp_params["E"][0,1] *= 2 # initializing all Gompertz c parameters to 2
        temp_params["E"][0,2] *= 40 # initializing all Gompoertz s(cale) parameters to 50

        temp_params["E"][1,0] *= 0.6 # initializing all Bernoulli p parameters to 0.5
        temp_params["E"][1,1] *= 3 # initializing all Gompertz c parameters to 2
        temp_params["E"][1,2] *= 50 # initializing all Gompoertz s(cale) parameters to 50

        for lineage_num in range(numLineages): # for each lineage in our population
            fake_param_list.append(temp_params.copy()) # create a new dictionary holding the parameters and append it
            assert(len(fake_param_list) == lineage_num+1)
        t.paramlist = fake_param_list

        t.MSD = t.get_Marginal_State_Distributions() # rerun these with new parameters
        t.EL = t.get_Emission_Likelihoods() # rerun these with new parameters
        # run Viterbi with new parameter list
        deltas, state_ptrs = get_leaf_deltas(t) # gets the deltas matrix
        self.assertLessEqual(len(deltas), 50) # there are <=50 lineages in X
        self.assertLessEqual(len(state_ptrs), 50) # there are <=50 lineages in X
        get_nonleaf_deltas(t, deltas, state_ptrs)
        self.assertLessEqual(len(deltas), 50) # there are <=50 lineages in X
        self.assertLessEqual(len(state_ptrs), 50) # there are <=50 lineages in X
        all_states = Viterbi(t, deltas, state_ptrs)
        self.assertLessEqual(len(all_states), 50) # there are <=50 lineages in X
        num_of_zeros = 0 # counts how many lineages were all of the 0 states
        num_of_ones = 0 # counts how many lineages were all of the 1 states
        for num in range(numLineages):
            curr_all_states = all_states[num]
            if curr_all_states[0] == 0:
                all_zeros = curr_all_states
                self.assertFalse(all(all_zeros))
                # this should be true since the homogenous lineage is all of state 0
                num_of_zeros+=1
            else:
                all_ones = curr_all_states
                self.assertTrue(all(all_ones))
                # this should be true since the homogenous lineage is all of state 1
                num_of_ones+=1
        self.assertGreater(num_of_zeros,num_of_ones)
        # there should be a greater number of lineages with all ones than all zeros as hidden states
        
    ####################################
    # DownwardRecursion.py tests below #
    ####################################

    def test_get_gammas(self):
        '''
        Calls gamma related functions and
        ensures the output is of correct data type and
        structure.
        '''
        X = remove_NaNs(self.X)
        numStates = 2
        tHMMobj = tHMM(X, numStates=numStates) # build the tHMM class with X
        NF = get_leaf_Normalizing_Factors(tHMMobj)
        betas = get_leaf_betas(tHMMobj, NF)
        get_nonleaf_NF_and_betas(tHMMobj, NF, betas)
        gammas = get_root_gammas(tHMMobj, betas)
        get_nonroot_gammas(tHMMobj, gammas, betas)
        self.assertLessEqual(len(gammas), 50) # there are <=50 lineages in the population
        for ii in range(len(gammas)):
            self.assertGreaterEqual(gammas[ii].shape[0], 0) # at least zero cells in each lineage
            for state_k in range(numStates):
                self.assertEqual(gammas[ii][0,state_k],betas[ii][0,state_k])
                
    ############################
    # BaumWelch.py tests below #
    ############################

    #def test_Baum_Welch_(self):
    #    X = remove_NaNs(self.X2)
    #    numStates = 2
    #    tHMMobj = tHMM(X, numStates=numStates) # build the tHMM class with X
    #    fit(tHMMobj, verbose=True)
       
    #def test_Baum_Welch_multipletimes(self):
    #    X = remove_NaNs(self.X3)
    #    reps = 10
    #    numStates = 2
    #    bern = np.zeros(reps)
    #    c = np.zeros(reps)
    #    scale = np.zeros(reps)
    #    for num in range(reps):
    #        
    #        tHMMobj = tHMM(X, numStates=numStates) # build the tHMM class with X
    #        fit(tHMMobj, verbose=False)
    #        
    #        diag = np.diagonal(tHMMobj.paramlist[0]["T"])
    #        chosen_state = np.argmax(diag)

    #        bern[num] = (tHMMobj.paramlist[0]["E"][chosen_state,0])
    #        print("\nRun {} Bernoulli p: {}".format(num, bern[num]))
    #        c[num] = tHMMobj.paramlist[0]["E"][chosen_state,1]
    #        print("Run {} Gompertz c: {}".format(num, c[num]))
    #        scale[num] = tHMMobj.paramlist[0]["E"][chosen_state,2]
    #        print("Run {} Gompertz scale: {}".format(num, scale[num]))
    #        print("Run {} Initial Probabilities: ".format(num))
    #        print(tHMMobj.paramlist[0]["pi"])
    #        print("Run {} Transition Matrix: ".format(num))
    #        print(tHMMobj.paramlist[0]["T"])
    #        print("Expected Bernoulli p: {}".format(0.999))
    #    print('Bernoulli p mean: {} and SD: {}'.format(np.mean(bern), np.std(bern)))
    #    print("Expected Gompertz c: {}".format(2))
    #    print('Gompertz c mean: {} and SD: {}'.format(np.mean(c), np.std(c)))
    #    print("Expected Gompertz scale: {}".format(40))
    #   print('Gompertz scale mean: {} and SD: {}'.format(np.mean(scale), np.std(scale)))
