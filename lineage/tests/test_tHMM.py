''' Unit test file. Contains tests for the functions in Lineage_utils.py, tHMM_utils.py, UpwardRecursion.py, DownwardRecursion.py, BaumWelch.py'''
import unittest
import numpy as np

from ..BaumWelch import fit
from ..DownwardRecursion import get_root_gammas, get_nonroot_gammas
from ..Viterbi import get_leaf_deltas, get_nonleaf_deltas, Viterbi
from ..UpwardRecursion import get_leaf_Normalizing_Factors, get_leaf_betas, get_nonleaf_NF_and_betas
from ..tHMM import tHMM
from ..tHMM_utils import max_gen, get_gen, get_parents_for_level, getAccuracy, get_mutual_info
from ..Lineage_utils import remove_singleton_lineages, remove_unfinished_cells, get_numLineages, init_Population, generatePopulationWithTime as gpt

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
        cell21.right = None  # reset the right pointer to None to effectively delete the cell from the lineage
        self.lineage3 = [cell20, cell21, cell22, self.cell23]

        # example where lineage is just one cell
        self.cell30 = CellNode(startT=0, linID=4)
        self.lineage4 = [self.cell30]

        # create a common population to use in all tests
        experimentTime = 50.
        initCells = [50]  # there should be 50 lineages b/c there are 50 initial cells
        locBern = [0.8]
        betaExp = [40]
        self.X = gpt(experimentTime, initCells, locBern, betaExp)  # generate a population

        initCells = [40, 10]  # there should be around 50 lineages b/c there are 50 initial cells
        locBern = [0.999, 0.8]
        betaExp = [40, 50]
        self.X2 = gpt(experimentTime, initCells, locBern, betaExp)

    ################################
    # Lineage_utils.py tests below #
    ################################
    
    def test_remove_unfinished_cells(self):
        '''
        Checks to see that cells with a NaN of tau
        are eliminated from a population list.
        '''
        experimentTime = 100.
        initCells = [50, 50]
        locBern = [0.6, 0.8]
        betaExp = [40, 50]
        X = gpt(experimentTime, initCells, locBern, betaExp) # generate a population
        X = remove_unfinished_cells(X) # remove unfinished cells
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
        self.assertLessEqual(numLin, 50)  # call func

        # case where the lineages follow different parameter sets
        experimentTime = 50.
        initCells = [50, 42, 8]  # there should be 100 lineages b/c there are 100 initial cells
        locBern = [0.6, 0.8, 0.7]
        betaExp = [40, 50, 45]
        X = gpt(experimentTime, initCells, locBern, betaExp)  # generate a population
        numLin = get_numLineages(X)
        self.assertLessEqual(numLin, 100)  # call func

    def test_init_Population(self):
        '''
        Tests that populations are lists
        of lineages and each cell in a
        lineage has the correct linID.
        '''
        numLin = get_numLineages(self.X)
        pop = init_Population(self.X, numLin)
        self.assertEqual(len(pop), numLin)  # len(pop) corresponds to the number of lineages

        # check that all cells in a lineage have same linID
        for lineage in pop:  # for each lineage
            prev_cell_linID = lineage[0].linID
            for cell in lineage:  # for each cell in said lineage
                self.assertEqual(prev_cell_linID, cell.linID)  # linID should correspond with i
                prev_cell_linID = cell.linID
                # sometimes lineages have one cell and those are removed
                # possibly a better way to test

    #####################
    # tHMM_utils. below #
    #####################

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
        self.assertEqual(max_gen(self.lineage4), 1)  # lineage 4 is just one cell

    def test_get_gen(self):
        '''
        Checks to make sure get_gen of a
        certain lineage returns the proper
        cells.
        '''
        temp1 = get_gen(3, self.lineage1)  # bottom row of lineage row
        self.assertIn(self.cell4, temp1)
        self.assertIn(self.cell5, temp1)
        self.assertIn(self.cell6, temp1)
        self.assertIn(self.cell7, temp1)
        self.assertEqual(len(temp1), 4)

        temp2 = get_gen(2, self.lineage2)  # 2nd generation of lineage 2
        self.assertIn(self.cell11, temp2)
        self.assertIn(self.cell12, temp2)
        self.assertEqual(len(temp2), 2)

        temp3 = get_gen(3, self.lineage2)  # 3rd generation of lineage 2
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

    def test_getAccuracy(self):
        """
        checks whether the accuracy is in the range
        """
        numStates = 2

        switchT = 200
        experimentTime = switchT + 150
        initCells = [1]
        locBern = [0.99999999999]
        betaExp1 = [75]
        bern2 = [0.6]
        betaExp2 = [50]

        LINEAGE = gpt(experimentTime, initCells, locBern, betaExp1, switchT, bern2, betaExp2, FOM='E')
        LINEAGE = remove_unfinished_cells(LINEAGE)
        LINEAGE = remove_singleton_lineages(LINEAGE)
        while len(LINEAGE) <= 5:
            LINEAGE = gpt(experimentTime, initCells, locBern, betaExp1, switchT, bern2, betaExp2, FOM='E')
            LINEAGE = remove_unfinished_cells(LINEAGE)
            LINEAGE = remove_singleton_lineages(LINEAGE)

        X = LINEAGE
        t = tHMM(X, numStates=2)
        fit(t, max_iter=500, verbose=True)

        deltas, state_ptrs = get_leaf_deltas(t)  # gets the deltas matrix
        get_nonleaf_deltas(t, deltas, state_ptrs)
        all_states = Viterbi(t, deltas, state_ptrs)

        t.Accuracy, t.states, t.stateAssignment = getAccuracy(t, all_states, verbose=False)
        check_acc = all(1.0 >= x >= 0.0 for x in t.Accuracy)
        self.assertTrue(check_acc)

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

    ##########################
    # Viterbi.py tests below #
    ##########################

    def test_viterbi(self):
        '''
        Builds the tHMM class and calls
        the Viterbi function to find
        the optimal hidden states.
        '''
        X = remove_unfinished_cells(self.X)
        X = remove_singleton_lineages(X)
        t = tHMM(X, numStates=2)  # build the tHMM class with X
        deltas, state_ptrs = get_leaf_deltas(t)  # gets the deltas matrix
        self.assertLessEqual(len(deltas), 50)  # there are <=50 lineages in X
        self.assertLessEqual(len(state_ptrs), 50)  # there are <=50 lineages in X
        get_nonleaf_deltas(t, deltas, state_ptrs)
        self.assertLessEqual(len(deltas), 50)  # there are <=50 lineages in X
        self.assertLessEqual(len(state_ptrs), 50)  # there are <=50 lineages in X
        all_states = Viterbi(t, deltas, state_ptrs)
        self.assertLessEqual(len(all_states), 50)  # there are <=50 lineages in X

    def test_viterbi2(self):
        '''
        Builds the tHMM class and calls
        the Viterbi function to find
        the optimal hidden states.
        Now trying to see if altering the parameters
        gives one different optimal state
        trees.
        '''
        X = remove_unfinished_cells(self.X)
        X = remove_singleton_lineages(X)
        numStates = 2
        t = tHMM(X, numStates=numStates)  # build the tHMM class with X
        fake_param_list = []
        numLineages = t.numLineages
        temp_params = {"pi": np.ones((numStates), dtype=int),  # inital state distributions [K] initialized to 1/K
                       "T": np.ones((numStates, numStates), dtype=int) / (numStates),  # state transition matrix [KxK] initialized to 1/K
                       "E": np.ones((numStates, 2))}  # sequence of emission likelihood distribution parameters [Kx2]
        temp_params["pi"][1] = 0  # the hidden state for the second node should always be 1
        to_state_one = np.zeros((numStates, numStates), dtype=int)
        to_state_one[:, 1] = np.ones((numStates), dtype=int)
        temp_params["T"] = to_state_one  # should always end up in state 1 regardless of previous state
        # since transition matrix is a dependent matrix (0 is now a trivial state)
        temp_params["E"][:, 0] *= 0.5  # initializing all Bernoulli p parameters to 0.5
        temp_params["E"][:, 1] *= 50  # initializing all Gompoertz s(cale) parameters to 50

        for lineage_num in range(numLineages):  # for each lineage in our population
            fake_param_list.append(temp_params.copy())  # create a new dictionary holding the parameters and append it
            assert len(fake_param_list) == lineage_num + 1
        t.paramlist = fake_param_list
        t.MSD = t.get_Marginal_State_Distributions()  # rerun these with new parameters
        t.EL = t.get_Emission_Likelihoods()  # rerun these with new parameters
        # run Viterbi with new parameter list
        deltas, state_ptrs = get_leaf_deltas(t)  # gets the deltas matrix
        self.assertLessEqual(len(deltas), 50)  # there are <=50 lineages in X
        self.assertLessEqual(len(state_ptrs), 50)  # there are <=50 lineages in X
        get_nonleaf_deltas(t, deltas, state_ptrs)
        self.assertLessEqual(len(deltas), 50)  # there are <=50 lineages in X
        self.assertLessEqual(len(state_ptrs), 50)  # there are <=50 lineages in X
        all_states = Viterbi(t, deltas, state_ptrs)
        self.assertLessEqual(len(all_states), 50)  # there are <=50 lineages in X
        for num in range(numLineages):
            curr_all_states = all_states[num]
            self.assertEqual(curr_all_states[0], 0)  # the first state should always be 0
            all_ones = curr_all_states[1:]  # get all the items in the list except the first item
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
        X = remove_unfinished_cells(self.X2)
        X = remove_singleton_lineages(X)
        numStates = 2
        t = tHMM(X, numStates=numStates, FOM='E')  # build the tHMM class with X

        fake_param_list = []
        numLineages = t.numLineages
        temp_params = {"pi": np.ones((numStates), dtype=float) / (numStates),  # inital state distributions [K] initialized to 1/K
                       "T": np.eye(2, dtype=int),  # state transition matrix [KxK] initialized to identity (no transitions)
                       # should always end up in state 1 regardless of previous state
                       "E": np.ones((numStates, 3))}  # sequence of emission likelihood distribution parameters [Kx2]

        temp_params["pi"][0] = 4 / 5  # the population is distributed as such 2/5 is of state 0
        temp_params["pi"][1] = 1 / 5  # state 1 occurs 3/5 of the time

        temp_params["E"][0, 0] *= 0.999  # initializing all Bernoulli p parameters to 0.5
        temp_params["E"][0, 1] *= 40  # initializing all Gompoertz s(cale) parameters to 50

        temp_params["E"][1, 0] *= 0.6  # initializing all Bernoulli p parameters to 0.5
        temp_params["E"][1, 1] *= 50  # initializing all Gompoertz s(cale) parameters to 50

        for lineage_num in range(numLineages):  # for each lineage in our population
            fake_param_list.append(temp_params.copy())  # create a new dictionary holding the parameters and append it
            assert len(fake_param_list) == lineage_num + 1
        t.paramlist = fake_param_list

        t.MSD = t.get_Marginal_State_Distributions()  # rerun these with new parameters
        t.EL = t.get_Emission_Likelihoods()  # rerun these with new parameters
        # run Viterbi with new parameter list
        deltas, state_ptrs = get_leaf_deltas(t)  # gets the deltas matrix
        self.assertLessEqual(len(deltas), 50)  # there are <=50 lineages in X
        self.assertLessEqual(len(state_ptrs), 50)  # there are <=50 lineages in X
        get_nonleaf_deltas(t, deltas, state_ptrs)
        self.assertLessEqual(len(deltas), 50)  # there are <=50 lineages in X
        self.assertLessEqual(len(state_ptrs), 50)  # there are <=50 lineages in X
        all_states = Viterbi(t, deltas, state_ptrs)
        self.assertLessEqual(len(all_states), 50)  # there are <=50 lineages in X
        num_of_zeros = 0  # counts how many lineages were all of the 0 states
        num_of_ones = 0  # counts how many lineages were all of the 1 states
        for num in range(numLineages):
            curr_all_states = all_states[num]
            if curr_all_states[0] == 0:
                all_zeros = curr_all_states
                self.assertFalse(all(all_zeros))
                # this should be true since the homogenous lineage is all of state 0
                num_of_zeros += 1
            else:
                all_ones = curr_all_states
                self.assertTrue(all(all_ones))
                # this should be true since the homogenous lineage is all of state 1
                num_of_ones += 1
        self.assertGreater(num_of_zeros, num_of_ones)
        # there should be a greater number of lineages with all zeros than all ones as hidden states

    ####################################
    # DownwardRecursion.py tests below #
    ####################################

    def test_get_gammas(self):
        '''
        Calls gamma related functions and
        ensures the output is of correct data type and
        structure.
        '''
        X = remove_unfinished_cells(self.X)
        X = remove_singleton_lineages(self.X)
        numStates = 2
        tHMMobj = tHMM(X, numStates=numStates)  # build the tHMM class with X
        NF = get_leaf_Normalizing_Factors(tHMMobj)
        betas = get_leaf_betas(tHMMobj, NF)
        get_nonleaf_NF_and_betas(tHMMobj, NF, betas)
        gammas = get_root_gammas(tHMMobj, betas)
        get_nonroot_gammas(tHMMobj, gammas, betas)
        self.assertLessEqual(len(gammas), 50)  # there are <=50 lineages in the population
        for ii, gammasLin in enumerate(gammas):
            self.assertGreaterEqual(gammasLin.shape[0], 0)  # at least zero cells in each lineage
            for state_k in range(numStates):
                self.assertEqual(gammasLin[0, state_k], betas[ii][0, state_k])

    ############################
    # BaumWelch.py tests below #
    ############################

    def test_Baum_Welch_4(self):
        ''' one state exponential estimation'''
        numStates = 1

        experimentTime = 250
        initCells = [1]
        locBern = [0.99999999999]
        betaExp = [75]

        LINEAGE = gpt(experimentTime, initCells, locBern, betaExp=betaExp, FOM='E')
        LINEAGE = remove_unfinished_cells(LINEAGE)
        LINEAGE = remove_singleton_lineages(LINEAGE)
        while len(LINEAGE) <= 10:
            LINEAGE = gpt(experimentTime, initCells, locBern, betaExp=betaExp, FOM='E')
            LINEAGE = remove_unfinished_cells(LINEAGE)
            LINEAGE = remove_singleton_lineages(LINEAGE)

        X = LINEAGE
        tHMMobj = tHMM(X, numStates=numStates, FOM='E')  # build the tHMM class with X
        fit(tHMMobj, max_iter=100, verbose=False)

        deltas, state_ptrs = get_leaf_deltas(tHMMobj)  # gets the deltas matrix
        get_nonleaf_deltas(tHMMobj, deltas, state_ptrs)
        all_states = Viterbi(tHMMobj, deltas, state_ptrs)
        getAccuracy(tHMMobj, all_states, verbose=True)
        get_mutual_info(tHMMobj, all_states, verbose=True)

    def test_Baum_Welch_5(self):
        '''two state exponential estimation. creating a heterogeneous tree'''

        numStates = 2

        switchT = 300
        experimentTime = switchT + 150
        initCells = [1]
        locBern = [0.99999999999]
        betaExp = [75]
        bern2 = [0.6]
        betaExp2 = [25]

        LINEAGE = gpt(experimentTime, initCells, locBern, betaExp, switchT, bern2, betaExp2=betaExp2, FOM='E')

        while len(LINEAGE) <= 10:
            LINEAGE = gpt(experimentTime, initCells, locBern, betaExp, switchT, bern2, betaExp2=betaExp2, FOM='E')

        X = LINEAGE
        X = remove_unfinished_cells(X)
        X = remove_singleton_lineages(X)
        tHMMobj = tHMM(X, numStates=numStates, FOM='E')  # build the tHMM class with X
        fit(tHMMobj, max_iter=100, verbose=False)

        deltas, state_ptrs = get_leaf_deltas(tHMMobj)  # gets the deltas matrix
        get_nonleaf_deltas(tHMMobj, deltas, state_ptrs)
        all_states = Viterbi(tHMMobj, deltas, state_ptrs)
        getAccuracy(tHMMobj, all_states, verbose=True)
        get_mutual_info(tHMMobj, all_states, verbose=True)
