'''Generates 4 types of figures: Lineage Length, Number of Lineages in a Popuation, KL Divergence effects, and AIC Calculation. Currently, only the Depth_Two_State_Lineage is used.'''

import numpy as np
import scipy
import scipy.stats
import logging
from ..Depth_Two_State_Lineage import Depth_Two_State_Lineage
from ..Breadth_Two_State_Lineage import Breadth_Two_State_Lineage
from ..Analyze import Analyze
from ..BaumWelch import fit
from ..tHMM_utils import getAccuracy, getAIC
from ..Lineage_utils import remove_singleton_lineages, remove_unfinished_cells

def KL_per_lineage(T_MAS=500, T_2=100, reps=2, MASinitCells=[1], MASlocBern=[0.8], MASbeta=[80], initCells2=[1], locBern2=[0.99], beta2=[20], numStates=2, max_lin_length=200, min_lin_length=80, FOM='E', verbose=False):
    """Run the KL divergence on emmission likelihoods."""
    # Make the master cells equal to the same thing
    MASlocBern_array, MASbeta_array= [], []
    for i in range(reps): MASlocBern_array.append(MASlocBern), MASbeta_array.append(MASbeta)
    
    # Make the downstream subpopulation a random distribution
    locBern2 = np.random.uniform(0.8, 0.99, size = reps)
    beta2 = np.random.randint(20, 80, size = reps)
    
    #arrays to hold for each rep
    KL_h1 = []
    acc_h1 = []  # list of lists of lists
    cell_h1 = []
    bern_MAS_h1 = []
    bern_2_h1 = []
    betaExp_MAS_h1 = []
    betaExp_2_h1 = []

    for rep in range(reps):  
        X, newLineage, masterLineage, sublineage2 = Depth_Two_State_Lineage(T_MAS, MASinitCells, [MASlocBern_array[rep]], T_2, initCells2, [locBern2[rep]], FOM, [MASbeta_array[rep]], [beta2[rep]])

        
        while len(newLineage) > max_lin_length or len(masterLineage) < min_lin_length or (len(newLineage) - len(masterLineage)) < min_lin_length:
            #re calculate distributions if they are too large, or else model wont run
            if len(newLineage) > max_lin_length or len(masterLineage) < min_lin_length or (len(newLineage) - len(masterLineage)) < min_lin_length:
                locBern2[rep] = [np.random.uniform(0.8, 0.99, size = 1)][0][0]
                beta2[rep] = [np.random.randint(20, 40, size = 1)][0][0]
            #generate new lineage
            X, newLineage, masterLineage, sublineage2 = Depth_Two_State_Lineage(T_MAS, MASinitCells, [MASlocBern_array[rep]], T_2, initCells2, [locBern2[rep]], FOM, [MASbeta_array[rep]], [beta2[rep]])
        logging.info('Repetition Number: {}'.format(rep+1))
        
        X = remove_singleton_lineages(newLineage)
        _, _, all_states, tHMMobj, _, _ = Analyze(X, numStates)
        
        #arrays to hold values for each lineage within the population that the rep made
        KL_h2 = []
        acc_h2 = []
        cell_h2 = []
        bern_MAS_h2 = []
        bern_2_h2 = []
        betaExp_MAS_h2 = []
        betaExp_2_h2 = []

        for lin in range(tHMMobj.numLineages):
            AccuracyPop, _, stateAssignmentPop = getAccuracy(tHMMobj, all_states, verbose=False)
            accuracy = AccuracyPop[lin]
            state_1 = stateAssignmentPop[lin][0]
            state_2 = stateAssignmentPop[lin][1]
            lineage = tHMMobj.population[lin]
            E = tHMMobj.paramlist[lin]["E"]
            EL = tHMMobj.EL[lin]

            KL = scipy.stats.entropy(EL[0,:], EL[1,:]) #this must be made for more than 1 state, such that all state other than the one of interest (ie p) are binned into the q distribution           
            
            KL_h2.append(KL)
            acc_h2.append(100 * accuracy)
            cell_h2.append(len(lineage))
            bern_MAS_h2.append(E[state_1, 0])
            bern_2_h2.append(E[state_2, 0])
            betaExp_MAS_h2.append(E[state_1, 1])
            betaExp_2_h2.append(E[state_2, 1])


            if verbose:
                logging.info('pi', pi)
                logging.info('T', T)
                logging.info('E', E)
                logging.info('accuracy:', accuracy)
                logging.info('KL:', KL)
                logging.info('MAS length, 2nd lin length:', len(masterLineage), len(newLineage) - len(masterLineage))

        KL_h1.extend(KL_h2)
        acc_h1.extend(acc_h2)
        cell_h1.extend(cell_h2)
        bern_MAS_h1.extend(bern_MAS_h2)
        bern_2_h1.extend(bern_2_h2)
        betaExp_MAS_h1.extend(betaExp_MAS_h2)
        betaExp_2_h1.extend(betaExp_2_h2)

    logging.info("x shape: {} and acc_h1 shape: {}".format(KL_h1, acc_h1))
    data = (KL_h1, acc_h1, bern_MAS_h1, bern_2_h1, MASlocBern, locBern2, MASbeta, beta2, betaExp_MAS_h1, betaExp_2_h1)
    return data


def Lineage_Length(T_MAS=500, T_2=100, reps=10, MASinitCells=[1], MASlocBern=[0.8], MASbeta=[80], initCells2=[1],
                   locBern2=[0.99], beta2=[20], numStates=2, max_lin_length=300, min_lin_length=5, FOM='E', verbose=False, switchT=False, AIC=False, numState_start=1, numState_end=3):
    '''This has been modified for an exponential distribution'''

    accuracy_h1 = []  # list of lists of lists
    number_of_cells_h1 = []
    bern_MAS_h1 = []
    bern_2_h1 = []
    betaExp_MAS_h1 = []
    betaExp_2_h1 = []
    AIC_h1 = {str(numState): [] for numState in range(numState_start, numState_end+1)} 
    LL_h1 = {str(numState): [] for numState in range(numState_start, numState_end+1)}
    for rep in range(reps):
        print(rep)
        logging.info('Rep:', rep)
        # Make the lineage type
        if not switchT:
            X, newLineage, masterLineage, subLineage2 = Depth_Two_State_Lineage(T_MAS, MASinitCells, MASlocBern, T_2, initCells2, locBern2, FOM=FOM, betaExp=MASbeta, betaExp2=beta2)
            while len(newLineage) > max_lin_length or len(masterLineage) < min_lin_length or (len(newLineage) - len(masterLineage)) < min_lin_length:
                X, newLineage, masterLineage, subLineage2 = Depth_Two_State_Lineage(T_MAS, MASinitCells, MASlocBern, T_2, initCells2, locBern2, FOM=FOM, betaExp=MASbeta, betaExp2=beta2)

        elif switchT:
            X, newLineage, masterLineage, subLineage2 = Breadth_Two_State_Lineage(
                experimentTime=T_MAS + T_2, initCells=MASinitCells, locBern=MASlocBern, betaExp=MASbeta, switchT=T_MAS, bern2=locBern2, betaExp2=beta2, FOM=FOM, verbose=False)
            while len(newLineage) > max_lin_length or len(masterLineage) < min_lin_length or (len(newLineage) - len(masterLineage)) < min_lin_length:
                X, newLineage, masterLineage, subLineage2 = Breadth_Two_State_Lineage(
                    experimentTime=T_MAS + T_2, initCells=MASinitCells, locBern=MASlocBern, betaExp=MASbeta, switchT=T_MAS, bern2=locBern2, betaExp2=beta2, FOM=FOM, verbose=False)
    
        X = remove_unfinished_cells(X)
        X = remove_singleton_lineages(X)
        logging.info('X size: {}, masterLineage size: {}, subLineage2 size: {}'.format(len(X), len(masterLineage), len(subLineage2)))
        
        #Call function for AIC 
        if AIC:
            AICval = []
            LLval = []
            for numState in range(numState_start, numState_end+1):
                logging.info(f'numState:{numState}')
                _, _, all_states, tHMMobj, _, _ = Analyze(X, numStates=numState)
                tHMMobj, NF, betas, gammas, LL = fit(tHMMobj, max_iter=100, verbose=False)
                AIC_ls, LL_ls, AIC_degrees_of_freedom = getAIC(tHMMobj, LL)
                AICval.append(sum(AIC_ls)) # make numstate be a single value not an array of a value
                LLval.append(sum(LL_ls))
            AIC_rel_0 = AICval #make aic plot to be relative to the lowest value 
            LL_rel_0 = LLval #make aic plot to be relative to the lowest value 
            for ii, numState in enumerate(range(numState_start, numState_end+1)):
                AIC_h1[str(numState)].append(AIC_rel_0[ii])
                LL_h1[str(numState)].append(LL_rel_0[ii]) # take total AIC across all lineages for this numstate
        else:
            _, _, all_states, tHMMobj, _, _ = Analyze(X, numStates)
            accuracy_h2 = []
            number_of_cells_h2 = []
            bern_MAS_h2 = []
            bern_2_h2 = []
            betaExp_MAS_h2 = []
            betaExp_2_h2 = []


            for lin in range(tHMMobj.numLineages):
                AccuracyPop, _, stateAssignmentPop = getAccuracy(tHMMobj, all_states, verbose=False)
                accuracy = AccuracyPop[lin]

                state_1 = stateAssignmentPop[lin][0]
                state_2 = stateAssignmentPop[lin][1]
                lineage = tHMMobj.population[lin]
                E = tHMMobj.paramlist[lin]["E"]
                logging.info('accuracy: {}'.format(accuracy))

                accuracy_h2.append(accuracy * 100)
                number_of_cells_h2.append(len(lineage))
                bern_MAS_h2.append(E[state_1, 0])
                bern_2_h2.append(E[state_2, 0])
                if FOM == 'E':
                    betaExp_MAS_h2.append(E[state_1, 1])
                    betaExp_2_h2.append(E[state_2, 1])

            accuracy_h1.extend(accuracy_h2)
            number_of_cells_h1.extend(number_of_cells_h2)
            bern_MAS_h1.extend(bern_MAS_h2)
            bern_2_h1.extend(bern_2_h2)
            betaExp_MAS_h1.extend(betaExp_MAS_h2)
            betaExp_2_h1.extend(betaExp_2_h2)

    if AIC:
        numstates_ls = []
        AIC_mean, AIC_std = [], []
        LL_mean, LL_std = [], []
        for ii, numState in enumerate(range(numState_start, numState_end+1)):
            numstates_ls.append(numState)
            AIC_mean.append(np.mean(AIC_h1[str(numState)]))
            AIC_std.append(np.std(AIC_h1[str(numState)]))
            LL_mean.append(np.mean(LL_h1[str(numState)]))
            LL_std.append(np.std(LL_h1[str(numState)]))
        data = (numstates_ls, AIC_mean, AIC_std, LL_mean, LL_std)
    else:
        data = (number_of_cells_h1, accuracy_h1, bern_MAS_h1, bern_2_h1, MASlocBern, locBern2, MASbeta, beta2, betaExp_MAS_h1, betaExp_2_h1)
    
    return data


def Lineages_per_Population_Figure(lineage_start=1, lineage_end=2, numStates=2, T_MAS=500, T_2=100, reps=1, MASinitCells=[1], MASlocBern=[0.8], MASbeta=[80], initCells2=[1], locBern2=[0.999], beta2=[20], max_lin_length=300, min_lin_length=5, FOM='E', verbose=True, switchT=True):
    '''Creates four figures of how accuracy, bernoulli parameter, gomp c, and gomp scale change as the number of lineages in a population are varied'''
    if verbose:
        logging.info('starting')
    lineages = range(lineage_start, lineage_end + 1)
    accuracy_h1 = []  # list of lists of lists
    number_of_cells_h1 = []
    bern_MAS_h1 = []
    bern_2_h1 = []
    betaExp_MAS_h1 = []
    betaExp_2_h1 = []
    numb_of_lineage_h1 = []

    X1 = []
    for lineage_num in lineages:  # a pop with num number of lineages
        accuracy_h2 = []
        number_of_cells_h2 = []
        bern_MAS_h2 = []
        bern_2_h2 = []
        betaExp_MAS_h2 = []
        betaExp_2_h2 = []

        for rep in range(reps):

            if verbose:
                logging.info('making lineage')
            for num in range(lineage_num):

                if not switchT:
                    X, newLineage, masterLineage, subLineage2 = Depth_Two_State_Lineage(T_MAS, MASinitCells, MASlocBern, T_2, initCells2, locBern2, FOM=FOM, betaExp=MASbeta, betaExp2=beta2)
                    while len(newLineage) > max_lin_length or len(masterLineage) < min_lin_length or (len(newLineage) - len(masterLineage)) < min_lin_length:
                        X, newLineage, masterLineage, subLineage2 = Depth_Two_State_Lineage(T_MAS, MASinitCells, MASlocBern, T_2, initCells2, locBern2, FOM=FOM, betaExp=MASbeta, betaExp2=beta2)

                elif switchT:
                    X, newLineage, masterLineage, subLineage2 = Breadth_Two_State_Lineage(
                        experimentTime=T_MAS + T_2, initCells=MASinitCells, locBern=MASlocBern, betaExp=MASbeta, switchT=T_MAS, bern2=locBern2, betaExp2=beta2, FOM=FOM, verbose=False)
                    while len(newLineage) > max_lin_length or len(masterLineage) < min_lin_length or (len(newLineage) - len(masterLineage)) < min_lin_length:
                        X, newLineage, masterLineage, subLineage2 = Breadth_Two_State_Lineage(
                            experimentTime=T_MAS + T_2, initCells=MASinitCells, locBern=MASlocBern, betaExp=MASbeta, switchT=T_MAS, bern2=locBern2, betaExp2=beta2, FOM=FOM, verbose=False)

                X = remove_unfinished_cells(X)
                X = remove_singleton_lineages(X)
                X1.extend(newLineage)

            X1 = remove_unfinished_cells(X1)
            X1 = remove_singleton_lineages(X1)  # this is one single list with a number of lineages equal to what is inputted
            logging.info(len(X1))
            _, _, all_states, tHMMobj, _, _ = Analyze(X1, numStates)
            accuracy_h3 = []
            number_of_cells_h3 = []
            bern_MAS_h3 = []
            bern_2_h3 = []
            betaExp_MAS_h3 = []
            betaExp_2_h3 = []

            if verbose:
                logging.info('analyzing')
            for lin in range(tHMMobj.numLineages):
                AccuracyPop, _, stateAssignmentPop = getAccuracy(tHMMobj, all_states, verbose=False)
                accuracy = AccuracyPop[lin]
                state_1 = stateAssignmentPop[lin][0]
                state_2 = stateAssignmentPop[lin][1]
                lineage = tHMMobj.population[lin]
                E = tHMMobj.paramlist[lin]["E"]

                accuracy_h3.append(100 * accuracy)
                number_of_cells_h3.append(len(lineage))
                bern_MAS_h3.append(E[state_1, 0])
                bern_2_h3.append(E[state_2, 0])
                betaExp_MAS_h3.append(E[state_1, 1])
                betaExp_2_h3.append(E[state_2, 1])

            accuracy_h2.extend(accuracy_h3)
            number_of_cells_h2.extend(number_of_cells_h3)
            bern_MAS_h2.extend(bern_MAS_h3)
            bern_2_h2.extend(bern_2_h3)
            betaExp_MAS_h2.extend(betaExp_MAS_h3)
            betaExp_2_h2.extend(betaExp_2_h3)

        accuracy_h1.append(np.mean(accuracy_h2))
        number_of_cells_h1.extend(number_of_cells_h2)
        bern_MAS_h1.append(np.mean(bern_MAS_h2))
        bern_2_h1.append(np.mean(bern_2_h2))
        betaExp_MAS_h1.append(np.mean(betaExp_MAS_h2))
        betaExp_2_h1.append(np.mean(betaExp_2_h2))
        numb_of_lineage_h1.append(lineage_num)

        if verbose:
            logging.info('Accuracy of', lineage_num, 'is', np.mean(accuracy_h2))

    data = (numb_of_lineage_h1, accuracy_h1, bern_MAS_h1, bern_2_h1, MASlocBern, locBern2, betaExp_MAS_h1, betaExp_2_h1, MASbeta, beta2)

    return data
