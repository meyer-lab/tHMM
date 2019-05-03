'''Generates 4 types of figures: Lineage Length, Number of Lineages in a Popuation, KL Divergence effects, and AIC Calculation. Currently, only the Depth_Two_State_Lineage is used.'''

import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

from .Depth_Two_State_Lineage import Depth_Two_State_Lineage
from ..Analyze import Analyze
from .Matplot_gen import Matplot_gen
from ..tHMM_utils import getAccuracy, getAIC
from ..Lineage_utils import remove_singleton_lineages, remove_unfinished_cells


def Lineage_Length(T_MAS=510, T_2=175, reps=100, MASinitCells=[1], MASlocBern=[0.999], MASbeta=[100], initCells2=[1], locBern2=[0.8], beta2=[20], numStates=2, max_lin_length=300, min_lin_length=5, FOM='E', verbose=False):
    '''This has been modified for an exonential distribution'''

    accuracy_h1 = []  # list of lists of lists
    number_of_cells_h1 = []
    bern_MAS_h1 = []
    bern_2_h1 = []
    betaExp_MAS_h1 = []
    betaExp_2_h1 = []

    for rep in range(reps):
        print('Rep:', rep)
        X, newLineage, masterLineage, subLineage2 = Depth_Two_State_Lineage(T_MAS, MASinitCells, MASlocBern, T_2, initCells2, locBern2, FOM=FOM, betaExp=MASbeta, betaExp2=beta2)

        lives = np.zeros(len(masterLineage))
        for ii, cell in enumerate(masterLineage):
            lives[ii] = cell.tau

        lives2 = np.zeros(len(subLineage2))
        for ii, cell in enumerate(subLineage2):
            lives2[ii] = cell.tau

        lives = lives[~np.isnan(lives)]
        lives2 = lives2[~np.isnan(lives2)]

        (KS, p_val) = stats.ks_2samp(lives, lives2)
        while len(newLineage) > max_lin_length or len(masterLineage) < min_lin_length or (len(newLineage) - len(masterLineage)) < min_lin_length or p_val > 0.02:
            X, newLineage, masterLineage, subLineage2 = Depth_Two_State_Lineage(T_MAS, MASinitCells, MASlocBern, T_2, initCells2, locBern2, FOM=FOM, betaExp=MASbeta, betaExp2=beta2)

            lives = np.zeros(len(masterLineage))
            for ii, cell in enumerate(masterLineage):
                lives[ii] = cell.tau

            lives2 = np.zeros(len(subLineage2))
            for ii, cell in enumerate(subLineage2):
                lives2[ii] = cell.tau

            lives = lives[~np.isnan(lives)]
            lives2 = lives2[~np.isnan(lives2)]

            (KL, p_val) = stats.ks_2samp(lives, lives2)
        #---------------------------------------------------#

        print('X size: {}, masterLineage size: {}, subLineage2 size: {}'.format(len(X),len(masterLineage),len(subLineage2)))
        X = remove_unfinished_cells(X)
        X = remove_singleton_lineages(X)
        _, _, all_states, tHMMobj, _, _ = Analyze(X, numStates)
        print('analyzed')
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
            T = tHMMobj.paramlist[lin]["T"]
            E = tHMMobj.paramlist[lin]["E"]
            pi = tHMMobj.paramlist[lin]["pi"]
            print('accuracy: {}'.format(accuracy))


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
        

    if FOM == 'E':
        data = (number_of_cells_h1, accuracy_h1, bern_MAS_h1, bern_2_h1, MASlocBern, locBern2, MASbeta, beta2, betaExp_MAS_h1, betaExp_2_h1)

    return data


def Lineages_per_Population_Figure(lineage_start=1, lineage_end=4, numStates=2, T_MAS=130, T_2=61, reps=1, MASinitCells=[1], MASlocBern=[0.8], MASbetaExp=[40], initCells2=[1], locBern2=[0.99], betaExp2=[18], max_lin_length=300, min_lin_length=50, verbose=True):
    '''Creates four figures of how accuracy, bernoulli parameter, gomp c, and gomp scale change as the number of lineages in a population are varied'''
    if verbose:
        print('starting')
    lineages = range(lineage_start, lineage_end + 1)
    accuracy_h1 = []  # list of lists of lists
    number_of_cells_h1 = []
    bern_MAS_h1 = []
    bern_2_h1 = []
    betaExp_MAS_h1 = []
    betaExp_2_h1 = []
    lineage_h1 = []

    for lineage_num in lineages:  # a pop with num number of lineages
        accuracy_h2 = []
        number_of_cells_h2 = []
        bern_MAS_h2 = []
        bern_2_h2 = []
        betaExp_MAS_h2 = []
        betaExp_2_h2 = []

        for rep in range(reps):
            X1 = []
            if verbose:
                print('making lineage')

            for num in range(lineage_num):
                X, masterLineage, newLineage = Depth_Two_State_Lineage(T_MAS, MASinitCells, MASlocBern, MASbetaExp, T_2, initCells2, locBern2, betaExp2)
                while len(newLineage) > max_lin_length or len(masterLineage) < min_lin_length or (len(newLineage) - len(masterLineage)) < min_lin_length:
                    X, masterLineage, newLineage = Depth_Two_State_Lineage(T_MAS, MASinitCells, MASlocBern, MASbetaExp, T_2, initCells2, locBern2, betaExp2)
                X1.extend(newLineage)

            X = remove_singleton_lineages(X1)  # this is one single list with a number of lineages equal to what is inputted
            print(len(X))
            _, _, all_states, tHMMobj, _, _ = Analyze(X, numStates)
            accuracy_h3 = []
            number_of_cells_h3 = []
            bern_MAS_h3 = []
            bern_2_h3 = []
            betaExp_MAS_h3 = []
            betaExp_2_h3 = []

            if verbose:
                print('analyzing')
            for lin in range(tHMMobj.numLineages):
                AccuracyPop, _, stateAssignmentPop = getAccuracy(tHMMobj, all_states, verbose=False)
                accuracy = AccuracyPop[lin]
                state_1 = stateAssignmentPop[lin][0]
                state_2 = stateAssignmentPop[lin][1]
                lineage = tHMMobj.population[lin]
                T = tHMMobj.paramlist[lin]["T"]
                E = tHMMobj.paramlist[lin]["E"]
                pi = tHMMobj.paramlist[lin]["pi"]

                accuracy_h3.append(100 * accuracy)
                number_of_cells_h3.append(len(lineage))
                bern_MAS_h3.append(E[state_1, 0])
                bern_2_h3.append(E[state_2, 0])
                betaExp_MAS_h3.append(E[state_1, 1])
                betaExp_2_h3.append(E[state_2, 1])

                if verbose:
                    print('pi', pi)
                    print('T', T)
                    print('E', E)
                    print('accuracy:', accuracy)
                    print('MAS length, 2nd lin length:', len(masterLineage), len(newLineage) - len(masterLineage))

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
        lineage_h1.append(lineage_num)

        if verbose:
            print('Accuracy of', lineage_num, 'is', np.mean(accuracy_h2))

    x = lineage_h1
    data = (x, accuracy_h1, bern_MAS_h1, bern_2_h1, MASlocBern, locBern2, betaExp_MAS_h1, betaExp_2_h1, MASbetaExp, betaExp2)
    return data


def AIC_Figure(T_MAS=130, T_2=61, state1=1, state2=4, reps=1, MASinitCells=[1], MASlocBern=[0.8], MASbetaExp=[40],
               initCells2=[1], locBern2=[0.99], betaExp2=[18], max_lin_length=1500, min_lin_length=100, verbose=False):
    '''Calculates and plots an AIC for all inputted states'''

    states = range(state1, state2 + 1)
    accuracy_h1 = []  # list of lists of lists
    number_of_cells_h1 = []
    AIC_h1 = []
    X, masterLineage, newLineage = Depth_Two_State_Lineage(T_MAS, MASinitCells, MASlocBern, MASbetaExp, T_2, initCells2, locBern2, betaExp2)

    while len(newLineage) > max_lin_length or len(masterLineage) < min_lin_length or (len(newLineage) - len(masterLineage)) < min_lin_length:
        X, masterLineage, newLineage = Depth_Two_State_Lineage(T_MAS, MASinitCells, MASlocBern, MASbetaExp, T_2, initCells2, locBern2, betaExp2)

    if verbose:
        print(len(masterLineage), len(newLineage))

    for numStates in states:  # a pop with num number of lineages
        _, _, all_states, tHMMobj, _, LL = Analyze(X, numStates)
        AccuracyPop, _, _ = getAccuracy(tHMMobj, all_states, verbose=False)
        accuracy_h2 = []
        number_of_cells_h2 = []
        AIC_h2 = []

        for rep in range(reps):
            AIC_value_holder_rel_0 = getAIC(tHMMobj, LL)
            accuracy_h3 = []
            number_of_cells_h3 = []
            AIC_h3 = []

            for lin in range(tHMMobj.numLineages):
                number_of_cells_h3.append(len(tHMMobj.population[lin]))
                accuracy_h3.append(100 * AccuracyPop[lin])
                AIC_h3.append(AIC_value_holder_rel_0[lin])

            number_of_cells_h2.extend(number_of_cells_h3)
            accuracy_h2.append(accuracy_h3)
            AIC_h2.append(AIC_value_holder_rel_0)

        accuracy_h1.extend(accuracy_h2)
        number_of_cells_h1.extend(number_of_cells_h2)
        AIC_h1.extend(AIC_h2)

        if verbose:
            print('h1', number_of_cells_h1)

    x = states
    fig, axs = plt.subplots(nrows=1, ncols=1, sharex=True)
    ax = axs
    ax.errorbar(x, AIC_h1, fmt='o', c='b', marker="*", fillstyle='none')
    ax.set_title('AIC')
    ax.set_xlabel('Number of States')
    ax.set_ylabel('Cost function', rotation=90)
    fig.suptitle('Akaike Information Criterion')
    plt.savefig('TEST_AIC_classification.png')
