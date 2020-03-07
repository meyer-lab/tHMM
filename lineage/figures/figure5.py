"""
This creates Figure 5.
"""
from .figureCommon import subplotLabel, getSetup


def makeFigure():
<<<<<<< HEAD
    """ makes figure 5 """
    # Get list of axis objects
    ax, f = getSetup((7, 6), (3, 4))
=======
    """
    Makes figure 11.
    """
    ax, f = getSetup((21, 6), (1, 3))

    figure_maker(ax, 0, *AIC_increased_cells1())
    figure_maker(ax, 1, *AIC_increased_cells2())
    figure_maker(ax, 2, *AIC_increased_cells3())

    return f


def AIC_increased_cells1():
    """
    Calculates accuracy and parameter estimation by increasing the number of cells in a lineage for a one-state model.
    """

    # pi: the initial probability vector
    pi = np.ones(2) / 2.0

    # T: transition probability matrix
    T = np.ones((2, 2)) / 2.0

    # bern, gamma_a, gamma_scale
    state_obj0 = StateDistribution(0.99, 20, 5)
    state_obj1 = StateDistribution(0.99, 20, 5)
    E = [state_obj0, state_obj1]

    desired_num_states = [1, 2, 3]
    num_to_evaluate = 10

    list_of_populations = []
    for idx in range(num_to_evaluate):
        # Creating an unpruned and pruned lineage
        list_of_populations.append([LineageTree(pi, T, E, (2**8) - 1, 1000000000, prune_condition='fate', prune_boolean=False)])

    AIC_holder = []
    for num_states_to_evaluate in desired_num_states:
        tmp_AIC_holder_by_state = []
        # Analyze the lineages in the list of populations
        output = run_Analyze_over(list_of_populations, num_states_to_evaluate)
        # Collecting the results of analyzing the lineages
        for idx, (tHMMobj, _, LL) in enumerate(output):
            AIC, _ = getAIC(tHMMobj, LL)
            tmp_AIC_holder_by_state.append(AIC)

        AIC_holder.append(tmp_AIC_holder_by_state)

    return desired_num_states, AIC_holder


def AIC_increased_cells2():
    """
    Calculates accuracy and parameter estimation by increasing the number of cells in a lineage for a two-state model.
    """

    # pi: the initial probability vector
    pi = np.ones(2) / 2.0

    # T: transition probability matrix
    T = np.ones((2, 2)) / 2.0

    # bern, gamma_a, gamma_scale
    state_obj0 = StateDistribution(0.99, 20, 5)
    state_obj1 = StateDistribution(0.88, 10, 1)
    E = [state_obj0, state_obj1]

    desired_num_states = [1, 2, 3]
    num_to_evaluate = 10

    list_of_populations = []
    for idx in range(num_to_evaluate):
            # Creating an unpruned and pruned lineage
        list_of_populations.append([LineageTree(pi, T, E, (2**8) - 1, 1000000000, prune_condition='fate', prune_boolean=False)])

    AIC_holder = []
    for num_states_to_evaluate in desired_num_states:
        tmp_AIC_holder_by_state = []
        # Analyze the lineages in the list of populations
        output = run_Analyze_over(list_of_populations, num_states_to_evaluate)
        # Collecting the results of analyzing the lineages
        for idx, (tHMMobj, _, LL) in enumerate(output):
            AIC, _ = getAIC(tHMMobj, LL)
            tmp_AIC_holder_by_state.append(AIC)

        AIC_holder.append(tmp_AIC_holder_by_state)

    return desired_num_states, AIC_holder


def AIC_increased_cells3():
    """
    Calculates accuracy and parameter estimation by increasing the number of cells in a lineage for a three-state model.
    """

    # pi: the initial probability vector
    pi = np.ones(3) / 3.0

    # T: transition probability matrix
    T = np.ones((3, 3)) / 3.0

    # E: states are defined as StateDistribution objects

    # bern, gamma_a, gamma_scale
    state_obj0 = StateDistribution(0.7, 5.0, 1.0)
    state_obj1 = StateDistribution(0.85, 10.0, 2.0)
    state_obj2 = StateDistribution(0.99, 15.0, 3.0)

    E = [state_obj0, state_obj1, state_obj2]

    desired_num_states = [1, 2, 3]
    num_to_evaluate = 10

    list_of_populations = []
    for idx in range(num_to_evaluate):
            # Creating an unpruned and pruned lineage
        list_of_populations.append([LineageTree(pi, T, E, (2**8) - 1, 1000000000, prune_condition='fate', prune_boolean=False)])

    AIC_holder = []
    for num_states_to_evaluate in desired_num_states:
        tmp_AIC_holder_by_state = []
        # Analyze the lineages in the list of populations
        output = run_Analyze_over(list_of_populations, num_states_to_evaluate)
        # Collecting the results of analyzing the lineages
        for idx, (tHMMobj, _, LL) in enumerate(output):
            AIC, _ = getAIC(tHMMobj, LL)
            tmp_AIC_holder_by_state.append(AIC)

        AIC_holder.append(tmp_AIC_holder_by_state)

    return desired_num_states, AIC_holder
>>>>>>> master

    subplotLabel(ax[0], 'A')

    return f