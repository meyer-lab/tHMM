""" Common utilities used between states regardless of distribution. """

class Time:
    """
    Class that stores all the time related observations in a neater format.
    This will assist in pruning based on experimental time as well as
    obtaining attributes of the lineage as a whole, such as the
    average growth rate.
    """
    def __init__(self, startT, endT):
        self.startT = startT
        self.endT = endT


def assign_times(lineageObj):
    """
    Assigns the start and end time for each cell in the lineage.
    The time observation will be stored in the cell's observation parameter list
    in the second position (index 1). See the other time functions to understand.
    """
    # traversing the cells by generation
    for gen_minus_1, level in enumerate(lineageObj.full_list_of_gens[1:]):
        true_gen = gen_minus_1 + 1  # generations are 1-indexed
        if true_gen == 1:
            for cell in level:
                assert cell.isRootParent()
                cell.time = Time(0, cell.obs[1])
        else:
            for cell in level:
                cell.time = Time(cell.parent.time.endT, cell.parent.time.endT + cell.obs[1])


def get_experiment_time(lineageObj):
    """
    This function returns the longest experiment time
    experienced by cells in the lineage.
    We can simply find the leaf cell with the
    longest end time. This is effectively
    the same as the experiment time for synthetic lineages.
    """
    longest = 0.0
    for cell in lineageObj.output_leaves:
        if cell.time.endT > longest:
            longest = cell.time.endT
    return longest


def fate_censor_rule(cell):
    """
    User-defined function that checks whether a cell's subtree should be removed.
    Our example is based on the standard requirement that the first observation
    (index 0) is a measure of the cell's fate (1 being alive, 0 being dead).
    Clearly if a cell has died, its subtree must be removed.
    """
    if cell.obs[0] == 0:
        if not cell.isLeafBecauseTerminal():
            cell.left.censored = True
            cell.right.censored = True
        assert cell.isLeaf()
    if not cell.isRootParent():
        if cell.parent.censored:
            cell.censored = True


def time_censor_rule(cell, desired_experiment_time):
    """
    User-defined function that checks whether a cell's subtree should be removed.
    Our example is based on the standard requirement that the second observation
    (index 1) is a measure of the cell's lifetime.
    If a cell has lived beyond a certain experiment time, then its subtree
    must be removed.
    """
    if cell.time.endT > desired_experiment_time:
        cell.time.endT = desired_experiment_time
        cell.obs[1] = cell.time.endT - cell.time.startT
        if not cell.isLeafBecauseTerminal():
            cell.left.censored = True
            cell.right.censored = True
        assert cell.isLeaf()
    if not cell.isRootParent():
        if cell.parent.censored:
            cell.censored = True

            
def skew(data):
    """
    skew is third central moment / variance**(1.5)
    """
    data = np.ravel(data)
    mu = data.mean()
    m2 = ((data - mu)**2).mean()
    m3 = ((data - mu)**3).mean()
    return m3 / np.power(m2, 1.5)