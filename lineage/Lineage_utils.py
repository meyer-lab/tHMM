# utility and helper functions for cleaning up input
# populations and lineages
# and other needs in the tHMM class

def remove_NaNs(X):
    '''Removes unfinished cells in Population'''
    ii = 0 # establish a count outside of the loop
    while ii in range(len(X)): # for each cell in X
        if X[ii].isUnfinished(): # if the cell has NaNs in its times
            if X[ii].parent is None: # do nothing if the parent pointer doesn't point to a cell
                pass
            elif X[ii].parent.left is X[ii]: # if it is the left daughter of the parent cell
                X[ii].parent.left = None # replace the cell with None
            elif X[ii].parent.right is X[ii]: # or if it is the right daughter of the parent cell
                X[ii].parent.right = None # replace the cell with None
            X.pop(ii) # pop the unfinished cell at the current position
        else:
            ii += 1 # only move forward in the list if you don't delete a cell
    return X

def get_numLineages(X):
    ''' Outputs total number of cell lineages in given Population. '''
    root_cell_holder = [] # temp list to hold the root cells in the population
    root_cell_linID_holder = [] # temporary list to hold all the linIDs of the root cells in the population
    for cell in X: # for each cell in the population
        if cell.isRootParent():
            root_cell_holder.append(cell)
            root_cell_linID_holder.append(cell.linID) # append the linID of each cell
    assert(len(root_cell_holder) == len(root_cell_linID_holder))
    numLineages = len(root_cell_holder) # the number of lineages is the number of root cells
    return numLineages

def init_Population(X, numLineages):
    '''Creates a full population list of lists which contain each lineage in the Population.'''
    root_cell_holder = [] # temp list to hold the root cells in the population
    for cell in X: # for each cell in the population
        if cell.isRootParent():
            root_cell_holder.append(cell)
    population = []
    for lineage_num in range(numLineages): # iterate over the number of lineages in the population
        temp_lineage = [] # temporary list to hold the cells of a certain lineage with a particular linID
        for cell in X: # for each cell in the population
            if cell.get_root_cell() is root_cell_holder[lineage_num]: # if the cell's root cell is the root cell we're on
                assert(cell.linID == cell.get_root_cell().linID)
                temp_lineage.append(cell) # append the cell to that certain lineage
        if len(temp_lineage)>0:
            population.append(temp_lineage) # append the lineage to the Population holder
    return population