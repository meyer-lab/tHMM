""" This is a file full of helper functions used in tHMM.py """

def remove_NaNs(X):
    '''Removes unfinished cells in a population'''
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

def max_gen(lineage):
    '''finds the max generation in a lineage'''
    gen_holder = 1
    for cell in lineage:
        if cell.gen > gen_holder:
            gen_holder = cell.gen
    return gen_holder

def get_gen(gen, lineage):
    '''creates a list with all cells in the given generation'''
    first_set = []
    for cell in lineage:
        if cell.gen == gen:
            first_set.append(cell)
    return first_set

def get_numLineages(X):
    ''' Outputs total number of cell lineages in given Population. '''
    linID_holder = [] # temporary list to hold all the linIDs of the cells in the population
    for cell in X: # for each cell in the population
        linID_holder.append(cell.linID) # append the linID of each cell
    numLineages = max(linID_holder)+1 # the number of lineages is the maximum linID+1
    return numLineages

def init_Population(X, numLineages):
    ''' Creates a full population list of lists which contain each lineage in the population. '''
    population = []
    for lineage_num in range(numLineages): # iterate over the number of lineages in the population
        temp_lineage = [] # temporary list to hold the cells of a certain lineage with a particular linID
        for cell in X: # for each cell in the population
            if cell.linID == lineage_num: # if the cell's linID is the lineage num
                temp_lineage.append(cell) # append the cell to that certain lineage
        population.append(temp_lineage) # append the lineage to the Population holder
    return population

def get_parents_for_level(level, lineage):
    """ Returns a set of all the parents of all the cells in a given level/generation. For example this would give you all the non-leaf cells in the generation above the one given. """
    parent_holder = set() #set makes sure only one index is put in and no overlap
    for cell in level:
        parent_cell = cell.parent
        parent_holder.add(lineage.index(parent_cell))
    return parent_holder

def get_daughters(cell):
    """ Returns a list of the daughters of a given cell. """
    temp = []
    if cell.left:
        temp.append(cell.left)
    if cell.right:
        temp.append(cell.right)
    return temp
