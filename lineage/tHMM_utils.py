'''utility and helper functions for recursions and other needs in the tHMM class'''

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

def get_parents_for_level(level, lineage):
    """
        Returns a set of all the parents of all the cells in a
        given level/generation. For example this would give you
        all the non-leaf cells in the generation above the one given.
    """
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

def print_Assessment(tHMMobj):
    for num in range(tHMMobj.numLineages):
        print("\n")
        print("Initial Proabablities: ")
        print(tHMMobj.paramlist[num]["pi"])
        print("Transition State Matrix: ")
        print(tHMMobj.paramlist[num]["T"])
        print("Emission Parameters: ")
        print(tHMMobj.paramlist[num]["E"])
        print("Expected Emissions Parameters: ")
        print(expected_lineage_parameters[num])