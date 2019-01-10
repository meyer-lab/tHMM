def remove_NaNs(X)
    '''Removes unfinished cells in a population'''
    count = 0
    for cell in X:
        if cell.isUnfinished():
            X.pop(count)
            count-=1
        count+=1
            
class tHMM:
    def __init__(self, numstates, X):
        ''' Instantiates a tHMM.'''
        self.numstates = numstates
        self.X = X
        
    def get_lincount(self):
        '''outputs total number of cell lineages'''
        linID_holder = []
        for cell in self.X:
            linID_holder.append(cell.linID)
        self.numlineages = max(linID.holder)+1
        return(self.numlineages)
    
    def get_treelist(self):
        '''creates a full population list of lists which contain each lineage in the population'''
        numlineages = self.get_lincount()
        self.population = []
        for lineage in range(numlineages):
            temp_lineage = []
            for cell in self.X:
                if cell.linID = lineage:
                    temp_lineage.append(cell)
            self.population.append(temp_lineage)
        return(self.population)
            
    
    def get_leaves(self):
        '''Gets the leaves of the tree'''
        self.full_leaves = []
        for lineage in self.population:
            temp_leaves = []
            for cell in lineage:
                if cell.left is None and cell.right is None:
                    temp_leaves.append(cell)
            self.full_leaves.append(temp_leaves)
        return(self.full_leaves)
                    
    #make utils.py that stores all our helper functions (not related to the tree)
    
    
    