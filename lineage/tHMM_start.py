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
        self.leaves = []
        self.get_leaves()
        
    def get_leaves(self):
        '''Gets the leaves of the tree'''
        for cell in X:
            if cell.left is None and cell.right is None:
                self.leaves.append(cell)
    #make utils.py that stores all our helper functions (not related to the tree)