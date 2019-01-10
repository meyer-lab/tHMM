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
            
    
    def get_leaves(lineage):
        '''Gives a list of leaves in a lineage'''
        temp_leaves = []
        for cell in lineage:
            if cell.left is None and cell.right is None:
                temp_leaves.append(cell)
        return(temp_leaves)
                    
    #make utils.py that stores all our helper functions (not related to the tree)
    
    def tree_recursion(cell,subtree):
        if cell.isLeaf():
            return
        subtree.append(cell.left)
        subtree.append(cell.right)
        tree_recursion(cell.left, subtree)
        tree_recursion(cell.right, subtree)
        return
    
    def get_subtrees(node,lineage):
        '''Get subtrees for one lineage'''
        subtree_list = [node] 
        tree_recursion(node,subtree)
        not_subtree = []
        for cell in lineage:
            if cell not in subtree:
                not_subtree.append(cell)
        return subtree, not_subtree
    
    def find_two_subtrees(node,lineage):
        '''Gets the left and right subtrees from a cell'''
        left_sub,_ = get_subtrees(cell.left,lineage)
        right_sub,_ = get_subtrees(cell.right,lineage)
        neither_subtree=[]
        for cell in lineage:
            if cell not in left_sub and cell not in right_sub:
                neither_subtree.append(cell)
        return left_sub, right_sub, neither_subtree
    
    def get_mixed_subtrees(node_m,node_n,lineage):
        m_sub,_ = get_subtrees(node_m,lineage)
        n_sub,_ = get_subtrees(node_n,lineage)
        mixed_sub = []
        for cell in m_sub:
            if cell not in n_sub:
                mixed_sub.append(cell)
        not_mixed = []
        for cell in lineage:
            if cell not in mixed_sub:
                not_mixed.append(cell)
        return mixed_sub, not_mixed
            
        
    
    
    