''' This file is used to prune the tree, i.e., removing those cells that have been assigned to die and their descendants'''

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