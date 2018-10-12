# author : shakthi visagan (shak360), adam weiner (adamcweiner)
# description: a file to hold the cell class

class CellNode:
    def __init__(self, key, startT, endT, fate=True, left=None, right=None, parent=None):
        self.key = key
        self.startT = startT
        self.endT = endT
        self.tau = self.endT - self.startT # avoiding self.t, since that is a common function (i.e. transposing matrices)
        self.fate = fate
        self.left = left
        self.right = right
        self.parent = parent
    
    def hasLeft(self):
        return self.left

    def hasRight(self):
        return self.right

    def isLeft(self):
        return self.parent and self.parent.left == self

    def isRight(self):
        return self.parent and self.parent.right == self

    def isParent(self):
        return not self.parent

    def isChild(self):
        return not (self.left or self.right)

    def hasAnyChildren(self):
        return self.right or self.left

    def hasBothChildren(self):
        return self.right and self.left

    def replaceCellNodeData(self, key, startT, endT, fate, left, right):
        self.key = key
        self.startT = startT
        self.endT = endT
        self.tau = self.endT - self.startT
        self.fate = fate
        self.left = leftChild
        self.right = rightChild
        if self.hasLeftChild():
            self.left.parent = self
        if self.hasRightChild():
            self.right.parent = self










