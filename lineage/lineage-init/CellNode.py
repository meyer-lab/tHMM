# author : shakthi visagan (shak360), adam weiner (adamcweiner)
# description: a file to hold the cell class

class CellNode:
    def __init__(self, key, startT, parent=None):
        self.key = key
        self.startT = startT
        # self.endT = endT
        # self.tau = self.endT - self.startT # avoiding self.t, since that is a common function (i.e. transposing matrices)
        # self.fate = fate
        # self.left = left
        # self.right = right
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

    def calcTau(self):
        self.tau = self.endT - self.startT   # calculate tau here
    
    def die(self, endT):
        """ Cell dies without dividing. """
        self.fate = False   # no division
        self.endT = endT   # mark endT
    
    def divide(self, endT):
        """ Cell life ends through division. """
        self.fate = True   # division
        self.endT = endT   # mark endT

        # two daughter cells emerge at this time... not sure about 1st and 3rd arguments bc I don't know what "key" is and don't know what python's statement for "this" is.
        self.left = CellNode(self.key, endT, parent=self)
        self.right = CellNode(self.key, endT, parent=self)

    








