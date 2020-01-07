""" Unit test file. """
import unittest
import numpy as np
from ..CellVar import CellVar as c, _double


# pylint: disable=protected-access

class TestModel(unittest.TestCase):
    """
    Unit test class for the cell class.
    """

    def test_cellVar(self):
        """ Make sure cell state assignment is correct. """
        left_state = 0
        right_state = 1

        cell_left = c( 
            state=left_state,
            left=None,
            right=None,
            parent=None,
            gen=1)
        cell_right = c( 
            state=right_state,
            left=None,
            right=None,
            parent=None,
            gen=1)

        self.assertTrue(cell_left.state == 0)
        self.assertTrue(cell_right.state == 1)

    def test_cell_divide(self):
        """ Tests the division of the cells. """
        #cell will remain in its state
        T = np.array([[1.0, 0.0], 
                      [0.0, 1.0]]) 
        
        #this is the parent cell
        parent_state = 1
        cell = c(
            state=parent_state,
            left=None,
            right=None,
            parent=None,
            gen=1) 
        left_cell, right_cell = cell._divide(T)
       
        #left cell == 0, should return false
        self.assertTrue(left_cell.state == 1)
        
        #right cell == 1, should return true
        self.assertTrue(right_cell.state == 1) 
        self.assertTrue(
            right_cell.parent is cell and left_cell.parent is cell) 
        self.assertTrue(
            cell.left is left_cell and cell.right is right_cell)
        self.assertTrue(not cell.parent)
        self.assertTrue(cell.gen == 1)
        self.assertTrue(left_cell.gen == 2 and right_cell.gen == 2)
        
        #new assert
        self.assertFalse(left_cell.gen == 1 and right_cell.gen == 2) 

        parent_state = 0
        cell = c(
            state=parent_state,
            left=None,
            right=None,
            parent=None,
            gen=1)
        left_cell, right_cell = cell._divide(T)
        
        # the probability of switching states is 0
        self.assertTrue(left_cell.state == 0)
        
        #cells didn't divide
        self.assertTrue(right_cell.state == 0) 
        
        #cell is still the parent
        self.assertTrue(
            right_cell.parent is cell and left_cell.parent is cell) 
        self.assertTrue(
            cell.left is left_cell and cell.right is right_cell)
        
        #returns true because there is a left and right cell now
        self.assertTrue(not cell.parent) 
        self.assertTrue(cell.gen == 1)
        
        #cells divided
        self.assertTrue(left_cell.gen == 2 and right_cell.gen == 2) 

    def test_isParent(self):
        """ Tests the parent relationships of cells. """
        #cells will not change states
        T = np.array([[1.0, 0.0],
                      [0.0, 1.0]]) 

        #cell has no children right now
        parent_state = 1
        cell = c(
            state=parent_state, 
            left=None, 
            right=None,
            parent=None, 
            gen=1)
        #returns false because no left/right cells
        self.assertFalse(cell._isParent()) 
        #cell divides
        left_cell, right_cell = cell._divide(T) 
        #returns true because cell divided
        self.assertTrue(cell._isParent()) 
        #tests if divided cell is right
        self.assertTrue(left_cell._get_sister() is right_cell) 
        #tests if divided cell is left
        self.assertTrue(right_cell._get_sister() is left_cell) 

    def test_isChild(self):
        """ Tests the daughter relationships of cells. """
        #cell will not change states
        T = np.array([[1.0, 0.0],
                      [0.0, 1.0]]) 

        parent_state = 1
        cell = c(
            state=parent_state,
            left=None,
            right=None,
            parent=None,
            gen=1)
        #returns false because parent cell has no children
        self.assertFalse(cell._isChild()) 
        #cell divides
        left_cell, right_cell = cell._divide(T) 
        #returns true because parent cell divided
        self.assertTrue(left_cell._isChild() and right_cell._isChild()) 

    def test_isRootParent(self):
        """ Tests whether the correct root parent asserts work. """
        #cell will not change states
        T = np.array([[1.0, 0.0],
                      [0.0, 1.0]]) 

        parent_state = 1
        cell = c(
            state=parent_state,
            left=None,
            right=None,
            parent=None,
            gen=1)
        self.assertTrue(cell._isRootParent())
        self.assertTrue(cell._isLeaf()) 
        #cell divides
        left_cell, right_cell = cell._divide(T)
        self.assertFalse(left_cell._isRootParent() and right_cell._isRootParent())
        #tests if cell is root parent
        self.assertTrue(cell._isRootParent()) 
        #returns false because divided cells aren't root parent
        self.assertFalse(left_cell._isRootParent() 
                         and right_cell._isRootParent())
        self.assertTrue(left_cell._isLeaf() and right_cell._isLeaf())
        self.assertFalse(cell._isLeaf()) 
        
    

    def test_isLeaf(self):
        """ Tests whether the leaf cells are correctly checked. """
        T = np.array([[1.0, 0.0],
                      [0.0, 1.0]]) #cell will not change states

        parent_state = 1
        #parent cell has not divided
        cell = c(
            state=parent_state, 
            left=None,
            right=None,
            parent=None,
            gen=1)
        #parent cell is currently the last cell
        self.assertTrue(cell._isLeaf()) 
        #cell divides
        left_cell, right_cell = cell._divide(T) 
        #parent cell is no longer last cell
        self.assertFalse(cell._isLeaf()) 
        #daughter cells are the last cells
        self.assertTrue(left_cell._isLeaf() and right_cell._isLeaf()) 

    def test_get_sister(self):
        """ Tests the relationships between related cells. """
        #cell will not change states
        T = np.array([[1.0, 0.0],
                      [0.0, 1.0]]) 

        parent_state = 1
        #cell has not divided
        cell = c(
            state=parent_state, 
            left=None,
            right=None,
            parent=None,
            gen=1)
        #cell divides
        left_cell, right_cell = cell._divide(T) 
        #tests if sister cells are right and left
        self.assertTrue(left_cell._get_sister(
        ) is right_cell and right_cell._get_sister() is left_cell) 

    def test_get_root_cell(self):
        """ Tests the function that returns the root cell. """
        T = np.array([[1.0, 0.0],
                      [0.0, 1.0]])

        parent_state = 1
        cell = c(
            state=parent_state,
            left=None,
            right=None,
            parent=None,
            gen=1)
        #cell hasn't divided, is root cell
        self.assertTrue(cell._get_root_cell() is cell) 
        left_cell, right_cell = cell._divide(T)
        #given the left/right cell, gets the root cell
        self.assertTrue(left_cell._get_root_cell()
                        is cell and right_cell._get_root_cell() is cell) 

    def test_cell_double(self):
        """ Make sure double function creates the right and left states properly. """
        # transition matrix
        T = np.array([[1.0, 0.0],
                      [0.0, 1.0]])

        parent_state = 0
        left_state, right_state = _double(parent_state, T)
        self.assertTrue(left_state == 0) 
        self.assertTrue(right_state == 0)

        parent_state = 1
        left_state, right_state = _double(parent_state, T)
        self.assertTrue(left_state == 1) 
        self.assertTrue(right_state == 1)
        
    
    def test_cell_state_change1 (self):
        """
        Tests what will happen when probability of switching states is 100%
        """
        #transition matrix, valid matrix
        T = np.array([[0.0, 1.0],
                      [1.0, 0.0]])
        #cells should transition states from parent to daughter
        parent_state = 0
        left_state, right_state = _double (parent_state, T)
        self.assertTrue(left_state == 1)
        self.assertTrue(right_state == 1)
        
        parent_state = 1
        left_state, right_state = _double(parent_state, T)
        self.assertTrue(left_state == 0)
        self.assertTrue(right_state == 0)
        

    def test_cell_state_change2 (self):
        """
        Tests what will happen when probability from switching from state 0 to 1 is 100% 
        and probability from switching from state 1 to 0 is 0%
        """
        #transition matrix
        T = np.array([[0.0, 1.0],
                      [0.0, 1.0]])
        
        parent_state = 0
        left_state, right_state = _double (parent_state, T)
        self.assertTrue (left_state == 1)
        self.assertTrue (right_state == 1)
        
        parent_state = 1
        left_state, right_state = _double(parent_state, T)
        self.assertFalse(left_state == 0)
        self.assertFalse(right_state == 0)

        
        
#     #new function
#     def test_cell_state_change3 (self):
#         "Tests the transition states for SUM159"
#         #transition matrix
#         T = np.array([[0.58, 0.07, 0.35],
#                       [0.04, 0.47, 0.49],
#                       [0.01, 0.0, 0.99]])

#         parent_state = 0
#         left_state, right_state = _double (parent_state, T)
#         #generates random number that represents probability of switching states
#         randomNum = random.random() 
#         #if probability is less than or equal to 0.58, cell will stay in state 0
#         if (randomNum <= T[0][0]): 
#             self.assertTrue(left_state == 0 and right_state == 0)
#         #probability cell will switch from state 0 to 1 
#         if (randomNum > T[0][0] and randomNum <= T[0][0]+T[0][1]):
#             self.assertTrue (left_state == 1 and right_state == 1)
#         #probability cell will switch from state 0 to 2
#         if (randomNum > T[0][0] + T[0][1] and randomNum < 1):
#             self.assertTrue (left_state == 2 and right_state == 2)

#         parent_state = 1
#         left_state, right_state = _double (parent_state, T)
#         randomNum = random.random()
#         #probability cell will transition from state 1 to 0
#         if(randomNum <= T[1][0]):
#             self.assertTrue (left_state == 0 and right_state == 0)
#         #probability cell will remain in state 1
#         if (randomNum > T[1][0] and randomNum =< T[1][0]+T[1][1]):
#             self.assertTrue (left_state == 1 and right_state == 1)
#         #probability cell will transition from state 1 to 2
#         if(randomNum > T[1][0]+T[1][1] and randomNum < 1):
#             self.assertTrue (left_state == 2 and right_state == 2)


#         parent_state = 2
#         left_state, right_state = _double (parent_state, T)
#         randomNum = random.random()
#         #probability cell will transition from 2 to 0
#         if (randomNum <= T[2][0]):
#             self.assertTrue (left_state == 0 and right_state == 0)
#         #0% probability of transitioning from state 2 to 1
#         self.assertFalse (left_state == 1) 
#         self.assertFalse (right_state == 1)
#         #probability cell will remain in state 2
#         if (randomNum > T[2][0]+T[2][1] and RandomNum < 1):
#             self.assertTrue (left_state == 2 and right_state == 2)

