import scipy.stats as sp
import numpy as np

class cell:
    def __init__(self, state, left, right, parent):

        self.state = state
        self.left = left
        self.right = right
        self.parent = parent
#         self.observation = observation ## TODO
        
    def divide(self, state, T):

        state = self.state
        left_state, right_state = double(state, T)
        self.left = cell(state = left_state, left = None, right = None, parent = self)
        self.right = cell(state = right_state, left = None, right = None, parent = self)
        
        return [self.left, self.right]




# second function
def double(state, T):

    num_states = len(T[0])
#     assertIn(state, range(num_state)), "the state you provided is not in the range"

    state_decider = sp.multinomial.rvs(2, T[state])
    s = state_decider.tolist()

    if 2 in s:
        same_index = s.index(2)
        left_state = same_index
        right_state = same_index
        new_states = [left_state, right_state]

    elif 1 in s:
        indexes = [i for i, x in enumerate(s) if x == 1]
        left_state = indexes[0]
        right_state = indexes[1]
        new_states = [left_state, right_state]

    return new_states


# Main function
def generate(T, Pi, num_cells):
    state_dice = sp.multinomial.rvs(1, Pi)
    s = state_dice.tolist()
    first_cell_state = s.index(1) 
    first_cell = cell(state = first_cell_state, left = None, right = None, parent = None)
    X = [first_cell]
    
    for cel in X:
        if cel.left is None:
            state = cel.state
            cel_l, cel_r = cel.divide(state, T)
            X.append(cel_l)
            X.append(cel_r)

        if len(X) >= num_cells:
            break
    return X
