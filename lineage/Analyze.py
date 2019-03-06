def Analyze(X, numStates):
    tHMMobj = tHMM(X, numStates=numStates) # build the tHMM class with X
    fit(tHMMobj, max_iter=200, verbose=False)
    deltas, state_ptrs = get_leaf_deltas(tHMMobj) # gets the deltas matrix
    get_nonleaf_deltas(tHMMobj, deltas, state_ptrs)
    all_states = Viterbi(tHMMobj, deltas, state_ptrs) 
    return(deltas, state_ptrs, all_states)