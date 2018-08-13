


class cell:
    ''' This class defines what a cell is. All the variables are kept private to make sure the internal state
    is consistent. '''
    def __init__(self, tstart, parent):
        assert isinstance(parent, int) or parent is None

        self.__tstart = tstart
        self.__parent = parent
        self.__children = None
        self.__tstop = None

    def setDivide(self, tstop, children):
        assert self.__tstop is None
        # Check that children is setup correctly.

        self.__children = children
        self.__tstop = tstop

    def setDead(self, tstop):
        assert self.__tstop is None

        self.__tstop = tstop

    @property
    def tstart(self):
        return self.__tstart

    @property
    def parent(self):
        return self.__parent

    @property
    def children(self):
        return self.__children

    @property
    def tstop(self):
        return self.__tstop

class tree:
    def __init__(self):
        self.tree = list()

    def idxDivide(self, idx):
        assert isinstance(idx, int)
        assert idx < len(self.tree)

        # Add steps to divide cell here.
