



class Grid(object):
    """Creates GridWorld

        Grid is referenced by single index, starting from upper left corner.

        Example grid indices:

        -------------------------
        |  0  |  1  |  2  |  3  |
        |     |     |     |     |
        -------------------------
        |  4  |  5  |  6  |  7  |
        |     |     |     |     |
        -------------------------

    """

    def __init__(self):

        # Action direction constants
        self.UP = 0
        self.RIGHT = 1
        self.DOWN = 2
        self.LEFT = 3

        # Dimensions
        self.rows = 4
        self.cols = 12

        self.rows = 2
        self.cols = 3

        self.livingCost = -1

        # Non-determinism - this value is split between sideways moves
        self.noise = 0.2

        # self.grid = [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
        #              ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
        #              ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
        #              ' ', -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 126]
        #
        # self.rewards = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        #            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        #            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        #            -1, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 126]

        self.grid = [' ', ' ', ' ',
                     ' ', ' ', 10]

        self.livingReward = -1.0

    def getReward(self, s):
        """ Return reward from leaving state s.

            Not dependent on action or s'
        """
        cell = self.grid[s]
        if type(cell) == int or type(cell) == float:
            return float(cell)
        return self.livingReward


    # def getMove(self, s, action):
    #     """ Sanity check for making a move.
    #
    #         If moving into a wall, returns original position.
    #         Else, returns the move.
    #     """
    #     appActX, appActY = self.applyAction(s, action)
    #     validMove = self._isValidMove(appActX, appActY)
    #     if validMove:
    #         return self.XYToState(appActX, appActY)
    #     else:
    #         return s

    def applyAction(self, s, action):
        x = s // self.cols
        y = s % self.cols

        if action == self.UP:
            return (x - 1, y)
        elif action == self.RIGHT:
            return (x, y + 1)
        elif action == self.DOWN:
            return (x + 1, y)
        elif action == self.LEFT:
            return (x, y - 1)

    def _isValidMove(self, x, y):
        """ Check is x,y is valid location.

        """
        if x < 0 or x >= self.rows: return False
        if y < 0 or y >= self.cols: return False
        return True

    def XYToState(self, x, y):
        return x * self.cols + y


    def getNextStatesAndProbs(self, state, action):
        """

        :param state: current state
        :param action: action taken
        :return: List of successor states and transition probabilities
        """

        #convert to coordinates
        x = state // self.cols
        y = state % self.cols

        self.noise = 0.2

        successors = []

        # UP
        if self._isValidMove(x-1, y):
            upState = self.XYToState(x-1, y)
        else:
            upState = state

        # LEFT
        if self._isValidMove(x, y-1):
            leftState = self.XYToState(x, y-1)
        else:
            leftState = state

        # DOWN
        if self._isValidMove(x+1, y):
            downState = self.XYToState(x+1, y)
        else:
            downState = state

        # RIGHT
        if self._isValidMove(x, y+1):
            rightState = self.XYToState(x, y+1)
        else:
            rightState = state

        # Return the correct one
        if action == self.UP or action == self.DOWN:
            if action == self.UP:
                successors.append([upState,1.0-self.noise])
            else:
                successors.append([downState,1.0-self.noise])

            massLeft = self.noise
            if massLeft > 0.0:
                successors.append([leftState,massLeft/2.0])
                successors.append([rightState,massLeft/2.0])

        if action == self.LEFT or action == self.RIGHT:
            if action == self.LEFT:
                successors.append([leftState,1.0-self.noise])
            else:
                successors.append([rightState,1.0-self.noise])

            massLeft = self.noise
            if massLeft > 0.0:
                successors.append([ upState,massLeft/2.0])
                successors.append([downState,massLeft/2.0])

        return successors