



class Grid(object):
    """Creates GridWorld


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
        self.cols = 2

        self.livingCost = -1

        # self.grid = [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
        #              ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
        #              ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
        #              ' ', -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 126]
        #
        # self.rewards = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        #            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        #            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        #            -1, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 126]

        self.grid = [
                     ' ', ' ',
                     ' ', 10]

        self.rewards = [-1, -1,
                        -1, 10]

    def getMove(self, s, action):
        """

        :param action:
        :return: Next State
        """
        appActX, appActY = self.applyAction(s, action)
        validMove = self._isValidMove(appActX, appActY)
        if validMove:
            return self.XYToState(appActX, appActY)
        else:
            return s

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
        if x < 0 or x >= self.rows: return False
        if y < 0 or y >= self.cols: return False
        return True

    def XYToState(self, x, y):
        return x * self.cols + y


    def getNextStatesAndProbs(self, state, action):


        # If terminal, no actions left
        # if type(self.grid[state]) == int:
        #     return []

        #convert to coordinates
        x = state // self.cols
        y = state % self.cols

        self.noise = 0.0

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
                successors.append([self.UP, upState,1.0-self.noise])
            else:
                successors.append([self.DOWN, downState,1.0-self.noise])

            massLeft = self.noise
            if massLeft > 0.0:
                successors.append([self.LEFT, leftState,massLeft/2.0])
                successors.append([self.RIGHT, rightState,massLeft/2.0])

        if action == self.LEFT or action == self.RIGHT:
            if action == self.LEFT:
                successors.append([self.LEFT, leftState,1.0-self.noise])
            else:
                successors.append([self.RIGHT, rightState,1.0-self.noise])

            massLeft = self.noise
            if massLeft > 0.0:
                successors.append([self.UP, upState,massLeft/2.0])
                successors.append([self.DOWN, downState,massLeft/2.0])

        return successors