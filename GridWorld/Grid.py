
import numpy as np


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
        self.rows = 2
        self.cols = 3

        self.rows = 4
        self.cols = 12

        # Non-determinism - this value is split between sideways moves
        self.noise = 0.20

        self.grid = [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
                     ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
                     ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
                     ' ', -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 200]


        # self.grid = [' ', ' ', ' ',
        #              ' ', ' ', 10]

        self.livingReward = -1.0


    def printGrid(self, q=None):
        print("GridWorld - current grid")
        for r in range(self.rows):
            for c in range(self.cols):
                char = self.grid[r * self.cols + c]
                if type(char) == str:
                    if q is None:
                        char = char.ljust(8, '-')
                    else:
                        asd = 2
                        char = round(max(q[r * self.cols + c]), 1)
                        char = str(char)
                        char = char.ljust(8, ' ')
                else:
                    char = ' ' + str(char)
                    char = char.ljust(8, ' ')
                print(char, end='')
            print("")
        print("")

    def getReward(self, s):
        """ Return reward from leaving state s.

            Not dependent on action or s'
        """
        cell = self.grid[s]
        if type(cell) == int or type(cell) == float:
            return float(cell)
        return self.livingReward


    def getMove(self, s, action):
        """ Sanity check for making a move
            As well as returning non-deterministic move.
            If moving into a wall, returns original position.
            Else, returns the move.
        """

        # If non-deterministic
        if self.noise > 0:

            # Get list of successor states and transition probabilities
            succs = self.getNextStatesAndProbs(s, action)

            # Get a move based on Transition probs
            actionChoices = []
            actionProbs = []
            for action, state, prob in succs:
                actionChoices.append(action)
                actionProbs.append(prob)
            action = np.random.choice(actionChoices, p=actionProbs)

        # Apply the action
        appActX, appActY = self._applyAction(s, action)

        # Check if valid move
        validMove = self._isValidMove(appActX, appActY)

        # Return based on valid or not
        if validMove:
            return self.XYToState(appActX, appActY)
        else:
            return s

    def isTerminal(self, s):
        """Returns True if s is a terminal state.
        """
        return type(self.grid[s]) == int or type(self.grid[s]) == float

    def _applyAction(self, s, action):
        """Returns x,y of new state when in state s and taking action a
        """
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
        """ Returns a list of [Action, resulting state, transition probability]
        """

        #convert to coordinates
        x = state // self.cols
        y = state % self.cols


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