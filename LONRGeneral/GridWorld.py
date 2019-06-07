
import numpy as np

from LONRGeneral.LONR import *

class Grid(MDP):
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

    def __init__(self, noise=0.0, startState=None):
        super().__init__(startState=startState)

        # One player MDP
        self.N = 1

        # Action direction constants
        self.UP = 0
        self.RIGHT = 1
        self.DOWN = 2
        self.LEFT = 3

        # Dimensions
        self.rows = 4
        self.cols = 5

        # self.rows = 3  #THIS IS EXP3
        # self.cols = 4

        self.numberOfStates = self.rows * self.cols
        self.numberOfActions = 4

        # Non-determinism - this value is split between sideways moves
        self.noise = noise

        self.grid = [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
                     ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
                     ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
                     ' ', -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 200]

        # self.grid = [' ', ' ', ' ', ' ',
        #              ' ', ' ', ' ', ' ',
        #              ' ', -1, -1,  10]

        # self.grid = [' ', ' ', ' ', ' ', ' ',
        #              ' ', ' ', ' ', ' ', ' ',
        #              ' ', -1, -1, -1, 10]
        #
        self.grid = [' ', ' ', ' ', ' ', ' ',
                     ' ', ' ', ' ', ' ', ' ',
                     ' ', ' ', ' ', ' ', ' ',
                     ' ', -1, -1, -1, 100]
        #
        # self.grid = [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
        #              ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
        #              ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
        #              ' ', -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 10]

        # self.grid = [' ', ' ', ' ', ' ',
        #              ' ', -100, -100, 2000]

        # self.grid = [' ', 1]

        # Q
        self.Q = {}
        self.Q_bu = {}
        self.QSums = {}
        self.QSum = []
        self.QTouched = {}

        # Policy
        self.pi = {}
        self.pi_sums = {}
        self.regret_sums = {}

        # Misc
        self.iterations = []
        self.weights = {}
        self.runningRewards = {}

        # Initialize everything
        # Ex: Q[player][state][action] -> Q[n][s][a]

        for n in range(self.N):
            self.Q[n] = {}
            self.Q_bu[n] = {}
            self.QSums[n] = {}
            self.QTouched[n] = {}

            self.pi[n] = {}
            self.pi_sums[n] = {}
            self.regret_sums[n] = {}
            self.weights[n] = {}
            self.runningRewards[n] = {}
            for s in self.getStates():
                self.Q[n][s] = {}
                self.Q_bu[n][s] = {}
                self.QSums[n][s] = {}
                self.QTouched[n][s] = {}

                self.pi[n][s] = {}
                self.regret_sums[n][s] = {}
                self.pi_sums[n][s] = {}
                self.weights[n][s] = {}
                self.runningRewards[n][s] = {}
                for a in self.getActions(s,0):
                    self.Q[n][s][a] = 0.0
                    self.Q_bu[n][s][a] = 0.0
                    self.QSums[n][s][a] = 0.0
                    self.QTouched[n][s][a] = 0.0

                    self.pi[n][s][a] = 1.0 / len(self.getActions(s, 0)) # uniform init
                    self.regret_sums[n][s][a] = 0.0
                    self.pi_sums[n][s][a] = 0.0
                    self.weights[n][s][a] = 1.0
                    self.runningRewards[n][s][a] = 0.0




        self.livingReward = -1.0#0.0 #-1.0 #0.0

        logSettings = True
        if logSettings:
            print("Grid Settings: ")
            print("     Grid: ")
            gcell = 0
            for x in range(self.rows):
                for y in range(self.cols):
                    g = self.grid[x * self.cols + y]
                    g = " ***** " if g == ' ' else g
                    if type(g) == int:
                        g = " " + str(g) + " "
                    print(g, end='')
                print("")
            # print("           ", self.grid)
            print("     Noise: ", self.noise)
            print("     Living reward: ", self.livingReward)

    def getActions(self, s, n):
        if self.isTerminal(s):
            return ["exit"]
        else:
            return [self.UP, self.RIGHT, self.DOWN, self.LEFT]

    def getStates(self):
        return list(range(self.rows * self.cols))

    def getStateRep(self, s):
        return s

    def getReward(self, s, a_current, n, a_notN):
        """ Return reward from leaving state s.

            Not dependent on action or s'
        """
        cell = self.grid[s]
        if type(cell) == int or type(cell) == float:
            return float(cell)
        return self.livingReward


    # Only needed for O-LONR
    def getMove(self, s, action):
        """  Get s_prime given s, action
        """

        succs = self.getNextStatesAndProbs(s, action, 0)

        # Get a move based on Transition probs
        actionChoices = []
        actionProbs = []
        for state, prob, reward in succs:
            actionChoices.append(state)
            actionProbs.append(prob)

        #print("S: ", s, " action: ", action, "ActionChoices: ", actionChoices)
        next_state = np.random.choice(actionChoices, p=actionProbs)
        return next_state

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


    def getNextStatesAndProbs(self, state, action, n_current):
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
                successors.append([upState,1.0-self.noise, self.getReward(state,action,0,0)])
            else:
                successors.append([downState,1.0-self.noise, self.getReward(state,action,0,0)])

            massLeft = self.noise
            if massLeft > 0.0:
                successors.append([leftState,massLeft/2.0, self.getReward(state,self.LEFT,0,0)])
                successors.append([rightState,massLeft/2.0, self.getReward(state,self.RIGHT,0,0)])

        if action == self.LEFT or action == self.RIGHT:
            if action == self.LEFT:
                successors.append([leftState,1.0-self.noise, self.getReward(state,action,0,0)])
            else:
                successors.append([rightState,1.0-self.noise, self.getReward(state,action,0,0)])

            massLeft = self.noise
            if massLeft > 0.0:
                successors.append([upState,massLeft/2.0, self.getReward(state,self.UP,0,0)])
                successors.append([downState,massLeft/2.0, self.getReward(state,self.DOWN,0,0)])
        # if np.random.randint(0, 1000) < 5: #reminder
        #     print("NEXT STATES CHANGED")
        return successors


    # def getNextStatesAndProbsGrid(self, state, action, n_current):
    #     """ Returns a list of [Action, resulting state, transition probability]
    #     """
    #
    #     #convert to coordinates
    #     x = state // self.cols
    #     y = state % self.cols
    #
    #
    #     successors = []
    #
    #     # UP
    #     if self._isValidMove(x-1, y):
    #         upState = self.XYToState(x-1, y)
    #     else:
    #         upState = state
    #
    #     # LEFT
    #     if self._isValidMove(x, y-1):
    #         leftState = self.XYToState(x, y-1)
    #     else:
    #         leftState = state
    #
    #     # DOWN
    #     if self._isValidMove(x+1, y):
    #         downState = self.XYToState(x+1, y)
    #     else:
    #         downState = state
    #
    #     # RIGHT
    #     if self._isValidMove(x, y+1):
    #         rightState = self.XYToState(x, y+1)
    #     else:
    #         rightState = state
    #
    #     # Return the correct one
    #     if action == self.UP or action == self.DOWN:
    #         if action == self.UP:
    #             successors.append([upState,1.0-self.noise, self.getReward(state,action,0,0), self.UP])
    #         else:
    #             successors.append([downState,1.0-self.noise, self.getReward(state,action,0,0), self.DOWN])
    #
    #         massLeft = self.noise
    #         if massLeft > 0.0:
    #             successors.append([leftState,massLeft/2.0, self.getReward(state,self.LEFT,0,0), self.LEFT])
    #             successors.append([rightState,massLeft/2.0, self.getReward(state,self.RIGHT,0,0), self.RIGHT])
    #
    #     if action == self.LEFT or action == self.RIGHT:
    #         if action == self.LEFT:
    #             successors.append([leftState,1.0-self.noise, self.getReward(state,action,0,0), self.LEFT])
    #         else:
    #             successors.append([rightState,1.0-self.noise, self.getReward(state,action,0,0), self.RIGHT])
    #
    #         massLeft = self.noise
    #         if massLeft > 0.0:
    #             successors.append([upState,massLeft/2.0, self.getReward(state,self.UP,0,0), self.UP])
    #             successors.append([downState,massLeft/2.0, self.getReward(state,self.DOWN,0,0), self.DOWN])
    #
    #     return successors


    def printGrid(self):
        gcell = 0
        for x in range(self.rows):
            for y in range(self.cols):
                g = self.grid[x * self.cols + y]
                g = " ***** " if g == ' ' else g
                if type(g) == int:
                    g = " " + str(g) + " "
                print(g, end='')
            print("")