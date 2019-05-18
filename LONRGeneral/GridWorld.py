
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
        self.rows = 2
        self.cols = 3

        self.rows = 4
        self.cols = 12

        self.numberOfStates = self.rows * self.cols
        self.numberOfActions = 4
        self.N = 1

        # Non-determinism - this value is split between sideways moves
        self.noise = noise

        self.grid = [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
                     ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
                     ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
                     ' ', -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 200]

        # self.Q = np.zeros((self.N, self.numberOfStates, self.numberOfActions))
        # self.Q_bu = np.zeros((self.N, self.numberOfStates, self.numberOfActions))
        # self.QSums = np.zeros((self.N, self.numberOfStates, self.numberOfActions))
        #
        # # Policies
        # self.pi = np.zeros((self.N, self.numberOfStates, self.numberOfActions))  # policy (if needed)
        # self.pi_sums = np.zeros((self.N, self.numberOfStates, self.numberOfActions))
        #
        # # Regret Sums
        # self.regret_sums = np.zeros((self.N, self.numberOfStates, self.numberOfActions))  # regret_sum (if needed)
        #
        # for n in range(self.N):
        #     for s in range(self.numberOfStates):
        #         for a in range(self.numberOfActions):
        #             self.pi[n][s][a] = 1.0 / self.numberOfActions

        self.Q = {}
        self.Q_bu = {}
        self.QSums = {}
        self.pi = {}
        self.pi_sums = {}
        self.regret_sums = {}
        for n in range(self.N):
            self.Q[n] = {}
            self.Q_bu[n] = {}
            self.QSums[n] = {}
            self.pi[n] = {}
            self.pi_sums[n] = {}
            self.regret_sums[n] = {}
            for s in self.getStates():
                self.Q[n][s] = {}
                self.Q_bu[n][s] = {}
                self.QSums[n][s] = {}
                self.pi[n][s] = {}
                self.regret_sums[n][s] = {}
                self.pi_sums[n][s] = {}
                for a in self.getActions(s,0):
                    self.Q[n][s][a] = 0.0
                    self.Q_bu[n][s][a] = 0.0
                    self.QSums[n][s][a] = 0.0
                    self.pi[n][s][a] = 1.0 / 4.0  # len(list(total_actions.keys())
                    self.regret_sums[n][s][a] = 0.0
                    self.pi_sums[n][s][a] = 0.0

        # self.grid = [' ', ' ', ' ',
        #              ' ', ' ', 10]

        self.livingReward = -1.0

    def getActions(self, s, n):
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
                successors.append([leftState,massLeft/2.0, self.getReward(state,action,0,0)])
                successors.append([rightState,massLeft/2.0, self.getReward(state,action,0,0)])

        if action == self.LEFT or action == self.RIGHT:
            if action == self.LEFT:
                successors.append([leftState,1.0-self.noise, self.getReward(state,action,0,0)])
            else:
                successors.append([rightState,1.0-self.noise, self.getReward(state,action,0,0)])

            massLeft = self.noise
            if massLeft > 0.0:
                successors.append([upState,massLeft/2.0, self.getReward(state,action,0,0)])
                successors.append([downState,massLeft/2.0, self.getReward(state,action,0,0)])

        return successors