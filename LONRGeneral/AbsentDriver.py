
import numpy as np
from random import shuffle

from LONRGeneral.LONR import *

class AbsentDriver(MDP):

    def __init__(self, noise=0.0, startState=None):
        super().__init__(startState=startState)

        # One player MDP
        self.N = 1

        # Action direction constants
        self.CONT = "CONT"
        self.EXIT = "EXIT"

        self.numberOfActions = 2

        # Non-determinism - this value is split between sideways moves
        self.noise = noise

        self.version = 0
        self.TLProb = 0.5

        self.Q = {}

        self.QA = {}
        self.QB = {}
        self.QA_bu = {}
        self.QB_bu = {}
        self.QASums = {}
        self.QBSums = {}
        self.QATouched = {}
        self.QBTouched = {}

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

            # Double Q Learning
            self.QA[n] = {}
            self.QB[n] = {}
            self.QA_bu[n] = {}
            self.QB_bu[n] = {}
            self.QASums[n] = {}
            self.QATouched[n] = {}
            self.QBSums[n] = {}
            self.QBTouched[n] = {}

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

                # Double Q Learning
                self.QA[n][s] = {}
                self.QB[n][s] = {}
                self.QA_bu[n][s] = {}
                self.QB_bu[n][s] = {}
                self.QASums[n][s] = {}
                self.QATouched[n][s] = {}
                self.QBSums[n][s] = {}
                self.QBTouched[n][s] = {}

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

                    # Double Q
                    self.QA[n][s][a] = 0.0
                    self.QB[n][s][a] = 0.0
                    self.QA_bu[n][s][a] = 0.0
                    self.QB_bu[n][s][a] = 0.0
                    self.QASums[n][s][a] = 0.0
                    self.QATouched[n][s][a] = 0.0
                    self.QBSums[n][s][a] = 0.0
                    self.QBTouched[n][s][a] = 0.0

                    self.pi[n][s][a] = 1.0 / len(self.getActions(s, 0)) # uniform init
                    self.regret_sums[n][s][a] = 0.0
                    self.pi_sums[n][s][a] = 0.0
                    self.weights[n][s][a] = 1.0
                    self.runningRewards[n][s][a] = 0.0




        self.livingReward = 0.0



    def getActions(self, s, n):
        if s == 0 or s == 1:
            return [self.CONT, self.EXIT]
        return ["No"]

    def getStates(self):
        # ls = [0,1,2,3,4]
        # shuffle(ls)
        if self.version == 0:
            return [0,1,2,3,4]
        if self.version == 1:
            return [0, 1, 2]
        else:
            return [1, 3, 4]
        #return [0,1,2,3,4]

    def getStateRep(self, s):
        if s == 0 or s == 1:
            return 0
        return s

    def getReward(self, s, a_current, n, a_notN):
        """ Return reward from leaving state s.

            Not dependent on action or s'
        """
        # if s == 0 and a_current == self.EXIT:
        #     return 0.0
        # if s == 0 and a_current == self.CONT:
        #     return 0.0
        # if s == 1 and a_current == self.EXIT:
        #     return 4.0
        # if s == 1 and a_current == self.CONT:
        #     return 1.0


        if self.isTerminal(s):
            if s == 2:
                return 0.0
            if s == 3:
                return 4.0
            if s == 4:
                return 1.0
        return 0.0


    # Only needed for LONR-A
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

        # print("S: ", s, " action: ", action, "ActionChoices: ", actionChoices)
        next_state = np.random.choice(actionChoices, p=actionProbs)
        return next_state

    def isTerminal(self, s):
        """Returns True if s is a terminal state.
        """
        if s == 2 or s == 3 or s == 4:
            return True
        else:
            return False


    def getNextStatesAndProbs(self, state, action, n_current):
        """ Returns a list of [Action, resulting state, transition probability]
        """
        successors = []
        if state == 0:
            if action == "CONT":
                successors.append([1, 1.0, 0.0])
            elif action == "EXIT":
                successors.append([2, 1.0, 0.0])
        if state == 1:
            if action == "CONT":
                successors.append([4, 1.0, 1.0])
            elif action == "EXIT":
                successors.append([3, 1.0, 4.0])

        # if action == self.U1:
        #     # 1/3 2/3
        #     successors.append([0, 1.0 / 3.0, self.getReward(state, action, 0, 0)])
        #     successors.append([1, 2.0 / 3.0, self.getReward(state, action, 0, 0)])
        # elif action == self.U2:
        #     successors.append([0, 2.0 / 3.0, self.getReward(state, action, 0, 0)])
        #     successors.append([1, 1.0 / 3.0, self.getReward(state, action, 0, 0)])
        # if successors == []:
        #     print("EMPTY")
        return successors
