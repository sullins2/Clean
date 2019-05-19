from LONRGeneral.LONR import *
import copy
import numpy as np



class NoSDE(MDP):

    def __init__(self):
        super().__init__()

        # Two player MG
        self.N = 2

        self.total_states = [1, 2]

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
                for a in self.getActions(s, n):
                    self.Q[n][s][a] = 0.0
                    self.Q_bu[n][s][a] = 0.0
                    self.QSums[n][s][a] = 0.0
                    self.pi[n][s][a] = 1.0 / len(self.getActions(s, n))
                    self.regret_sums[n][s][a] = 0.0
                    self.pi_sums[n][s][a] = 0.0

    def getActions(self, s, n):
        # Left state
        if s == 1:
            # Player0
            if n == 0:
                return ["KEEP", "SEND"]
            # Player1
            elif n == 1:
                return ["NOOP"]
        elif s == 2:
            # Player0
            if n == 0:
                return ["NOOP"]
            elif n == 1:
                return ["KEEP", "SEND"]
        print("Missed: ", s, " ", n)

    def getStates(self):
        return self.total_states

    def getStateRep(self, s):
        return s

    # Need to fix this entire function here and in general.
    def getReward(self, s, a_current, n, a_notN):

        return None
        #return 0.0
        #Left state
        # if s == 1:
        #     # Player0
        #     if n == 0:
        #         if a_current == "KEEP":
        #             return 1.0
        #         elif a_current == "SEND":
        #             return 0.0
        #
        #     if n == 1:
        #         if a_current == "KEEP":
        #             return 0.0
        #         elif a_current == "SEND":
        #             return 3.0
        # # Right state
        # elif s == 2:
        #     # Player0
        #     if n == 0:
        #         if a_current == "KEEP":
        #             return 3.0
        #         elif a_current == "SEND":
        #             return 0.0
        #     # Player1
        #     elif n == 1:
        #         if a_current == "KEEP":
        #             return 1.0
        #         elif a_current == "SEND":
        #             return 0.0
        #
        # #Left State
        # if n == 0:
        #     notN = 1
        # elif n == 1:
        #     notN = 0
        # # if s == 1:
        # #     if n == 0 and n
        # # print("Reward called: ")
        # if s == 1:
        #     if n == 0:
        #         if a_current == "KEEP" and a_notN == "NOOP":
        #             return 1.0
        #         if a_current == "SEND" and a_notN == "NOOP":
        #             return 0.0
        #     elif n == 1:
        #         if a_current == "NOOP" and a_notN == "KEEP":
        #             return 0.0
        #         if a_current == "NOOP" and a_notN == "SEND":
        #             return 3.0
        # elif s == 2:
        #     if n == 0:
        #         if a_current == "NOOP" and a_notN == "KEEP":
        #             return 3.0
        #         if a_current == "NOOP" and a_notN == "SEND":
        #             return 0.0
        #     elif n == 1:
        #         if a_current == "KEEP" and a_notN == "NOOP":
        #             return 1.0
        #         if a_current == "SEND" and a_notN == "NOOP":
        #             return 0.0

        #self.getReward(s, actions_A, n_current, actions_B)
        print("Reward missed: ", "s: ", s, " a_current: ", a_current, " n: ", n, " a_notN: ", a_notN)


    def getNextStatesAndProbs(self, s, a_current, n_current):

        successors = []

        #print("NextSP: ", " s: ", s, " a_current:", a_current, " n_current: ", n_current)
        otherN = 1 if n_current == 0 else 0


        # NoSDE is hand-crafted and all in here for now.
        for actions_notCurrentN in self.getActions(s, otherN):

            if s == 1 and a_current == "KEEP" and n_current == 0:
                successors.append([1, self.pi[otherN][s][actions_notCurrentN], 1.0])
            elif s == 1 and a_current == "SEND" and n_current == 0:
                successors.append([2, self.pi[otherN][s][actions_notCurrentN], 0.0])
            elif s == 2 and a_current == "NOOP" and n_current == 0:
                if actions_notCurrentN == "KEEP":
                    rew = 3.0
                    ns = 2
                else:
                    rew = 0.0
                    ns = 1
                successors.append([ns, self.pi[otherN][s][actions_notCurrentN], rew])
            elif s == 2 and a_current == "KEEP" and n_current == 1:
                successors.append([2, self.pi[otherN][s][actions_notCurrentN], 1.0])
            elif s == 2 and a_current == "SEND" and n_current == 1:
                successors.append([1, self.pi[otherN][s][actions_notCurrentN], 0.0])
            elif s == 1 and a_current == "NOOP" and n_current == 1:
                if actions_notCurrentN == "KEEP":
                    rew = 0.0
                    ns = 1
                else:
                    rew = 3.0
                    ns = 2
                successors.append([ns, self.pi[otherN][s][actions_notCurrentN], rew])
            else:
                print(actions_notCurrentN, " with: s:", s, " a_current: ", a_current, "n_current: ", n_current)

        return successors

    def isTerminal(self, s):
        return False