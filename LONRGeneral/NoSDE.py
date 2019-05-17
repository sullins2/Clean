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

    def getReward(self, s, a_current, n, a_notN):

        return 0.0
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

        # Left State
        # if n == 0:
        #     notN = 1
        # elif n == 1:
        #     notN = 0
        # if s == 1:
        #     if n == 0 and n
        #print("Reward called: ")
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
        #print("Reward missed: ", "s: ", s, " a_current: ", a_current, " n: ", n, " a_notN: ", a_notN)


    def getNextStatesAndProbs(self, s, a_current, n_current):

        successors = []

        actions_A = a_current

        otherN = 1 if n_current == 0 else 0

        #print("s: ", s, " a_current: ", a_current, " n_current: ", n_current)
        #print("Actions(s, otherN):", self.getActions(s, otherN) )

        # actions in s=1, otherN=1
        #print("")
        for actions_B in self.getActions(s, otherN):

            # Left State
            # if s == 1:
            #     # Player0
            #     if n_current == 0:
            #         if actions_A
            new_state = 1
            reward = 1.0


            if s == 1 and a_current == "KEEP" and n_current == 0:
                successors.append([1, self.pi[otherN][s][actions_B], 1.0])
            elif s == 1 and a_current == "SEND" and n_current == 0:
                successors.append([2, self.pi[otherN][s][actions_B], 0.0])
            elif s == 2 and a_current == "NOOP" and n_current == 0:
                rew = 3.0 if actions_B == "KEEP" else 0.0
                if actions_B == "KEEP":
                    rew = 3.0
                    ns = 2
                else:
                    rew = 0.0
                    ns = 1
                successors.append([ns, self.pi[otherN][s][actions_B], rew])
            elif s == 2 and a_current == "KEEP" and n_current == 1:
                successors.append([2, self.pi[otherN][s][actions_B], 1.0])
            elif s == 2 and a_current == "SEND" and n_current == 1:
                successors.append([1, self.pi[otherN][s][actions_B], 0.0])
            elif s == 1 and a_current == "NOOP" and n_current == 1:
                rew = 0.0 if actions_B == "KEEP" else 3.0
                if actions_B == "KEEP":
                    rew = 0.0
                    ns = 1
                else:
                    rew = 3.0
                    ns = 2
                successors.append([ns, self.pi[otherN][s][actions_B], rew])
            else:
                print(actions_B, " with: s:", s, " a_current: ", a_current, "n_current: ", n_current)

        return successors

    def isTerminal(self, s):
        return False