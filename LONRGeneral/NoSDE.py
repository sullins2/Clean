from LONRGeneral.LONR import *
import copy
import numpy as np



class NoSDE(MDP):

    def __init__(self, startState=1):
        super().__init__(startState=startState)

        # Two player MG
        self.N = 2

        self.version = 1


        # -1 to 7/4
        self.x = 0.0# -1.0 / 2.0 #3.0 / 4.0

        for i in range(10):
            print("SET GAMMA=0.75")

        seed = np.random.randint(100000)
        seed = 31  # 37331#53945#75574
        np.random.seed(seed)
        print("SEED: ", seed)

        #Randomize rewards
        self.extraRewards = []
        for er in range(8):
            newReward = np.random.randint(4)
            print("Rewards: ", newReward)
            self.extraRewards.append(newReward)

        self.total_states = [1, 2]

        self.Q = {}
        self.Q_bu = {}
        self.QSums = {}
        self.pi = {}
        self.pi_sums = {}
        self.regret_sums = {}
        self.QTouched = {}
        self.last_imm_regret = {}

        self.weights = {}

        self.Q_other = {}
        self.QQQQ = {}

        for n in range(self.N):
            self.Q[n] = {}
            self.Q_bu[n] = {}
            self.QSums[n] = {}
            self.QTouched[n] = {}
            self.weights[n] = {}

            self.pi[n] = {}
            self.pi_sums[n] = {}
            self.regret_sums[n] = {}

            self.last_imm_regret[n] = {}

            for s in self.getStates():
                self.Q[n][s] = {}
                self.Q_bu[n][s] = {}
                self.QSums[n][s] = {}
                self.QTouched[n][s] = {}
                self.pi[n][s] = {}
                self.regret_sums[n][s] = {}
                self.pi_sums[n][s] = {}
                self.weights[n][s] = {}

                self.last_imm_regret[n][s] = {}

                for a in self.getActions(s, n):
                    self.Q[n][s][a] = 0.0
                    self.Q_bu[n][s][a] = 0.0
                    self.QSums[n][s][a] = 0.0
                    self.QTouched[n][s][a] = 0.0
                    self.pi[n][s][a] = 1.0 / len(self.getActions(s, n))
                    self.regret_sums[n][s][a] = 0.0#0.00001
                    # for rrr in range(10):
                    #     print("ERASE NOSDE")
                    self.pi_sums[n][s][a] = 0.0
                    self.weights[n][s][a] = 1.0

                    self.last_imm_regret[n][s][a] = 0.0

        for n in range(self.N):
            self.Q_other[n] = {}
            for s in [1,2]:
                self.Q_other[n][s] = {}
                for a in ["SEND", "KEEP"]:
                    self.Q_other[n][s][a] = 0.0


    def getActions(self, s, n):
        # Left state
        if s == 1:
            # Player0
            if n == 0:
                return ["KEEP", "SEND"]
            # Player1
            elif n == 1:
                return ["NOOP"]#, "SEND", "KEEP"] # ADDING SEND AND KEEP
        elif s == 2:
            # Player0
            if n == 0:
                return ["NOOP"]#, "SEND", "KEEP"] # ADDIND SEND AND KEEP
            elif n == 1:
                return ["KEEP", "SEND"]
        print("Missed: ", s, " ", n)

    def getStates(self):
        return self.total_states

    def getStateRep(self, s):
        return s

    def getStartState(self):
        return np.random.randint(1, 3)

    def getNextStates(self, n, currentState,randomAction0, randomAction1):

        if n == 0:
            if currentState == 1:
                if randomAction0 == "SEND":
                    currentState = 2

        if n == 1:
            if currentState == 2:
                if randomAction1 == "SEND":
                    currentState = 1
        return currentState

    # Need to fix this entire function here and in general.
    def getReward(self, s, a_current, n, a_notN):

        # x = 1.0 / 4.0
        #return 0.0
        #Left state
        #print("CALLED")
        return None
        # if s == 1:
        #     # Player0
        #     if n == 0:
        #         if a_current == "KEEP":
        #             return 1.0
        #         elif a_current == "SEND":
        #             return 0.0 + self.x
        #         # print("NO1")
        #
        #     if n == 1:
        #         if a_current == "KEEP":
        #             return 0.0
        #         elif a_current == "SEND":
        #             return 3.0
        #         # elif a_current == "NOOP":
        #         #     if a_notN == "SEND":
        #         #         return 3.0
        #         #     else:
        #         #         return 0.0
        # # Right state
        # elif s == 2:
        #     # Player0
        #     if n == 0:
        #         if a_current == "KEEP":
        #             return 3.0 + self.x
        #         elif a_current == "SEND":
        #             return 0.0
        #         # elif a_current == "NOOP":
        #         #     if a_notN == "SEND":
        #         #         return 0.0
        #         #     else:
        #         #         return 3.0
        #
        #     # Player1
        #     elif n == 1:
        #         if a_current == "KEEP":
        #             return 1.0
        #         elif a_current == "SEND":
        #             return 0.0
        #         # print("NO4")
        #
        #     # elif s == 2 and actions_notCurrentN == "KEEP" and n_current == 0:
        #     #     rew = 3.0
        #     #     ns = 2
        #     #     successors.append([ns, self.pi[otherN][s][actions_notCurrentN], rew])
        #     # elif s == 2 and actions_notCurrentN == "SEND" and n_current == 0:
        #     #     rew = 0.0
        #     #     ns = 1
        #     #     successors.append([ns, self.pi[otherN][s][actions_notCurrentN], rew])
        #     # elif s == 1 and actions_notCurrentN == "KEEP" and n_current == 1:
        #     #     rew = 0.0
        #     #     ns = 1
        #     #     successors.append([ns, self.pi[otherN][s][actions_notCurrentN], rew])
        #     # elif s == 1 and actions_notCurrentN == "SEND" and n_current == 1:
        #     #     rew = 3.0
        #     #     ns = 2
        #     #     successors.append([ns, self.pi[otherN][s][actions_notCurrentN], rew])
        #
        # else:
        #     print("MISISNG")
        #     return 0.0



    def getNextStatesAndProbs(self, s, a_current, n_current):


        successors = []
        otherN = 1 if n_current == 0 else 0


        # TODO: Generate a map from current[n_current][s][a_current] -> [ns, pi[otherN][s][actions_notCurrentN], reward])

        # NoSDE is hand-crafted and all in here for now.
        for actions_notCurrentN in self.getActions(s, otherN):
            if s == 1 and a_current == "KEEP" and n_current == 0:
                successors.append([1, self.pi[otherN][s][actions_notCurrentN], 1.0 - 1.0 + self.extraRewards[0]])
            elif s == 1 and a_current == "SEND" and n_current == 0:
                successors.append([2, self.pi[otherN][s][actions_notCurrentN], 0.0 + self.extraRewards[1]])
            elif s == 2 and a_current == "NOOP" and n_current == 0:
                if actions_notCurrentN == "KEEP":
                    rew = 3.0 - 3.0 + self.extraRewards[2]
                    ns = 2
                else:
                    rew = 0.0 + self.extraRewards[3]
                    ns = 1
                successors.append([ns, self.pi[otherN][s][actions_notCurrentN], rew])
            elif s == 2 and a_current == "KEEP" and n_current == 1:
                successors.append([2, self.pi[otherN][s][actions_notCurrentN], 1.0 - 1.0 + self.extraRewards[4]])
            elif s == 2 and a_current == "SEND" and n_current == 1:
                successors.append([1, self.pi[otherN][s][actions_notCurrentN], 0.0 + self.extraRewards[5]])
            elif s == 1 and a_current == "NOOP" and n_current == 1:
                if actions_notCurrentN == "KEEP":
                    rew = 0.0 + self.extraRewards[6]
                    ns = 1
                else:
                    rew = 3.0 - 3.0 + self.extraRewards[7]
                    ns = 2
                successors.append([ns, self.pi[otherN][s][actions_notCurrentN], rew])

            else:
                print("MISSING: ", actions_notCurrentN, " with: s:", s, " a_current: ", a_current, "n_current: ", n_current)

        return successors

    def isTerminal(self, s):
        return False
















    # def getMove(self, s, action):
    #     if s == 1 and action == "KEEP":
    #         return 1
    #     elif s == 1 and action == "SEND":
    #         return 2
    #     elif s == 2 and action == "NOOP":
    #        return 2
    #     elif s == 2 and action == "KEEP":
    #         return 2
    #     elif s == 2 and action == "SEND":
    #         return 1
    #     elif s == 1 and action == "NOOP":
    #        return 1
    #     print("GETMOVE() missed:  s: ", s, "  action: ",action)

    ####NOT SURE
    # elif s == 2 and actions_notCurrentN == "KEEP" and n_current == 0:
    #     rew = 3.0
    #     ns = 2
    #     successors.append([ns, self.pi[n_current][s][actions_notCurrentN], rew, actions_notCurrentN])
    # elif s == 2 and actions_notCurrentN == "SEND" and n_current == 0:
    #     rew = 0.0
    #     ns = 1
    #     successors.append([ns, self.pi[n_current][s][actions_notCurrentN], rew, actions_notCurrentN])
    # elif s == 1 and actions_notCurrentN == "KEEP" and n_current == 1:
    #     rew = 0.0
    #     ns = 1
    #     successors.append([ns, self.pi[n_current][s][actions_notCurrentN], rew, actions_notCurrentN])
    # elif s == 1 and actions_notCurrentN == "SEND" and n_current == 1:
    #     rew = 3.0
    #     ns = 2
    #     successors.append([ns, self.pi[n_current][s][actions_notCurrentN], rew, actions_notCurrentN])