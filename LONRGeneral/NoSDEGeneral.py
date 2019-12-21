from LONRGeneral.LONR import *
import copy
import numpy as np



class NoSDEGeneral(MDP):

    def __init__(self, startState=1, totalStates=2, totalActions=2):
        super().__init__(startState=startState)

        # Two player MG
        self.N = 2

        self.version = 1

        for i in range(10):
            print("SET GAMMA=0.75")

        seed = np.random.randint(100000)
        seed = 1111
        np.random.seed(seed)
        print("SEED: ", seed)

        #Randomize rewards
        self.extraRewards = []
        for er in range(8): #2^3? n^A
            newReward = np.random.randint(4)
            print("Rewards: ", newReward)
            self.extraRewards.append(newReward)

        self.total_states = [1, 2,]
        self.total_actions = [1, 2, 3]
        self.NOOP = self.total_actions[-1] + 1


        self.Q = {}
        self.Q_bu = {}
        self.QSums = {}
        self.pi = {}
        self.pi_sums = {}
        self.regret_sums = {}
        self.QTouched = {}
        self.last_imm_regret = {}

        self.weights = {}

        #Deterministic transition map n,s,a -> s
        self.TMap = {}
        # Rewards n,s,a -> R
        self.RMap = {}


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

            self.TMap[n] = {}
            self.RMap[n] = {}

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

                self.TMap[n][s] = {}
                self.RMap[n][s] = {}


                for a in self.getActions(s, n):
                    self.Q[n][s][a] = 0.0
                    self.Q_bu[n][s][a] = 0.0
                    self.QSums[n][s][a] = 0.0
                    self.QTouched[n][s][a] = 0.0
                    self.pi[n][s][a] = 1.0 / len(self.getActions(s, n))
                    self.regret_sums[n][s][a] = 0.0
                    self.pi_sums[n][s][a] = 0.0
                    self.weights[n][s][a] = 1.0
                    self.last_imm_regret[n][s][a] = 0.0
                    self.TMap[n][s][a] = -1
                    self.RMap[n][s][a] = -1




        #n 0-1, s 1-2
        # Every state that has actions>1 needs transitions
        # Only one player has more than one action in a state
        self.TMap[0][1][1] = 1
        self.TMap[0][1][2] = 2
        self.TMap[0][1][3] = 2
        self.TMap[1][2][1] = 2
        self.TMap[1][2][2] = 1
        self.TMap[1][2][3] = 1

        # self.TMap[0][1]["KEEP"] = 1
        # self.TMap[0][1]["SEND"] = 2
        # self.TMap[1][2]["KEEP"] = 2
        # self.TMap[1][2]["SEND"] = 1

        #n 0, 1
        #s 1, 2

        self.RMap[0][1][1] = 1 # Player 0, state 1, keep
        self.RMap[0][1][2] = 0
        self.RMap[0][1][3] = 0
        self.RMap[0][2][1] = 3 #reward for player 0, other state, other player's action
        self.RMap[0][2][2] = 0
        self.RMap[0][2][3] = 0

        self.RMap[1][2][1] = 1
        self.RMap[1][2][2] = 0
        self.RMap[1][2][3] = 0
        self.RMap[1][1][1] = 0
        self.RMap[1][1][2] = 3
        self.RMap[1][1][3] = 3

        # self.RMap[0][1]["KEEP"] = 1  # Player 0, state 1, keep
        # self.RMap[0][1]["SEND"] = 0
        # self.RMap[0][2]["KEEP"] = 3  # reward for player 0, other state, other player's action
        # self.RMap[0][2]["SEND"] = 0
        #
        # self.RMap[1][2]["KEEP"] = 1
        # self.RMap[1][2]["SEND"] = 0
        # self.RMap[1][1]["KEEP"] = 0
        # self.RMap[1][1]["SEND"] = 3



    def hasActions(self, n, s):

        if n == 0 and s % 2 == 1:
            return True
        elif n == 1 and s % 2 == 0:
            return True
        else:
            return False

        # if s == 1 and n == 0:
        #     return True
        # elif s == 2 and n == 1:
        #     return True
        # else:
        #     return False

    def getActions(self, s, n):
        # Even states, player 0 has all actions, other player 1 action (NOOP)

        if n == 0 and s % 2 == 1:
            return self.total_actions
        elif n == 1 and s % 2 == 0:
            return self.total_actions
        else:
            return [self.NOOP]

        # Left state
        # if s == 1:
        #     # Player0
        #     if n == 0:
        #         return self.total_actions# ["KEEP", "SEND"]
        #     # Player1
        #     elif n == 1:
        #         return [self.NOOP] #["NOOP"]#, "SEND", "KEEP"] # ADDING SEND AND KEEP
        # elif s == 2:
        #     # Player0
        #     if n == 0:
        #         return [self.NOOP] #["NOOP"]#, "SEND", "KEEP"] # ADDIND SEND AND KEEP
        #     elif n == 1:
        #         return self.total_actions # ["KEEP", "SEND"]
        # print("Missed: ", s, " ", n)

    def getStates(self):
        return self.total_states

    def getStateRep(self, s):
        return s

    def getStartState(self):
        return np.random.randint(1, 3)

    def getNextStates(self, n, currentState,randomAction0, randomAction1):

        return None
        # print("Called getNextStates()")
        # # Only one has an above 0 value
        # return max(self.TMap[n][currentState][randomAction0], self.TMap[n][currentState][randomAction1])

        # if n == 0:
        #     if currentState == 1:
        #         if randomAction0 == "SEND":
        #             currentState = 2
        #
        # if n == 1:
        #     if currentState == 2:
        #         if randomAction1 == "SEND":
        #             currentState = 1
        # return currentState

    # Need to fix this entire function here and in general.
    def getReward(self, s, a_current, n, a_notN):
        return None



    def getNextStatesAndProbs(self, s, a_current, n_current):


        successors = []
        otherN = 1 if n_current == 0 else 0



        # TODO: Generate a map from current[n_current][s][a_current] -> [ns, pi[otherN][s][actions_notCurrentN], reward])

        #TMap RMap
        #You want to return the successors of this n_current, s and a_current

        for actions_notCurrentN in self.getActions(s, otherN):
            if self.hasActions(n_current, s):
                successors.append([self.TMap[n_current][s][a_current], 1.0, self.RMap[n_current][s][a_current]])
            else:
                #KEEP/SEND
                # self.TMap[1][2]["KEEP"] = 2
                # self.RMap[0][2]["KEEP"] = 3
                sss = []
                transitionState = self.TMap[otherN][s][actions_notCurrentN] #0 2 keep
                prob = self.pi[otherN][s][actions_notCurrentN]
                rewardState = self.RMap[n_current][s][actions_notCurrentN]
                successors.append([transitionState, prob, rewardState])

        # # # NoSDE is hand-crafted and all in here for now.
        # for actions_notCurrentN in self.getActions(s, otherN):
        #     # if s == 1 and a_current == "KEEP" and n_current == 0:
        #     #     successors.append([1, self.pi[otherN][s][actions_notCurrentN], 1.0])
        #     #
        #     # elif s == 1 and a_current == "SEND" and n_current == 0:
        #     #     successors.append([2, self.pi[otherN][s][actions_notCurrentN], 0.0])
        #     if s == 2 and a_current == "NOOP" and n_current == 0:
        #         if actions_notCurrentN == "KEEP":
        #             rew = 3.0# - 3.0 + self.extraRewards[2]
        #             ns = 2
        #         else:
        #             rew = 0.0# + self.extraRewards[3]
        #             ns = 1
        #         successors.append([ns, self.pi[otherN][s][actions_notCurrentN], rew])
        #
        #         #successors.append([self.TMap[otherN][s][actions_notCurrentN],self.pi[otherN][s][actions_notCurrentN], self.RMap[n_current][s][actions_notCurrentN] ])
        #         # print(n_current, s, actions_notCurrentN)
        #         # print("SUCCS: ", successors)
        #         # print("TMAP : ", self.TMap[otherN][s][actions_notCurrentN])
        #         # print("RMAP : ", self.RMap[n_current][s][actions_notCurrentN])
        #         # print("pi   : ", self.pi[otherN][s][actions_notCurrentN])
        #     # elif s == 2 and a_current == "KEEP" and n_current == 1:
        #     #     successors.append([2, self.pi[otherN][s][actions_notCurrentN], 1.0])# - 1.0 + self.extraRewards[4]])
        #     # elif s == 2 and a_current == "SEND" and n_current == 1:
        #     #     successors.append([1, self.pi[otherN][s][actions_notCurrentN], 0.0])# + self.extraRewards[5]])
        #     elif s == 1 and a_current == "NOOP" and n_current == 1:
        #         if actions_notCurrentN == "KEEP":
        #             rew = 0.0# + self.extraRewards[6]
        #             ns = 1
        #         else:
        #             rew = 3.0# - 3.0 + self.extraRewards[7]
        #             ns = 2
        #         successors.append([ns, self.pi[otherN][s][actions_notCurrentN], rew])
        #
        #     # else:
        #     #     print("MISSING: ", actions_notCurrentN, " with: s:", s, " a_current: ", a_current, "n_current: ", n_current)

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