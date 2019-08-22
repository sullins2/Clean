
import numpy as np
import math
import random


#Change this to reflect that this is whatever is being fed in
class MDP(object):

    def __init__(self, startState=None):

        # Initialized here but must be defined/created in created class
        self.Q = None
        self.Q_bu = None
        self.QSums = None
        self.QTouched = None
        self.pi = None
        self.pi_sums = None
        self.regret_sums = None

        self.weights = None
        self.runningRewards = None

        # For LONR-A, a starting state should be specified
        self.startState = startState

    #TODO change to n,s
    def getActions(self, s, n):
        """ Returns list of actions

        """
        raise NotImplementedError("Please implement getActions method")


    # TODO: change to (n, s, a)
    def getNextStatesAndProbs(self, s, a_current, n_current):
        """ Returns list of [ [next_state, prob, rew] ]

        """
        raise NotImplementedError("Please implement getNextStatesAndProbs method")


    def getReward(self, s, a_current, n, a_notN):
        """ s=state, a_current=actionOfPlayer n, n is current player, a_notN is action of other player
            Joint Actions required for Markov Games
            a_notN can be set to 0 for MDPs and ignored

        Returns reward(s, a, a_other)

        """
        raise NotImplementedError("Please implement getReward method")

    def getStates(self):
        """Return List of States

        """
        raise NotImplementedError("Please implement getStates method")

    def isTerminal(self, s):
        raise NotImplementedError("Please implement isTerminal method")


    # The following are only needed for LONR-A

    def getMove(self, s, a):
        """Used to get actual move in the environment for O-LONR
        """
        raise NotImplementedError("Please implement getMove method")


    # This is used to handle the tiger game, tbd..
    def getStateRep(self, s):
        """ O-LONR use only

            For MDPs: simply returns state
            For TigerGame: Maps both to one (EX: TLL & TRL -> TLL) aka one set of Q Values
        """
        raise NotImplementedError("Please implement getStateRep method")




class LONR(object):

    # M: Input to solve. For example, Markov game (MDP, markov game, tiger game)
    # parameters: alpha, epsilon, gamma
    # regret_minimizers: RM, RM+, DCFR
    # randomize: for TigerGame

    def __init__(self, M=None, parameters=None, regret_minimizers=None, dcfr=None, randomize=False, showSettings=True, EXP3=False,exp3gamma=0.5):


        # Problem input
        self.M = M

        # Get parameters
        self.alpha = parameters['alpha'] if 'alpha' in parameters else 1.0
        self.epsilon = parameters['epsilon'] if 'epsilon' in parameters else 15
        self.gamma = parameters['gamma'] if 'gamma' in parameters else 1.0
        self.alphaDecay = parameters['alphaDecay'] if 'alphaDecay' in parameters else 1.0

        # Get regret minimizers used - note, no check if more than one is true
        #   - Left because there is no "conflict" between all being true
        self.RM = regret_minimizers['RM'] if 'RM' in regret_minimizers else True
        self.RMPLUS = regret_minimizers['RMPlus'] if 'RMPlus' in regret_minimizers else False
        self.DCFR = regret_minimizers['DCFR'] if 'DCFR' in regret_minimizers else False

        if sum([self.RM,self.RMPLUS,self.DCFR]) >= 2:
            print("More than one regret minimizer is true. Select only one.")
            exit()
        elif sum([self.RM,self.RMPLUS,self.DCFR]) == 0:
            print("No regret minimizer is true. Select only one.")
            exit()

        # Set regret parameters:
        self.alphaDCFR = None
        self.betaDCFR = None
        self.gammaDCFR = None

        # If Regret Matching (check this)
        if self.RM:
            self.alphaDCFR = 1.0
            self.betaDCFR = 1.0
            self.gammaDCFR = 1.0

        # Regret Matching + (via DCFR/parameters)
        elif self.RMPLUS:
            self.alphaDCFR = math.inf
            self.betaDCFR = -math.inf
            self.gammaDCFR = 2.0

        # DCFR


        elif self.DCFR:
            self.alphaDCFR = dcfr['alphaDCFR'] if 'alphaDCFR' in dcfr else 3.0 / 2.0
            self.betaDCFR = dcfr['betaDCFR'] if 'betaDCFR' in dcfr else 0.0
            self.gammaDCFR = dcfr['gammaDCFR'] if 'gammaDCFR' in dcfr else 2.0

        # This should be gone, will have:
        # LONR_V, LONR_A, LONR_B


        # This is for Tiger game, not great, fix
        self.randomize = randomize

        self.showSettings = showSettings

        # Plot stuff, figure out a better way
        # self.QSums = []
        # self.iters = []



        #EXP3 = False, exp3gamma = 0.5
        self.EXP3 = False
        self.exp3gamma = 0.1

        # Get rid of, will have LONR_V, etc
        # self.LONR = LONR

        if self.showSettings:
            print("LONR initialized with:")
            print("   gamma: ", self.gamma)
            print("   alpha: ", self.alpha)
            print(" epsilon: ", self.epsilon)
            print("     rm+: ", self.RMPLUS)
            print("    DCFR: ", self.DCFR)
            if self.DCFR or self.RMPLUS:
                print(" alphaDCFR: ", self.alphaDCFR)
                print("  betaDCFR: ", self.betaDCFR)
                print(" gammaDCFR: ", self.gammaDCFR)
            print("    EXP3: ", self.EXP3)
            print("exp3gama: ", self.exp3gamma)


    def lonr_train(self, iterations, log):
        raise NotImplementedError("Please implement lonr_train method")

    def _lonr_train(self, t):
        raise NotImplementedError("Please implement _lonr_train method")


    def printOut(self):

        print("Q Avg")
        for n in range(self.M.N):
            for k in sorted(self.M.QSums[n].keys()):
                tot = 0.0
                print(k, ": ", end='')
                for kk in self.M.QSums[n][k].keys():
                    print(kk, ": ", self.M.QSums[n][k][kk] / max(1.0, self.M.QTouched[n][k][kk]), " ", end='')
                print("")

        print("Q")
        for n in range(self.M.N):
            for k in sorted(self.M.Q[n].keys()):
                print(k, ": ", end='')
                for kk in self.M.Q[n][k].keys():
                    touched = max(self.M.QTouched[n][k][kk], 1.0)
                    #if touched == 0: touched = 1.0
                    print(kk, ": ", self.M.Q[n][k][kk], " ", end='')
                print("")

        print("Pi")
        for n in range(self.M.N):
            for k in sorted(self.M.pi[n].keys()):
                print(k, ": ", end='')
                for kk in self.M.pi[n][k].keys():
                    print(kk, ": ", self.M.pi[n][k][kk], " ", end='')
                print("")

        print("Pi Sums")
        for n in range(self.M.N):
            for k in sorted(self.M.pi_sums[n].keys()):
                print(k, ": ", end='')
                normalize = 0.0
                for kk in self.M.pi_sums[n][k].keys():
                    normalize += self.M.pi_sums[n][k][kk]
                if normalize <= 0:
                    normalize = 1.0
                for kk in self.M.pi_sums[n][k].keys():
                    print(kk, ": ", self.M.pi_sums[n][k][kk] / normalize, " ", end='')
                print("")


        print("Regret Sums")
        for n in range(self.M.N):
            for k in sorted(self.M.regret_sums[n].keys()):
                print(k, ": ", end='')
                for kk in self.M.regret_sums[n][k].keys():
                    print(kk, ": ", self.M.regret_sums[n][k][kk], " ", end='')
                print("")

        # print("Weights")
        # if self.M.weights is not None:
        #     for n in range(self.M.N):
        #         for k in sorted(self.M.weights[n].keys()):
        #             print(k, ": ", end='')
        #             for kk in self.M.weights[n][k].keys():
        #                 print(kk, ": ", self.M.weights[n][k][kk], " ", end='')
        #             print("")
        #
        # print("Touched")
        # if self.M.QTouched is not None:
        #     for n in range(self.M.N):
        #         for k in sorted(self.M.QTouched[n].keys()):
        #             print(k, ": ", end='')
        #             for kk in self.M.QTouched[n][k].keys():
        #                 print(kk, ": ", self.M.QTouched[n][k][kk], " ", end='')
        #             print("")


    def normalize_pi_sums(self, takeMax=False):
        if takeMax == False:
            for k in sorted(self.M.pi_sums[0].keys()):
                tot = 0.0
                for kk in self.M.pi_sums[0][k].keys():
                    tot += self.M.pi_sums[0][k][kk]
                if tot == 0: tot = 1.0
                for kk in self.M.pi_sums[0][k].keys():
                    self.M.pi_sums[0][k][kk] = self.M.pi_sums[0][k][kk] / tot
        # Make max = 1.0, else 0.0
        else:
            for k in sorted(self.M.pi_sums[0].keys()):
                maxPi = 0.0
                maxAction = 0

                for kk in self.M.pi_sums[0][k].keys():
                    if self.M.pi_sums[0][k][kk] > maxPi:
                        maxPi = self.M.pi_sums[0][k][kk]
                        maxAction = kk
                #if tot == 0: tot = 1.0
                for kk in self.M.pi_sums[0][k].keys():
                    if kk == maxAction:
                        self.M.pi_sums[0][k][kk] = 1.0
                    else:
                        self.M.pi_sums[0][k][kk] = 0.0
                        # self.M.pi_sums[0][k][kk] = self.M.pi_sums[0][k][kk] / tot





    def QUpdate(self, n, s, a_current2, randomS=None):

        # print("IN:  s: ", s, " RANDOMS: ", randomS)
        for a_current in a_current2:

            if self.M.isTerminal(s):

                # Value iteration
                reward = self.M.getReward(s, a_current, n, n)
                # print("Reward: ", reward)
                if reward == None:
                    reward = 0.0
                # if s == 4:
                #     print("VALUUUUUUUUUUUUUUUUUUUUUUUUE: ", reward)
                self.M.Q_bu[n][s][a_current] = (1.0 - self.alpha)*reward + self.alpha*self.M.Q[n][s][a_current]
                # self.M.QSums[n][s][a_current] += self.M.Q_bu[n][s][a_current]
                # self.M.QTouched[n][s][a_current] += 1.0
                continue


            # Get next states and transition probs for MDP
            # Get next states and other players policy for MG
            succs = self.M.getNextStatesAndProbs(s, a_current, n)
            # print("s: ", s, " a:", a_current, "   succs: ", succs)

            #print("  SUCCS: ", succs)
            Value = 0.0

            # Loop thru each next state and prob
            for s_prime, prob, reward in succs:
                # print(" - s_prime: ", s_prime, "  prob: ", prob, "  reward: ", reward)

                # Terminal reached
                if s_prime == None:
                    #print("HERE: ", reward)
                    # self.M.Q_bu[n][s][a_current] = (1.0 - self.alpha) * self.M.Q[n][s][a_current] + self.alpha*reward
                    Value += prob * reward
                    #Value += reward
                    continue

                tempValue = 0.0

                # print("sprime: ", s_prime)
                # Loop thru actions in s_prime for player n
                for a_current_prime in self.M.getActions(s_prime, n):

                    tempValue += self.M.Q[n][s_prime][a_current_prime] * self.M.pi[n][self.M.getStateRep(s_prime)][a_current_prime]

                reww = self.M.getReward(s, a_current, n, 0)

                if reww is None:
                    Value += prob * (reward + self.gamma * tempValue)
                else:
                    Value += prob * (reww + self.gamma * tempValue)


                ###################################################
                # for a_current_prime in self.M.getActions(s_prime, n):
                #
                #     tempValue += self.M.Q[n][s_prime][a_current_prime] * self.M.pi[n][self.M.getStateRep(s_prime)][a_current_prime]
                ###################################################

            self.M.Q_bu[n][s][a_current] = (1.0 - self.alpha) * self.M.Q[n][s][a_current] + (self.alpha) * Value
            # print("Q: ", self.M.Q_bu[n][s][a_current], " VALUE: ", Value)
            # self.M.QSums[n][s][a_current] += self.M.Q[n][s][a_current]
            # self.M.QBSums[n][self.M.getStateRep(s)][a_current] += self.M.QB[n][self.M.getStateRep(s)][a_current]
            # self.M.QTouched[n][s][a_current] += 1.0
            #self.M.QBTouched[n][self.M.getStateRep(s)][a_current] += 1.0



    def QBackup(self, n, WW=None):

        # def QBackup(self, n, WW=None):
        #
        for s in self.M.getStates():
        # totalStates = [0,1,2,3,4]
        # for s in totalStates:
            for a in self.M.getActions(s, n):
                #print("QQ: ",  self.M.Q[n][s][a], " QQBU: ", self.M.Q_bu[n][s][a])
                self.M.Q[n][s][a] = self.M.Q_bu[n][s][a]
                # self.M.QSums[n][s][a] += self.M.Q_bu[n][s][a]
                #self.M.QTouched[n][s][a] += 1.0
                # self.M.QSums[n][self.M.getStateRep(s)][a] += self.M.Q[n][self.M.getStateRep(s)][a]

        # if WW is None:
        #     for s in self.M.getStates():
        #         for a in self.M.getActions(s, n):
        #             self.M.Q[n][s][a] = self.M.Q_bu[n][s][a]
        #             self.M.QSums[n][self.M.getStateRep(s)][a] += self.M.Q[n][self.M.getStateRep(s)][a]
        #
        # else:
        #     for s in WW:
        #         for a in self.M.getActions(s, n):
        #             self.M.Q[n][self.M.getStateRep(s)][a] = self.M.Q_bu[n][self.M.getStateRep(s)][a]
        #             self.M.QSums[n][self.M.getStateRep(s)][a] += self.M.Q[n][self.M.getStateRep(s)][a]

    def regretUpdate(self, n, currentState, t):

        iters = t + 1

        alphaW = pow(iters, self.alphaDCFR)
        alphaWeight = (alphaW / (alphaW + 1))

        betaW = pow(iters, self.betaDCFR)
        betaWeight = (betaW / (betaW + 1))


        gammaWeight = pow((iters / (iters + 1)), self.gammaDCFR)
        #print("BETAWEIGHT: ", alphaWeight, betaWeight, gammaWeight)

        # Calculate regret - this variable name needs a better name
        target = 0.0

        # Skip terminal states
        if self.M.isTerminal(currentState) == True:
            return

        for a in self.M.getActions(currentState, n):
            target += self.M.Q[n][currentState][a] * self.M.pi[n][self.M.getStateRep(currentState)][a]

        for a in self.M.getActions(currentState, n):
            action_regret = self.M.Q[n][currentState][a] - target
            #print("ACTION_REGRET: ", action_regret, self.M.Q[n][currentState][a], target)

            # if self.DCFR:
            #     if action_regret > 0:
            #         action_regret *= alphaWeight
            #     else:
            #         action_regret *= betaWeight

            # if action_regret > 3:
            #     print("AR: ", action_regret, " CS: ", currentState, "  CA: ", a)
            # if action_regret < 0:
            #     action_regret = 0

            # RMPLUS = False
            # if self.RMPLUS:
            #     self.M.regret_sums[n][self.M.getStateRep(currentState)][a] = max(0.0, self.M.regret_sums[n][self.M.getStateRep(currentState)][a] + action_regret)
            # else:
            self.M.regret_sums[n][self.M.getStateRep(currentState)][a] += action_regret

            # if self.DCFR:
            #     if self.M.regret_sums[n][self.M.getStateRep(currentState)][a] > 0:
            #         self.M.regret_sums[n][self.M.getStateRep(currentState)][a] *= alphaWeight
            #     else:
            #         self.M.regret_sums[n][self.M.getStateRep(currentState)][a] *= betaWeight



        for a in self.M.getActions(currentState, n):

            if self.M.isTerminal(currentState):
                continue
            # # Sum up total regret
            rgrt_sum = 0.0
            for k in self.M.regret_sums[n][self.M.getStateRep(currentState)].keys():
                rgrt_sum += self.M.regret_sums[n][self.M.getStateRep(currentState)][k] if self.M.regret_sums[n][self.M.getStateRep(currentState)][k] > 0 else 0.0
            if rgrt_sum > 0:
                self.M.pi[n][self.M.getStateRep(currentState)][a] = (max(self.M.regret_sums[n][self.M.getStateRep(currentState)][a], 0.)) / rgrt_sum
            else:
                self.M.pi[n][self.M.getStateRep(currentState)][a] = 1.0 / len(self.M.getActions(currentState, n))

            # Add to policy sum
            if self.DCFR:
                self.M.pi_sums[n][self.M.getStateRep(currentState)][a] += self.M.pi[n][self.M.getStateRep(currentState)][a] * gammaWeight
            else:

                self.M.pi_sums[n][self.M.getStateRep(currentState)][a] += self.M.pi[n][self.M.getStateRep(currentState)][a]




    def regretUpdate2(self, n, currentState, t, action):

        iters = t + 1
        alphaR = 3.0 / 2.0  # accum pos regrets
        betaR = 0.0  # accum neg regrets
        gammaR = 2.0  # contribution to avg strategy

        alphaW = pow(iters, self.alphaDCFR)
        alphaWeight = (alphaW / (alphaW + 1))

        betaW = pow(iters, self.betaDCFR)
        betaWeight = (betaW / (betaW + 1))

        gammaWeight = pow((iters / (iters + 1)), self.gammaDCFR)

        # Calculate regret - this variable name needs a better name
        target = 0.0
        action_regret = 0.0
        # Skip terminal states
        if self.M.isTerminal(currentState) == True:
            return

        for a in self.M.getActions(currentState, n):
            target += self.M.Q[n][currentState][a] * self.M.pi[n][self.M.getStateRep(currentState)][a]

        for a in self.M.getActions(currentState, n):

            if a != action:
                continue

            action_regret = self.M.Q[n][currentState][a] - target
            if self.DCFR or self.RMPLUS:
                if action_regret > 0:
                    action_regret *= alphaWeight
                else:
                    action_regret *= betaWeight

            # if action_regret < 0:
            #     action_regret = 0.0

            # RMPLUS = False
            if self.RMPLUS:
                self.M.regret_sums[n][self.M.getStateRep(currentState)][a] = max(0.0, self.M.regret_sums[n][self.M.getStateRep(currentState)][a] + action_regret)
            else:
                self.M.regret_sums[n][currentState][a] += action_regret


        # if t % 100 != 0:
        #     return
        # Skip terminal states
        # if self.M.isTerminal(self.M.getStateRep(currentState)) == True:
        #     continue

        for a in self.M.getActions(currentState, n):

            # if a != action:
            #     continue

            if self.M.isTerminal(currentState):
                continue
            # # Sum up total regret
            rgrt_sum = 0.0
            for k in self.M.regret_sums[n][self.M.getStateRep(currentState)].keys():
                rgrt_sum += self.M.regret_sums[n][self.M.getStateRep(currentState)][k] if self.M.regret_sums[n][self.M.getStateRep(currentState)][k] > 0 else 0.0
            if rgrt_sum > 0:
                self.M.pi[n][self.M.getStateRep(currentState)][a] = (max(self.M.regret_sums[n][self.M.getStateRep(currentState)][a], 0.)) / rgrt_sum
            else:
                # print("REG SUMS: ", rgrt_sum)
                self.M.pi[n][self.M.getStateRep(currentState)][a] = 1.0 / len(self.M.getActions(currentState, n))
                # print("  PI: ", self.M.pi[n][self.M.getStateRep(currentState)][a])

            # Add to policy sum
            if self.DCFR or self.RMPLUS:
                self.M.pi_sums[n][self.M.getStateRep(currentState)][a] += self.M.pi[n][self.M.getStateRep(currentState)][a] * gammaWeight
            else:

                self.M.pi_sums[n][self.M.getStateRep(currentState)][a] += self.M.pi[n][self.M.getStateRep(currentState)][a]

        return action_regret

    def regretUpdateNEW(self, n, currentState, t, reach, cfr_reach):

        # print("")
        # print("reach: ", reach, " CFR_REACH: ", cfr_reach)
        # print("")

        iters = t + 1

        alphaW = pow(iters, self.alphaDCFR)
        alphaWeight = (alphaW / (alphaW + 1))

        betaW = pow(iters, self.betaDCFR)
        betaWeight = (betaW / (betaW + 1))

        gammaWeight = pow((iters / (iters + 1)), self.gammaDCFR)


        # Calculate regret - this variable name needs a better name
        target = 0.0

        # Skip terminal states
        if self.M.isTerminal(currentState) == True:
            return

        for a in self.M.getActions(currentState, n):
            target += self.M.Q[n][currentState][a] * self.M.pi[n][self.M.getStateRep(currentState)][a]

        for a in self.M.getActions(currentState, n):
            action_regret = self.M.Q[n][currentState][a] - target
            ### action_regret *= cfr_reach

            if self.DCFR:
                if action_regret > 0:
                    action_regret *= alphaWeight
                else:
                    action_regret *= betaWeight

            if self.RMPLUS or self.RM:
                if action_regret < 0:
                    action_regret *= 0.0#alphaWeight
            # RMPLUS = False
            if self.RMPLUS:
                self.M.regret_sums[n][self.M.getStateRep(currentState)][a] = max(0.0, self.M.regret_sums[n][self.M.getStateRep(currentState)][a] + action_regret)
            else:
                # self.M.regret_sums[n][currentState][a] += action_regret
                self.M.regret_sums[n][self.M.getStateRep(currentState)][a] += action_regret

        # Skip terminal states
        # if self.M.isTerminal(self.M.getStateRep(currentState)) == True:
        #     continue

        for a in self.M.getActions(currentState, n):

            if self.M.isTerminal(currentState):
                continue
            # # Sum up total regret
            rgrt_sum = 0.0
            for k in self.M.regret_sums[n][self.M.getStateRep(currentState)].keys():
                rgrt_sum += self.M.regret_sums[n][self.M.getStateRep(currentState)][k] if self.M.regret_sums[n][self.M.getStateRep(currentState)][k] > 0 else 0.0
            if rgrt_sum > 0:
                self.M.pi[n][self.M.getStateRep(currentState)][a] = (max(self.M.regret_sums[n][self.M.getStateRep(currentState)][a], 0.)) / rgrt_sum
            else:
                #print("REG SUMS: ", rgrt_sum)
                self.M.pi[n][self.M.getStateRep(currentState)][a] = 1.0 / len(self.M.getActions(currentState, n))
                #print("  PI: ", self.M.pi[n][self.M.getStateRep(currentState)][a])

            # Add to policy sum
            if self.DCFR:
                self.M.pi_sums[n][self.M.getStateRep(currentState)][a] += self.M.pi[n][self.M.getStateRep(currentState)][a] * gammaWeight### * reach
            elif self.RMPLUS:
                self.M.pi_sums[n][self.M.getStateRep(currentState)][a] += self.M.pi[n][self.M.getStateRep(currentState)][a] ###* reach   # * gammaWeight==1
            else:

                self.M.pi_sums[n][self.M.getStateRep(currentState)][a] += self.M.pi[n][self.M.getStateRep(currentState)][a] # * gammaWeight ###* reach

################################################################
# LONR Value Iteration
################################################################
class LONR_V(LONR):

    def __init__(self, M=None, parameters=None, regret_minimizers=None, dcfr=None, randomize=True):
        super().__init__(M=M, parameters=parameters, regret_minimizers=regret_minimizers, dcfr=dcfr)
        self.randomize = True

    ################################################################
    # LONR Value Iteration
    ################################################################
    def lonr_train(self, iterations=-1, log=-1, randomize=False):

        if log != -1: print("Starting training..")

        print("self.M.version: ", self.M.version)

        if self.M.version == 0:
            self.M.version = 1

        for t in range(1, iterations + 1):
            # print("Version: ", self.M.version)
            # self.M.version = 2
            if (t + 0) % log == 0:
                print("Iteration: ", t + 0, " alpha: ", self.alpha, " gamma: ", self.gamma)

            # Call one full update via LONR-V
            self._lonr_train(t=t)

            # No-op unless alphaDecay is not 1.0
            self.alpha *= self.alphaDecay
            self.alpha = max(0.0, self.alpha)

            self.randomize = True

            if randomize:
                TigerOnLeftProb = self.M.TLProb
                v = np.random.choice([1, 2], p=[TigerOnLeftProb, 1.0 - TigerOnLeftProb])
                self.M.version = v
                if self.M.version == 1:
                    self.M.version = 2
                else:
                    self.M.version = 1
                # self.M.version = v


        if log != -1: print("Finish Training")


    ######################################
    ## LONR VALUE ITERATION
    ######################################
    def _lonr_train(self, t):
        """ One full update via LONR-V
        """

        # Q Update
        for n in range(self.M.N):


            # Loop through all states
            for s in self.M.getStates():

                if self.M.isTerminal(s):
                    a = self.M.getActions(s, n)
                    self.QUpdate(n, s, a, randomS=s)

                    continue

                # Loop through actions of current player n
                # for a in self.M.getActions(s, n):
                a = self.M.getActions(self.M.getStateRep(s), n)
                #if self.randomize == False:


                self.QUpdate(n, self.M.getStateRep(s), a, randomS=None)

                for aa in a:
                    self.M.QSums[n][s][aa] += self.M.Q_bu[n][s][aa]
                    self.M.QTouched[n][s][aa] += 1.0


        # Q Backup
        for n in range(self.M.N):
            self.QBackup(n)


        for n in range(self.M.N):
            for s in self.M.getStates():
                self.regretUpdate(n, s, t)


################################################################
# LONR Asynchronous Value Iteration
################################################################
class LONR_A(LONR):

    def __init__(self, M=None, parameters=None, regret_minimizers=None, dcfr=None):
        super().__init__(M=M, parameters=parameters, regret_minimizers=regret_minimizers, dcfr=dcfr)

    def lonr_train(self, iterations=-1, log=-1, randomize=False):


        if log != -1: print("Starting training..")
        for t in range(1, iterations+1):

            if self.M.version == 0:
                self.M.version = 1

            if (t+1) % log == 0:
                print("Iteration: ", t+1, " alpha:", self.alpha, " epsilon: ", self.epsilon)

            self._lonr_train(t=t)#, totalIterations=iterations, randomized=randomized)

            self.alpha *= self.alphaDecay
            self.alpha = max(0.0, self.alpha)



            if randomize:
                # if self.M.version == 1:
                #     self.M.version = 2
                # else:
                #     self.M.version = 1
                # TigerOnLeftProb = self.M.TLProb
                # v = np.random.choice([1,2], p=[TigerOnLeftProb, 1.0-TigerOnLeftProb])
                # self.M.version = v
                # TigerOnLeftProb = self.M.TLProb
                # v = np.random.choice([1, 2], p=[TigerOnLeftProb, 1.0 - TigerOnLeftProb])
                # # self.M.version = v
                if self.M.version == 1:
                    self.M.version = 2
                else:
                    self.M.version = 1


        if log != -1: print("Finished Training")

    ################################################################
    # LONR Asynchronous Value Iteration
    ################################################################
    def _lonr_train(self, t):


        currentState = self.M.startState


        done = False
        n = 0  # One player

        # Episode loop - until terminal state is reached
        while done == False:

            # Check this - tiger game stuff
            if self.M.getStateRep(currentState) == None:
                done = True
                continue

            # Check this - catching terminals
            if self.M.isTerminal(self.M.getStateRep(currentState)):
                a = self.M.getActions(currentState, 0)
                self.QUpdate(n, self.M.getStateRep(currentState), a, self.M.getStateRep(currentState))
                #for aa in a:

                for n in range(self.M.N):
                    self.QBackup(n)
                done = True
                continue

            # if self.M.isTerminal(self.M.getStateRep(currentState)):
            #     a = self.M.getActions(self.M.getStateRep(currentState), 0)
            #     self.QUpdate(n, self.M.getStateRep(currentState), a, self.M.getStateRep(currentState))
            #     done = True
            #     continue


            # Get possible actions
            a = self.M.getActions(currentState, 0)


            # Update Q of actions
            self.QUpdate(n, currentState, a, currentState)


            # Q Backup
            # for n in range(self.M.N):
            self.QBackup(n)

            s = currentState
            for aa in a:
                self.M.QSums[n][currentState][aa] += self.M.Q_bu[n][s][aa]
                self.M.QTouched[n][s][aa] += 1.0

            # Don't update terminal states
            # Check if these are ever even hit
            if self.M.isTerminal(self.M.getStateRep(currentState)) == True:
                done = True
                continue

            if self.M.getStateRep(currentState) == None:
                done = True
                continue

            # Update regrets
            self.regretUpdate(n, self.M.getStateRep(currentState), t)

            # Epsilon Greedy action selection
            if np.random.randint(0, 100) < int(self.epsilon):
                totalActions = self.M.getActions(currentState, 0)
                randomAction = np.random.randint(0, len(totalActions))
                randomAction = totalActions[randomAction]

            else:

                totalActions = []
                totalActionsProbs = []
                ta = self.M.getActions(currentState, 0)
                for action in ta:
                    totalActions.append(action)
                    totalActionsProbs.append(self.M.pi[n][self.M.getStateRep(currentState)][action])

                randomAction = np.random.choice(totalActions, p=totalActionsProbs)



            nextPossStates = self.M.getNextStatesAndProbs(currentState, randomAction, 0)
            if nextPossStates == []:
                done = True
                continue

            # If there is only one successor state, pick that
            if len(nextPossStates) == 1:
                # nextPossStates is list of lists
                # nextPossStates = [[next_state, prob, reward]]
                currentState = nextPossStates[0][0]
            else:
                nextStates = []
                nextStateProbs = []
                for ns, nsp, _ in nextPossStates:
                    nextStates.append(ns)
                    nextStateProbs.append(nsp)

                currentState = np.random.choice(nextStates, p=nextStateProbs)


################################################################
# LONR Asynchronous Value Iteration - random one state updates
################################################################
class LONR_AV(LONR):

    def __init__(self, M=None, parameters=None, regret_minimizers=None, dcfr=None):
        super().__init__(M=M, parameters=parameters, regret_minimizers=regret_minimizers, dcfr=dcfr)

    def lonr_train(self, iterations=-1, log=-1, randomize=False):

        self.totalIterations = iterations

        whoGoes = 0.0

        if log != -1: print("Starting training..")
        for t in range(1, iterations+1):

            if self.M.version == 0:
                self.M.version = 1

            if (t+1) % log == 0:
                print("Iteration: ", t+1, " alpha:", self.alpha, " epsilon: ", self.epsilon)

            self._lonr_train(t=t)#, totalIterations=iterations, randomized=randomized)

            self.alpha *= self.alphaDecay
            self.alpha = max(0.0, self.alpha)



            if randomize:

                # whoGoes += 1.0
                # if whoGoes > 3.0:
                #     whoGoes = 0.0
                if self.M.version == 1:
                    self.M.version = 2
                else:
                    self.M.version = 1


                # TigerOnLeftProb = self.M.TLProb
                # v = np.random.choice([1, 2], p=[TigerOnLeftProb, 1.0 - TigerOnLeftProb])
                # self.M.version = v



        if log != -1: print("Finished Training")

    ################################################################
    # LONR Asynchronous Value Iteration - random one state updates
    ################################################################
    def _lonr_train(self, t):


        # currentState = self.M.startState
        # if self.M.version == 1:
        #     currentState = 0
        # else:
        #     currentState = 1
        currentState = np.random.randint(0,48)

        done = False
        n = 0  # One player

        # Episode loop - until terminal state is reached
        # movesAllowed = 31
        visitedList = []
        # print("")
        while done == False:


            # Check this - tiger game stuff
            if self.M.getStateRep(currentState) == None:
                done = True
                continue

            if self.M.isTerminal(currentState) == False:
                visitedList.append(currentState)

            # Check this - catching terminals
            if self.M.isTerminal(self.M.getStateRep(currentState)):
                # print(" Terminal: ", currentState)
                a = self.M.getActions(currentState, 0)
                self.QUpdate(n, self.M.getStateRep(currentState), a, self.M.getStateRep(currentState))
                for aa in a:
                    self.M.QSums[n][currentState][aa] += self.M.Q_bu[n][currentState][aa]
                    self.M.QTouched[n][currentState][aa] += 1.0

                for n in range(self.M.N):
                    self.QBackup(n)
                done = True
                continue

            # if self.M.isTerminal(self.M.getStateRep(currentState)):
            #     a = self.M.getActions(self.M.getStateRep(currentState), 0)
            #     self.QUpdate(n, self.M.getStateRep(currentState), a, self.M.getStateRep(currentState))
            #     done = True
            #     continue


            # Get possible actions
            a = self.M.getActions(currentState, 0)

            # print(" a: ", a)

            # Update Q of actions
            self.QUpdate(n, currentState, a, currentState)

            # s = currentState
            for aa in a:
                self.M.QSums[n][currentState][aa] += self.M.Q_bu[n][currentState][aa]
                self.M.QTouched[n][currentState][aa] += 1.0
                # self.regretUpdate2(n, self.M.getStateRep(currentState), t, aa)

            # Q Backup
            # for n in range(self.M.N):
            #     self.QBackup(n)

            # Don't update terminal states
            # Check if these are ever even hit
            if self.M.isTerminal(self.M.getStateRep(currentState)) == True:
                done = True
                continue

            if self.M.getStateRep(currentState) == None:
                done = True
                continue

            # Update regrets
            self.regretUpdate(n, self.M.getStateRep(currentState), t)

            # print(" Q: ", self.M.Q_bu[n][currentState]["CONT"])
            # print(" Q: ", self.M.Q_bu[n][currentState]["EXIT"])
            # print(" Regret updated: ", self.M.regret_sums[n][currentState]["CONT"])
            # print(" Regret updated: ", self.M.regret_sums[n][currentState]["EXIT"])

            # Epsilon Greedy action selection
            if np.random.randint(0, 100) < int(self.epsilon):
                totalActions = self.M.getActions(currentState, 0)
                randomAction = np.random.randint(0, len(totalActions))
                randomAction = totalActions[randomAction]

            else:

                totalActions = []
                totalActionsProbs = []
                ta = self.M.getActions(currentState, 0)
                for action in ta:
                    totalActions.append(action)
                    totalActionsProbs.append(self.M.pi[n][self.M.getStateRep(currentState)][action])

                randomAction = np.random.choice(totalActions, p=totalActionsProbs)



            nextPossStates = self.M.getNextStatesAndProbs(currentState, randomAction, 0)
            if nextPossStates == []:
                done = True
                continue

            # If there is only one successor state, pick that
            if len(nextPossStates) == 1:
                # nextPossStates is list of lists
                # nextPossStates = [[next_state, prob, reward]]
                currentState = nextPossStates[0][0]
            else:
                nextStates = []
                nextStateProbs = []
                for ns, nsp, _ in nextPossStates:
                    nextStates.append(ns)
                    nextStateProbs.append(nsp)

                currentState = np.random.choice(nextStates, p=nextStateProbs)
            # movesAllowed = movesAllowed - 1
            # currentState = np.random.choice(stateList, p=stateProbs)
            # if movesAllowed <= 0:
            #     done = True
            done = True


        # for s in visitedList:
        #     for a in self.M.getActions(s, n):
        #         self.M.Q[n][s][a] = self.M.Q_bu[n][s][a]
        # if t % 101 == 0:
        #     for n in range(self.M.N):
        #         self.QBackup(n)
        #     for s in [0,1]:
        #         # print("REGUPD: ", s)
        #         self.regretUpdate(n, s, t)


################################################################
# LONR-VCF - LONR with counterfactual reach updates
###############################################################

class LONR_VCF(LONR):

    def __init__(self, M=None, parameters=None, regret_minimizers=None, dcfr=None, randomize=True):
        super().__init__(M=M, parameters=parameters, regret_minimizers=regret_minimizers, dcfr=dcfr)
        self.randomize = False

    ################################################################
    # LONR Value Iteration with cfr
    ################################################################
    def lonr_train(self, iterations=-1, log=-1, randomize=False):

        if log != -1: print("Starting training..")

        print("self.M.version: ", self.M.version)

        if self.M.version == 0:
            self.M.version = 1

        for t in range(1, iterations + 1):
            # print("Version: ", self.M.version)
            # self.M.version = 2
            if (t + 0) % log == 0:
                print("Iteration: ", t + 0, " alpha: ", self.alpha, " gamma: ", self.gamma)

            # Call one full update via LONR-V
            self._lonr_train(t=t)

            # No-op unless alphaDecay is not 1.0
            self.alpha *= self.alphaDecay
            self.alpha = max(0.0, self.alpha)

            # self.randomize = True

            # if randomize:
            #     TigerOnLeftProb = self.M.TLProb
            #     v = np.random.choice([1, 2], p=[TigerOnLeftProb, 1.0 - TigerOnLeftProb])
            #     self.M.version = v
            #     if self.M.version == 1:
            #         self.M.version = 2
            #     else:
            #         self.M.version = 1
                # self.M.version = v


        if log != -1: print("Finish Training")


    ######################################
    ## LONR VALUE ITERATION WITH COUNTERFACTUAL REACH UPDATES
    ######################################
    def _lonr_train(self, t):
        """ One full update via LONR-V
        """

        # Q Update

        for n in range(self.M.N):

            # Loop through all states
            for s in self.M.getStates():

                if self.M.isTerminal(s):
                    a = self.M.getActions(s, 0)
                    self.QUpdate(n, s, a, randomS=s)

                    continue

                # Loop through actions of current player n
                # for a in self.M.getActions(s, n):
                a = self.M.getActions(self.M.getStateRep(s), n)
                #if self.randomize == False:


                self.QUpdate(n, s, a, randomS=None)

                for aa in a:
                    self.M.QSums[n][s][aa] += self.M.Q_bu[n][s][aa]
                    self.M.QTouched[n][s][aa] += 1.0


        # Q Backup
        for n in range(self.M.N):
            self.QBackup(n)


        for n in range(self.M.N):
            for s in self.M.getStates():
                self.regretUpdate(n, s, t)


####################################################################################
################################################################
# LONR TD
################################################################
class LONR_TD(LONR):

    def __init__(self, M=None, parameters=None, regret_minimizers=None, dcfr=None):
        super().__init__(M=M, parameters=parameters, regret_minimizers=regret_minimizers, dcfr=dcfr)

    def lonr_train(self, iterations=-1, log=-1, randomize=False):

        self.totalIterations = iterations

        whoGoes = 0.0

        if log != -1: print("Starting training..")
        for t in range(1, iterations+1):

            if self.M.version == 0:
                self.M.version = 1

            # print(self.M.version)

            # r = float(self.epsilon)
            # f = float(r) * 0.99999
            # rr = 1.0 - (2.0*t / self.totalIterations)
            # # self.epsilon = int(self.epsilon * 0.99999999)
            # self.epsilon = 100.0 * rr
            # self.epsilon = np.random.randint(0,5)

            if (t+1) % log == 0:
                print("Iteration: ", t+1, " alpha:", self.alpha, " epsilon: ", self.epsilon, " pi: ", self.M.pi[0][self.M.startState])

            self._lonr_train(t=t)

            self.alpha *= self.alphaDecay
            self.alpha = max(0.0, self.alpha)

            if randomize:

                if self.M.version == 1:
                    self.M.version = 2
                else:
                    self.M.version = 1



        if log != -1: print("Finished Training")

    ################################################################
    # LONR TD
    ################################################################
    def _lonr_train(self, t):

        n = 0  # One player
        for s in self.M.getStates():
            for a in self.M.getActions(s, n):
                self.M.elig[n][s][a] = 0.0


        currentState = self.M.startState
        # currentState = np.random.randint(0,48)
        done = False

        # if t % 100 == 0:
        #     for s in self.M.getStates():
        #         for a in self.M.getActions(s, n):
        #             if self.M.regret_sums[0][s][a] < 0:
        #                 self.M.regret_sums[0][s][a] = 0.0
        #             else:
        #                 self.M.regret_sums[0][s][a] = 0.2


        # Episode loop - until terminal state is reached
        visitedList = []
        randomAction = None
        # print("-----")
        randomAct = False
        while done == False:

            # if self.M.isTerminal(currentState) == False:
            #     if [currentState, randomAction] in visitedList:
            #         if randomAction != None:
            #             self.regretUpdate(n, currentState, t)

            # if self.M.isTerminal(currentState) == False:
            #     if [currentState, randomAction] not in visitedList:
            #         visitedList.append([currentState, randomAction])
            #     # else:
            #     #     self.regretUpdate(n, currentState, t)

            # Check this - tiger game stuff
            if self.M.getStateRep(currentState) == None:
                done = True
                continue

            # Epsilon Greedy action selection
            if np.random.randint(0, 100) < int(self.epsilon):
                totalActions = self.M.getActions(currentState, 0)
                randomAction = np.random.randint(0, len(totalActions))
                randomAction = totalActions[randomAction]
                randomAct = True

            else:

                totalActions = []
                totalActionsProbs = []
                ta = self.M.getActions(currentState, 0)
                for action in ta:
                    totalActions.append(action)
                    totalActionsProbs.append(self.M.pi[n][self.M.getStateRep(currentState)][action])

                # print("Current state: ", currentState)
                # print("totProb: ", totalActionsProbs)
                randomAction = np.random.choice(totalActions, p=totalActionsProbs)
                randomAct = False

            # if self.M.isTerminal(currentState) == False:
            if [currentState,randomAction] not in visitedList:
                visitedList.append([currentState,randomAction])
            # else:
            #     self.regretUpdate(n, currentState, t)#, randomAction)


            nextPossStates = self.M.getNextStatesAndProbs(currentState, randomAction, 0)
            if nextPossStates == []:
                done = True
                continue

            # If there is only one successor state, pick that
            if len(nextPossStates) == 1:
                # nextPossStates is list of lists
                # nextPossStates = [[next_state, prob, reward]]
                nextState = nextPossStates[0][0]
            else:
                nextStates = []
                nextStateProbs = []
                for ns, nsp, _ in nextPossStates:
                    nextStates.append(ns)
                    nextStateProbs.append(nsp)

                nextState = np.random.choice(nextStates, p=nextStateProbs)


            if nextState == None:
                reward = self.M.getReward(currentState, randomAction, n, n)
                Value = 0.0 #reward

            else:
                reward = self.M.getReward(currentState, randomAction, n, n)
                Value = 0.0
                for a in self.M.getActions(nextState, n):
                    #print(n, nextState, a)
                    Value += self.M.Q[n][nextState][a] * self.M.pi[n][nextState][a]

            if randomAct:
                IS = 1.0#0.025
                tot = (reward + self.gamma * Value) #/ 0.025
            else:
                IS = 1.0
                tot = reward + self.gamma * Value

            # if self.M.isTerminal(currentState) == False:
            #     if [currentState,randomAction] not in visitedList:
            #         visitedList.append([currentState,randomAction])
            #     # else:
            #     #     self.regretUpdate(n, currentState, t)



            # self.M.elig[n][currentState][randomAction] = 1.0
            if randomAct == False:
                self.M.elig[n][currentState][randomAction] = 1.0
            else:
                for s,a in visitedList:
                    self.M.elig[n][s][a] = 0.0
                self.M.elig[n][currentState][randomAction] = 0.0
                visitedList = []
                currentState = nextState
                continue

            lamda = 0.0#1.0#0.90
            alpha = 0.01

            # target = 0.0
            # for a in self.M.getActions(currentState, n):
            #     target += self.M.Q[n][currentState][a]*self.M.pi[n][currentState][a]
            #
            #
            # var = target - self.M.Q[n][currentState][randomAction]
            # upD = 0.0
            # upDD = 0.0
            for s,a in visitedList:
                    #self.M.Q[n][s][a] = self.M.Q[n][s][a] + alpha*tot*self.M.elig[n][s][a]#-self.M.Q[n][s][a])
                    # self.M.Q[n][s][a] = (1.0 - alpha) * self.M.Q[n][s][a] + (alpha * (reward + self.gamma * Value - self.M.Q[n][s][a]) * self.M.elig[n][s][a])
                    # upD = self.M.Q[n][s][a] + (alpha * (( ((reward + self.gamma * Value) / IS) - self.M.Q[n][s][a]) )) * self.M.elig[n][s][a]
                    # upDD = self.M.Q[n][currentState][randomAction] + (alpha * ((((reward + self.gamma * Value) / IS) - self.M.Q[n][currentState][randomAction]))) * self.M.elig[n][s][a]
                    self.M.Q[n][s][a] = self.M.Q[n][s][a] + (alpha * (( ((reward + self.gamma * Value) / IS) - self.M.Q[n][s][a]) )) * self.M.elig[n][s][a]
                    self.M.elig[n][s][a] = self.M.elig[n][s][a] * self.gamma * lamda * self.M.pi[n][currentState][randomAction]
                    self.M.QSums[n][s][a] += self.M.Q[n][s][a]
                    self.M.QTouched[n][s][a] += 1.0
                    # self.regretUpdate2(n,s,t,a)
                    # self.regretUpdate(n, s, t)
            # for s in self.M.getStates():
            #     for a in self.M.getActions(s, n):
            # for s, a in visitedList:
            #     self.M.elig[n][s][a] = self.M.elig[n][s][a] * self.gamma * lamda
            # print("")
            #visitedList = []
                # self.M.QSums[n][currentState][randomAction] += self.M.Q[n][currentState][randomAction]#self.M.Q_bu[n][currentState][aa]
                # self.M.QTouched[n][currentState][randomAction] += 1.0

            # if randomAct == False:
            # if upDD > self.M.Q[n][currentState][randomAction]:
            self.regretUpdate(n, currentState, t)
            # self.regretUpdate2(n, currentState, t, randomAction)

            currentState = nextState

            # Check this - catching terminals
            if self.M.isTerminal(self.M.getStateRep(currentState)):
                #currentState = nextState
                self.M.elig[n][currentState][randomAction] = 1.0
                # print(" Terminal: ", currentState)
                a = self.M.getActions(currentState, 0)
                Value = 0.0
                for aa in a:
                    Value = self.M.getReward(currentState, aa, n, n)
                    # s, a_current, n, a_notN):
                    self.M.Q_bu[n][currentState][aa] = self.M.getReward(currentState, aa, n, n)
                    self.M.Q[n][currentState][aa] = self.M.getReward(currentState, aa, n, n)
                    self.M.QSums[n][currentState][aa] += self.M.Q[n][currentState][aa]
                    self.M.QTouched[n][currentState][aa] += 1.0

                for s, a in visitedList:
                    # print(currentState, aa)
                    self.M.elig[n][s][a] = self.M.elig[n][s][a] * self.gamma * lamda #* self.M.pi[n][currentState][randomAction]
                    self.M.Q[n][s][a] = self.M.Q[n][s][a] + (alpha * (reward + self.gamma * Value - self.M.Q[n][s][a])) * self.M.elig[n][s][a]

                    self.M.QSums[n][s][a] += self.M.Q[n][s][a]
                    self.M.QTouched[n][s][a] += 1.0



                #done = True
                #continue

            # Don't update terminal states
            # Check if these are ever even hit
            if self.M.isTerminal(self.M.getStateRep(currentState)) == True:
                done = True
                continue

            if self.M.getStateRep(currentState) == None:
                done = True
                continue
            #done = True
        # for s, a in visitedList:
        # #     # self.M.Q[n][s][a] = self.M.Q[n][s][a] + alpha*tot*self.M.elig[n][s][a]#-self.M.Q[n][s][a])
        # #     # self.M.Q[n][s][a] = (1.0 - alpha) * self.M.Q[n][s][a] + (alpha * (reward + self.gamma * Value - self.M.Q[n][s][a]) * self.M.elig[n][s][a])
        # #     self.M.Q[n][s][a] = self.M.Q[n][s][a] + (
        # #                 alpha * ((((reward + self.gamma * Value) / IS) - self.M.Q[n][s][a]))) * self.M.elig[n][s][a]
        # #     self.M.elig[n][s][a] = self.M.elig[n][s][a] * self.gamma * lamda * self.M.pi[n][currentState][randomAction]
        # #     self.M.QSums[n][s][a] += self.M.Q[n][s][a]
        # #     self.M.QTouched[n][s][a] += 1.0
        # #     self.regretUpdate2(n, s, t, a)
        #     self.regretUpdate(n, s, t )

        # for n in range(self.M.N):
        #     self.QBackup(n)
        #
        # for s,a in visitedList:
        #     self.regretUpdate(n, s, t)



#####################################################################################



######################################################################################


####################################################################################




#####################################################################################







class LONR_B(LONR):

    def __init__(self, M=None, parameters=None, regret_minimizers=None, dcfr=None):
        super().__init__(M=M, parameters=parameters, regret_minimizers=regret_minimizers, dcfr={})

    ################################################################
    # LONR Bandit
    ################################################################
    def lonr_train(self, iterations=-1, log=-1, randomize=False):

        if log != -1: print("Starting training..")

        self.totalIterations = iterations

        self.exp3gamma = 1.0

        for t in range(1, iterations + 1):

            if (t + 0) % log == 0:
                print("Iteration: ", t + 0, " alpha: ", self.alpha, " gamma: ", self.gamma, " exp3gamma: ", self.exp3gamma)

            # Call one full update via LONR-V
            self._lonr_train(t=t)

            self.exp3gamma = 1.0 - (t / self.totalIterations)
            self.exp3gamma = max(self.exp3gamma, 0.14)
            if t < int( (2.0 *self.totalIterations) / 3.0):
                # self.exp3gamma = math.sqrt(2.0 / (2.0*t))
                self.exp3gamma = 0.1 #1.0 - ((1.5 * t) / (1.0*self.totalIterations))
            else:
                self.exp3gamma = 0.1
            #self.exp3gamma = math.sqrt(4.0 / (4.0*t))
            # self.alpha = 0.99#1.0 - (t / self.totalIterations)

        if log != -1: print("Finish Training")

    ######################################
    ## LONR Bandit
    ######################################
    def _lonr_train(self, t):

        # Set start state or randomize start state
        currentState = self.M.startState
        #currentState = np.random.randint(0,48)

        n = 0
        done = False

        # for s in self.M.getStates():
        #     for a in self.M.getActions(s, n):
        #         self.M.elig[n][s][a] = 0.0

        # Episode loop - until terminal state is reached

        if t % 10000 == 0:
            print("T:", t)

        nextAction = None
        visitedList = []

        lamda = 0.00  # 1.0#0.90
        alpha = 0.101

        while done == False:

            # Periodic output
            if t % 2500 == -10:
                verbose = True
            else:
                verbose = False


            # Force terminal to be correct value based on reward
            if self.M.isTerminal(currentState):
                for a in self.M.getActions(currentState, n):
                    self.M.Q_bu[n][currentState][a] = self.M.getReward(currentState, a, 0, 0)
                    self.M.Q[n][currentState][a] = self.M.Q_bu[n][currentState][a]
                    self.M.QSums[n][currentState][a] += self.M.Q[n][currentState][a]
                    self.M.QTouched[n][currentState][a] += 1.0
                    self.M.pi_sums[n][self.M.getStateRep(currentState)][a] += self.M.pi[n][self.M.getStateRep(currentState)][a]

                # for s, a in visitedList:
                #     self.M.Q_bu[n][s][a] = (1.0 - self.alpha) * self.M.Q[n][s][a] + self.alpha * self.M.getReward(s, a, 0, 0)
                #     self.M.elig[n][s][a] = self.M.elig[n][s][a] * self.gamma * lamda# * self.M.pi[n][currentState][nextAction]
                #     self.M.QSums[n][s][a] += self.M.Q_bu[n][s][a]
                #     self.M.QTouched[n][s][a] += 1.0

                done = True
                continue

            # everything done in exp3, just get the action it used
            # Note: return action and state for non-determinism (might be different here)
            nextAction = self.exp3(currentState,n, t)

            # ###22
            # self.M.elig[n][currentState][nextAction] += 1.0

            s_prime = self.M.getMove(currentState, nextAction)

            rew = self.M.getReward(currentState, nextAction, n, n)

            if [currentState, nextAction] not in visitedList:
                visitedList.append([currentState,nextAction])

            Value = 0.0
            for a_prime in self.M.getActions(s_prime, n):
                Value += self.M.pi[n][self.M.getStateRep(s_prime)][a_prime] * self.M.Q[n][s_prime][a_prime]

            x = rew + self.gamma * Value
            # x = self.M.Q_bu[n][currentState][nextAction]

            self.M.Q_bu[n][currentState][nextAction] = self.alpha*x + (1.0 - self.alpha)*self.M.Q[n][currentState][nextAction] #(rew + self.gamma * Value)(1.0 / self.M.pi[0][currentState][nextAction]) *

            # if randomAct == False:
            self.M.elig[n][currentState][nextAction] = 1.0
            # else:
            #     for s,a in visitedList:
            #         self.M.elig[n][s][a] = 0.0
            #     self.M.elig[n][currentState][randomAction] = 0.0
            #     visitedList = []
            #     currentState = nextState
            #     continue



            # x = self.M.Q_bu[n][currentState][nextAction]
            # for s,a in visitedList:
            #     if s != currentState:
            #         #self.M.Q[n][s][a] = self.M.Q[n][s][a] + alpha*tot*self.M.elig[n][s][a]#-self.M.Q[n][s][a])
            #         # self.M.Q[n][s][a] = (1.0 - alpha) * self.M.Q[n][s][a] + (alpha * (reward + self.gamma * Value - self.M.Q[n][s][a]) * self.M.elig[n][s][a])
            #         #self.M.Q[n][s][a] = self.M.Q[n][s][a] + (alpha * (rew + self.gamma * Value - self.M.Q[n][s][a])) * self.M.elig[n][s][a]
            #         self.M.elig[n][s][a] = self.M.elig[n][s][a] * self.gamma * lamda * self.M.pi[n][currentState][nextAction]
            #         self.M.Q_bu[n][s][a] = (1.0 - self.alpha) * self.M.Q[n][currentState][nextAction] + self.alpha * x * self.M.elig[n][s][a]
            #
            #         self.M.QSums[n][s][a] += self.M.Q_bu[n][s][a]
            #         self.M.QTouched[n][s][a] += 1.0


            for aa in self.M.Q[n][currentState].keys():

                if aa == nextAction:
                    self.M.Q[n][currentState][aa] = self.M.Q_bu[n][currentState][aa]
                    self.M.QSums[n][currentState][aa] += self.M.Q[n][currentState][aa]
                    self.M.QTouched[n][currentState][aa] += 1.0
                else:
                    ddd = 3
                    # self.M.QSums[0][currentState][aa] += self.M.Q[0][currentState][aa]
                    # self.M.QTouched[0][currentState][aa] += 1.0
                    self.M.Q[n][currentState][aa] = 0.0
                    self.M.Q_bu[n][currentState][aa] = 0.0




            if verbose: print("")
            if verbose: print("Current State: ", currentState, "  currentAction: ", nextAction, " t=", t)

            nextState = self.M.getMove(currentState, nextAction)
            currentState = nextState
            #done = True



    def distr(self, weights, gamma=0.0):

        theSum = 0.0
        theWeights = []
        for w in weights:
            theSum += math.exp(w)# * gamma / 4.0)
            theWeights.append(math.exp(w))# * gamma / 4.0))
        # theSum = float(sum(weights))
        rl = []
        if theSum <= 0:
            theSum = 1.0
        for w in theWeights:
            value = (1.0 - gamma) * (w / theSum) + (gamma / len(weights))
            rl.append(value)
        return rl
        # return list((1.0 - gamma) * (w / theSum) + (gamma / len(weights)) for w in weights)


    def exp3(self, currentState, n, t):

        # Keep around for decaying exp3 gamma
        #gamma = max(1.0 - (float(float(4.0/2.0)*t) / float(self.totalIterations)), 0.1)

        # Show occasional printout
        if t % 12500 == -10:
            verbose = True
        else:
            verbose = False

        rewardMin = -112.0
        rewardMax = 200.0

        numActions = 4.0
        #n = 0
        gamma = self.exp3gamma#0.1
        maxGap = 8.0



        # Get current weights for 0, 1, 2 ,3
        # Only for weights above
        weights = []
        maxWeight = -10000000
        if verbose: print("Weights: ")
        #print("N: ", n, " CS: ", currentState, " M: ", self.M.weights)
        for w in sorted(self.M.weights[n][self.M.getStateRep(currentState)].keys()):
            if verbose: print("   ", w, ": ", self.M.weights[n][self.M.getStateRep(currentState)][w])
            if self.M.weights[n][self.M.getStateRep(currentState)][w] > maxWeight:
                maxWeight = self.M.weights[n][self.M.getStateRep(currentState)][w]

        if verbose: print("Max Weight: ", maxWeight)

        maxWeightGap = maxWeight - maxGap
        if verbose: print("Only considering those above maxWeightGap: ", maxWeightGap, ", else 0")
        for w in sorted(self.M.weights[n][currentState].keys()):
            if self.M.weights[n][self.M.getStateRep(currentState)][w] > maxWeightGap:
                weights.append(self.M.weights[n][self.M.getStateRep(currentState)][w] - maxWeightGap)
            else:
                weights.append(0.0)



        if verbose: print("Final weights list: ", weights) # init: 1.0, 1.0, 1.0, 1.0

        # Get probDist from weights
        pd = self.distr(weights, gamma)

        if verbose: print("ProbDist: ", pd) # init: 0.25, 0.25, 0.25, 0.25

        # Dist without randomness (for final pi sum calc)
        piSUMDist = self.distr(weights, gamma=1.0)
        piSUMDist2 = self.distr(weights, gamma=0.0)

        # set pi as probDistbution
        probActions = []
        aa = 0
        for a in sorted(self.M.pi[n][currentState].keys()):
            if verbose: print("Setting pi of s: ", currentState, " a: ", a, " : ", pd[aa])
            self.M.pi_sums[n][self.M.getStateRep(currentState)][a] += piSUMDist2[aa]
            if self.M.weights[n][self.M.getStateRep(currentState)][a] > maxWeightGap:
                self.M.pi[n][self.M.getStateRep(currentState)][a] = pd[aa]
            else:
                self.M.pi[n][self.M.getStateRep(currentState)][a] = pd[aa]
            aa += 1
                #self.M.pi_sums[n][currentState][a] += piSUMDist2[int(a)]

        # Select a random action from pd
        acts = range(len(self.M.getActions(currentState, n)))
        randomAction = np.random.choice(acts, p=pd)
        if verbose: print("Selected random action: ", randomAction)

        randomActionIndex = randomAction
        randomAction = self.M.getActions(currentState, n)[randomAction]
        #Observe reward for action randomAction
        rew = self.M.getReward(currentState, randomAction, n, n)
        print("CS: ", currentState)
        s_prime = self.M.getMove(currentState, randomAction)
        Value = 0.0
        # print("THIS: ", self.M.getActions(s_prime, n), " sprime: ", s_prime)
        #maxQ = -10000.0
        for a_prime in sorted(self.M.getActions(s_prime, n)):
            Value += self.M.pi[n][self.M.getStateRep(s_prime)][a_prime] * self.M.Q[n][s_prime][a_prime]
        # print(rew, Value)
        x = rew + self.gamma*Value

        # Scale the reward
        scaledReward = (x - rewardMin) / (rewardMax - rewardMin)
        if scaledReward > 1.0:# and t % 2000 == 0:
            print("SCALED REWARD: ",scaledReward, " currentState: ", currentState)
        if scaledReward < 0:# and t % 2000 == 0:
            print("SCALED REWARD: ", scaledReward, " currentState: ", currentState)
        # x = max(x, 0.0)
        # x = min(x, 1.0)

        # Get expected reward
        estimatedReward = scaledReward



        runningEstimatedReward = estimatedReward / pd[randomActionIndex]

        # Find min weight
        # currentRunningWeights = []
        for r in sorted(self.M.Q_bu[n][currentState].keys()):
            if r == randomAction:
                # self.M.runningRewards[n][currentState][r] += runningEstimatedReward# / self.M.pi[n][currentState][r]

                #avgReward = self.M.runningRewards[0][currentState][r] / t

                self.M.Q_bu[n][currentState][r] = runningEstimatedReward# / self.M.pi[n][currentState][r]
            else:
                self.M.Q_bu[n][currentState][r] = 0.0



        # Set Weight for t+1
        # w_t+1() = w_t()e^(x_hat * gamma / K) = 1.0 ^ e^A * e^b
        # Keep sum of weights and put in exponent when calculating prob dist above
        if verbose: print(runningEstimatedReward)
        for j in sorted(self.M.weights[n][self.M.getStateRep(currentState)].keys()):
            if j == randomAction:
                self.M.weights[n][self.M.getStateRep(currentState)][j] += self.M.Q_bu[n][currentState][j] * gamma / float(numActions) #*= math.exp(self.M.runningRewards[0][currentState][j] * gamma / float(numActions))
                #self.M.weights[0][currentState][j] *= math.exp(self.M.runningRewards[0][currentState][j] * gamma / float(numActions))


        return randomAction







class LONR_RM(LONR):

    def __init__(self, M=None, parameters=None, regret_minimizers=None):
        super().__init__(M=M, parameters=parameters, regret_minimizers=regret_minimizers)

    ################################################################
    # LONR RM
    ################################################################
    def lonr_train(self, iterations=-1, log=-1, randomize=False):

        if log != -1: print("Starting training..")

        self.totalIterations = iterations

        self.exp3gamma = 1.0

        for t in range(1, iterations + 1):

            if (t + 0) % log == 0:
                print("Iteration: ", t + 0, " alpha: ", self.alpha, " gamma: ", self.gamma, " exp3gamma: ",
                      self.exp3gamma)

            # Call one full update via LONR-V
            self._lonr_train(t=t)

            # self.alpha = 0.99  # 1.0 - (t / self.totalIterations)

        if log != -1: print("Finish Training")

    ######################################
    ## LONR Bandit RM
    ######################################
    def _lonr_train(self, t):

        # Set start state or randomize start state
        currentState = self.M.startState
        # currentState = np.random.randint(0, 37)

        # Settings
        done = False
        n = 0  # One player
        nextAction = None

        # Episode loop - until terminal state is reached
        while done == False:

            # Periodic output
            if t % 32500 == 0:
                verbose = True
            else:
                verbose = False

            # Force terminal to be correct value based on reward
            if self.M.isTerminal(currentState):
                for a in self.M.getActions(currentState, n):
                    self.M.Q_bu[n][currentState][a] = self.M.getReward(currentState, a, 0, 0)
                    self.M.Q[n][currentState][a] = self.M.Q_bu[n][currentState][a]
                    self.M.QSums[0][currentState][a] += self.M.Q[0][currentState][a]
                    self.M.QTouched[0][currentState][a] += 1.0
                    self.M.pi_sums[n][currentState][a] += self.M.pi[n][currentState][a]
                done = True
                continue

            # everything done in exp3, just get the action it used
            # Note: return action and state for non-determinism (might be different here)
            nextAction = self.RM_E(currentState, t)

            s_prime = self.M.getMove(currentState, nextAction)

            rew = self.M.getReward(currentState, nextAction, 0, 0)

            Value = 0.0
            for a_prime in self.M.getActions(s_prime, n):
                Value += self.M.pi[n][s_prime][a_prime] * self.M.Q[n][s_prime][a_prime]

            x = rew + self.gamma * Value

            self.M.Q_bu[0][currentState][nextAction] = self.alpha * x + (1.0 - self.alpha) * self.M.Q[0][currentState][nextAction]  # (rew + self.gamma * Value)(1.0 / self.M.pi[0][currentState][nextAction]) *

            for aa in self.M.Q[0][currentState].keys():

                if aa == nextAction:
                    self.M.Q[0][currentState][aa] = self.M.Q_bu[0][currentState][aa]
                    self.M.QSums[0][currentState][aa] += self.M.Q[0][currentState][aa]
                    self.M.QTouched[0][currentState][aa] += 1.0
                else:
                    ddd = 3
                    # self.M.QSums[0][currentState][aa] += self.M.Q[0][currentState][aa]
                    # self.M.QTouched[0][currentState][aa] += 1.0
                    # self.M.Q[0][currentState][aa] = 0.0
                    # self.M.Q_bu[0][currentState][aa] = 0.0

            if verbose: print("")
            if verbose: print("Current State: ", currentState, "  currentAction: ", nextAction, " t=", t)

            self.regretUpdate2(n, currentState, t, nextAction)
            #regretUpdate2(self, n, currentState, t, action)
            nextState = self.M.getMove(currentState, nextAction)
            currentState = nextState


    def distr(self, weights, gamma=0.0):

        rl = []
        theSum = 0.0
        theWeights = []
        for w in weights:
            theSum += w #max(w,0)  # math.exp(w)# * gamma / 4.0)

        if theSum == 0:
            for w in range(len(weights)):
                value = 1.0 / 4.0
                rl.append(value)
                if np.random.randint(0, 20000) < 2:
                    print("Returning uniform")
            return rl

        tempPi = []
        leftOver = 0.0
        for w in weights:
            tempPi.append((1.0-gamma)*(max(w,0) / theSum))
            leftOver += gamma * (max(w,0) / theSum)


        for w in range(len(weights)):
            value = (tempPi[w] / theSum) + (gamma / 4.0)
            rl.append(value)

        rl_norm = []
        tot = 0.0
        for w in range(len(rl)):
            tot += rl[w]
        if tot == 0:
            tot = 1.0

        for w in range(len(rl)):
            rl_norm.append(rl[w] / tot)
        if rl_norm == []:
            print("Returning empty: ", rl_norm)
        return rl_norm
        # theSum = 0.0
        # theWeights = []
        # for w in weights:
        #     theSum += max(w, 0.0)  #math.exp(w)# * gamma / 4.0)
        #     theWeights.append(w)#math.exp(w))# * gamma / 4.0))
        # # theSum = float(sum(weights))
        # rl = []
        # if theSum <= 0:
        #     theSum = 1.0
        # for w in theWeights:
        #     value = (1.0 - gamma) * (w / theSum) + (gamma / len(weights))
        #     rl.append(value)

        # return list((1.0 - gamma) * (w / theSum) + (gamma / len(weights)) for w in weights)


    def RM_E(self, currentState, t):

        # Show occasional printout
        if t % 12500 == 0:
            verbose = True
        else:
            verbose = False

        rewardMin = -101.0# -1.0
        rewardMax = 200.0

        numActions = 4.0
        n = 0
        if t < (self.totalIterations / 2):
            gamma = 1.0 * (1.0 - ((2.0*t) / self.totalIterations))
        else:
            gamma = 0.0
        # gamma = 0.2
        maxGap = 100.0



        # Get current weights for 0, 1, 2 ,3
        # Only for weights above
        weights = []
        # maxWeight = -10000000
        # if verbose: print("Weights: ")
        # for w in sorted(self.M.regret_sums[n][currentState].keys()):
        #     if verbose: print("   ", w, ": ", self.M.regret_sums[n][currentState][w])
        #     if self.M.regret_sums[n][currentState][w] > maxWeight:
        #         maxWeight = self.M.regret_sums[n][currentState][w]

        # if verbose: print("Max Weight: ", maxWeight)

        # maxWeightGap = maxWeight - maxGap
        # if verbose: print("Only considering those above maxWeightGap: ", maxWeightGap, ", else 0")
        for w in sorted(self.M.regret_sums[n][currentState].keys()):
            # if self.M.weights[n][currentState][w] > maxWeightGap:
            # print("ADDING TO WEIGHTS: CURRENT STATE: ", currentState, " KEY: ", w)
            weights.append(self.M.regret_sums[n][currentState][w])# - maxWeightGap)
            # else:
            #     weights.append(0.0)



        if verbose: print("Final weights list: ", weights) # init: 1.0, 1.0, 1.0, 1.0

        # Get probDist from weights
        pd = self.distr(weights, gamma)

        if pd == []:
            print("PD is empty: weights: ", weights)
        if verbose: print("ProbDist: ", pd) # init: 0.25, 0.25, 0.25, 0.25

        # Dist without randomness (for final pi sum calc)
        # piSUMDist = self.distr(weights, gamma=1.0)
        piSUMDist2 = self.distr(weights, gamma=0.0)

        # set pi as probDistbution
        probActions = []
        for a in sorted(self.M.pi[n][currentState].keys()):
            # print("KEYS: ", self.M.pi[n][currentState].keys())
            # print("PD: ", pd)
            # print("Setting pi of s: ", currentState)
            # print(" a: ", a, " : ", pd[a])

            self.M.pi_sums[n][currentState][a] += piSUMDist2[a]
            #if self.M.weights[n][currentState][a] > maxWeightGap:
            self.M.pi[n][currentState][a] = pd[a]

            # else:
            #     self.M.pi[n][currentState][a] = pd[a]
            #     #self.M.pi_sums[n][currentState][a] += piSUMDist2[int(a)]

        # Select a random action from pd
        #print("PD: ", pd)
        randomAction = np.random.choice([0,1,2,3], p=pd)
        if verbose: print("Selected random action: ", randomAction)

        #Observe reward for action randomAction
        rew = self.M.getReward(currentState, randomAction, 0, 0)
        s_prime = self.M.getMove(currentState, randomAction)
        Value = 0.0
        for a_prime in sorted(self.M.getActions(s_prime, n)):
            Value += self.M.pi[n][s_prime][a_prime] * self.M.Q[n][s_prime][a_prime]

        # This is the entire "reward", taken from paper
        x = rew + self.gamma*Value

        # Scale the reward
        # scaledReward = (x - rewardMin) / (rewardMax - rewardMin)
        # if scaledReward > 1.0:# and t % 2000 == 0:
        #     print("SCALED REWARD: ",scaledReward, " currentState: ", currentState)
        # if scaledReward < 0:# and t % 2000 == 0:
        #     print("SCALED REWARD: ", scaledReward, " currentState: ", currentState)
        # x = max(x, 0.0)
        # x = min(x, 1.0)

        # Get expected reward
        # estimatedReward = scaledReward



        # runningEstimatedReward = estimatedReward / pd[randomAction]

        # Find min weight
        # currentRunningWeights = []
        # for r in sorted(self.M.runningRewards[0][currentState].keys()):
        #     if r == randomAction:
        #         self.M.runningRewards[0][currentState][r] += runningEstimatedReward# / self.M.pi[n][currentState][r]

                #avgReward = self.M.runningRewards[0][currentState][r] / t

                #self.M.Q_bu[0][currentState][r] = x# runningEstimatedReward# / self.M.pi[n][currentState][r]
            # else:
            #     self.M.Q_bu[0][currentState][r] = 0.0



        # Set Weight for t+1
        # w_t+1() = w_t()e^(x_hat * gamma / K) = 1.0 ^ e^A * e^b
        # Keep sum of weights and put in exponent when calculating prob dist above
        # if verbose: print(runningEstimatedReward)
        # for j in sorted(self.M.weights[0][currentState].keys()):
        #     if j == randomAction:
        #         self.M.weights[0][currentState][j] += self.M.Q_bu[0][currentState][j] * gamma / float(numActions) #*= math.exp(self.M.runningRewards[0][currentState][j] * gamma / float(numActions))
        #         #self.M.weights[0][currentState][j] *= math.exp(self.M.runningRewards[0][currentState][j] * gamma / float(numActions))
        #
        # if verbose: print(self.M.weights[0][4])

        return randomAction





################################################################
# LONR TDE
################################################################
# class LONR_TDE(LONR):
#
#     def __init__(self, M=None, parameters=None, regret_minimizers=None, dcfr=None):
#         super().__init__(M=M, parameters=parameters, regret_minimizers=regret_minimizers, dcfr=dcfr)
#
#     def lonr_train(self, iterations=-1, log=-1, randomize=False):
#
#         self.totalIterations = iterations
#
#         whoGoes = 0.0
#
#         if log != -1: print("Starting training..")
#         for t in range(1, iterations+1):
#
#             if self.M.version == 0:
#                 self.M.version = 1
#
#             # r = float(self.epsilon)
#             # f = float(r) * 0.99999
#             # rr = 1.0 - (2.0*t / self.totalIterations)
#             # # self.epsilon = int(self.epsilon * 0.99999999)
#             # self.epsilon = 100.0 * rr
#             # self.epsilon = np.random.randint(0,5)
#
#             if (t+1) % log == 0:
#                 print("Iteration: ", t+1, " alpha:", self.alpha, " epsilon: ", self.epsilon, " pi: ", self.M.pi[0][self.M.startState])
#
#             self._lonr_train(t=t)
#
#             self.alpha *= self.alphaDecay
#             self.alpha = max(0.0, self.alpha)
#
#             if randomize:
#
#                 if self.M.version == 1:
#                     self.M.version = 2
#                 else:
#                     self.M.version = 1
#
#
#
#         if log != -1: print("Finished Training")
#
#     ################################################################
#     # LONR TD
#     ################################################################
#     def _lonr_train(self, t):
#
#         n = 0  # One player
#         # for s in self.M.getStates():
#         #     for a in self.M.getActions(s, n):
#         #         self.M.elig[n][s][a] = 0.0
#
#
#         currentState = self.M.startState
#         # currentState = np.random.randint(0,2)
#         done = False
#
#         # if t % 11 == 0:
#         #     for s in self.M.getStates():
#         #         for a in self.M.getActions(s, n):
#         #             if self.M.regret_sums[0][s][a] > 0:
#         #                 self.M.regret_sums[0][s][a] = 0.0
#         #             else:
#         #                 self.M.regret_sums[0][s][a] *= 0.5
#
#
#         # Episode loop - until terminal state is reached
#         visitedList = []
#         randomAction = None
#         # Epsilon Greedy action selection
#         if np.random.randint(0, 100) < int(self.epsilon):
#             totalActions = self.M.getActions(currentState, 0)
#             randomAction = np.random.randint(0, len(totalActions))
#             randomAction = totalActions[randomAction]
#             randomAct = True
#
#         else:
#
#             totalActions = []
#             totalActionsProbs = []
#             ta = self.M.getActions(currentState, 0)
#             for action in ta:
#                 totalActions.append(action)
#                 totalActionsProbs.append(self.M.pi[n][self.M.getStateRep(currentState)][action])
#
#             # print("Current state: ", currentState)
#             # print("totProb: ", totalActionsProbs)
#             randomAction = np.random.choice(totalActions, p=totalActionsProbs)
#             randomAct = False
#         # print("-----")
#         randomAct = False
#         while done == False:
#
#
#
#             # Check this - tiger game stuff
#             if self.M.getStateRep(currentState) == None:
#                 done = True
#                 continue
#
#
#
#             # if self.M.isTerminal(currentState) == False:
#             #     if [currentState,randomAction] not in visitedList:
#             #         visitedList.append([currentState,randomAction])
#             #     else:
#             #         self.regretUpdate(n, currentState, t)
#
#
#             nextPossStates = self.M.getNextStatesAndProbs(currentState, randomAction, 0)
#             if nextPossStates == []:
#                 done = True
#                 continue
#
#             # If there is only one successor state, pick that
#             if len(nextPossStates) == 1:
#                 # nextPossStates is list of lists
#                 # nextPossStates = [[next_state, prob, reward]]
#                 nextState = nextPossStates[0][0]
#             else:
#                 nextStates = []
#                 nextStateProbs = []
#                 for ns, nsp, _ in nextPossStates:
#                     nextStates.append(ns)
#                     nextStateProbs.append(nsp)
#
#                 nextState = np.random.choice(nextStates, p=nextStateProbs)
#
#             totalActions = []
#             totalActionsProbs = []
#             ta = self.M.getActions(nextState, 0)
#             for action in ta:
#                 totalActions.append(action)
#                 totalActionsProbs.append(self.M.pi[n][self.M.getStateRep(nextState)][action])
#
#             if np.random.randint(0, 100) < int(self.epsilon):
#                 totalActions = self.M.getActions(nextState, 0)
#                 randomActiont = np.random.randint(0, len(totalActions))
#                 randomAction2 = totalActions[randomActiont]
#
#             else:
#
#                 totalActions = []
#                 totalActionsProbs = []
#                 ta = self.M.getActions(nextState, 0)
#                 for action in ta:
#                     totalActions.append(action)
#                     totalActionsProbs.append(self.M.pi[n][self.M.getStateRep(nextState)][action])
#
#                 # print("Current state: ", currentState)
#                 # print("totProb: ", totalActionsProbs)
#                 randomAction2 = np.random.choice(totalActions, p=totalActionsProbs)
#                 randomAct = False
#
#             #A_t+1
#             #randomAction2 = np.random.choice(totalActions, p=totalActionsProbs)
#
#             if nextState == None:
#                 reward = self.M.getReward(currentState, randomAction, n, n)
#                 Value = 0.0 #reward
#                 aValue = 0.0
#
#             else:
#                 reward = self.M.getReward(currentState, randomAction, n, n)
#                 Value = 0.0
#                 for a in self.M.getActions(nextState, n):
#                     #print(n, nextState, a)
#                     Value += self.M.Q[n][nextState][a] * self.M.pi[n][nextState][a]
#
#                 aValue = 0.0
#                 for a in self.M.getActions(nextState, n):
#                     #print(n, nextState, a)
#                     if a != randomAction2:
#                         aValue += self.M.Q[n][nextState][a] * self.M.pi[n][nextState][a]
#
#
#             if self.M.isTerminal(currentState) == False:
#                 if [currentState,randomAction] not in visitedList:
#                     visitedList.append([currentState,randomAction])
#                 # else:
#                 #     self.regretUpdate(n, currentState, t)
#
#             # print(currentState, nextState, randomAction, randomAction2)
#
#             tot = reward + self.gamma*Value
#             updateTotal = reward + self.gamma*(Value - aValue)
#
#             # print(updateTotal, self.M.Q[n][nextState][randomAction2] * self.M.pi[n][nextState][randomAction2])
#
#             #self.M.elig[n][currentState][randomAction] = 1.0
#             # if randomAct == False:
#             #     self.M.elig[n][currentState][randomAction] = 1.0
#             # else:
#             #     for s,a in visitedList:
#             #         self.M.elig[n][s][a] = 0.0
#             #     self.M.elig[n][currentState][randomAction] = 0.0
#             #     visitedList = []
#             #     currentState = nextState
#             #     continue
#
#             lamda = 0.5#1.0#0.90
#             alpha = 0.1
#
#             self.M.Q[n][currentState][randomAction] = (1.0 - alpha)*self.M.Q[n][currentState][randomAction] + alpha * updateTotal#(alpha * (reward + self.gamma * Value - self.M.Q[n][s][a])) * self.M.elig[n][s][a]
#             # self.M.elig[n][s][a] = self.M.elig[n][s][a] * self.gamma * lamda
#             self.M.QSums[n][currentState][randomAction] += self.M.Q[n][currentState][randomAction]
#             self.M.QTouched[n][currentState][randomAction] += 1.0
#             # for s,a in visitedList:
#             #         #self.M.Q[n][s][a] = self.M.Q[n][s][a] + alpha*tot*self.M.elig[n][s][a]#-self.M.Q[n][s][a])
#             #         # self.M.Q[n][s][a] = (1.0 - alpha) * self.M.Q[n][s][a] + (alpha * (reward + self.gamma * Value - self.M.Q[n][s][a]) * self.M.elig[n][s][a])
#             #         self.M.Q[n][s][a] = self.M.Q[n][s][a] + (alpha * (reward + self.gamma * Value - self.M.Q[n][s][a])) * self.M.elig[n][s][a]
#             #         self.M.elig[n][s][a] = self.M.elig[n][s][a] * self.gamma * lamda
#             #         self.M.QSums[n][s][a] += self.M.Q[n][s][a]
#             #         self.M.QTouched[n][s][a] += 1.0
#             # for s in self.M.getStates():
#             #     for a in self.M.getActions(s, n):
#             # for s, a in visitedList:
#             #     self.M.elig[n][s][a] = self.M.elig[n][s][a] * self.gamma * lamda
#             # print("")
#             #visitedList = []
#                 # self.M.QSums[n][currentState][randomAction] += self.M.Q[n][currentState][randomAction]#self.M.Q_bu[n][currentState][aa]
#                 # self.M.QTouched[n][currentState][randomAction] += 1.0
#
#             self.regretUpdate(n, currentState, t)
#
#             currentState = nextState
#             randomAction = randomAction2
#
#             # Check this - catching terminals
#             if self.M.isTerminal(self.M.getStateRep(currentState)):
#                 # print(" Terminal: ", currentState)
#                 a = self.M.getActions(currentState, 0)
#                 # self.QUpdate(n, self.M.getStateRep(currentState), a, self.M.getStateRep(currentState))
#                 for aa in a:
#                     # s, a_current, n, a_notN):
#                     self.M.Q_bu[n][currentState][aa] = self.M.getReward(currentState, aa, n, n)
#                     self.M.Q[n][currentState][aa] = self.M.getReward(currentState, aa, n, n)
#                     self.M.QSums[n][currentState][aa] += self.M.Q[n][currentState][aa]
#                     self.M.QTouched[n][currentState][aa] += 1.0
#
#                 done = True
#                 continue
#
#             # Don't update terminal states
#             # Check if these are ever even hit
#             if self.M.isTerminal(self.M.getStateRep(currentState)) == True:
#                 done = True
#                 continue
#
#             if self.M.getStateRep(currentState) == None:
#                 done = True
#                 continue


################################################################
# QLEARN
################################################################
class QLEARN(LONR):

    def __init__(self, M=None, parameters=None, regret_minimizers=None, dcfr=None):
        super().__init__(M=M, parameters=parameters, regret_minimizers=regret_minimizers, dcfr=dcfr)

    def lonr_train(self, iterations=-1, log=-1, randomize=False):

        self.totalIterations = iterations

        whoGoes = 0.0

        if log != -1: print("Starting training..")
        for t in range(1, iterations+1):

            if self.M.version == 0:
                self.M.version = 1

            if (t+1) % log == 0:
                print("Iteration: ", t+1, " alpha:", self.alpha, " epsilon: ", self.epsilon, " pi: ", self.M.pi[0][self.M.startState])

            self._lonr_train(t=t)

            epp = max(10.0 - (10.0 * t / self.totalIterations), 2)
            self.epsilon = int(epp)

            self.alpha *= self.alphaDecay
            self.alpha = max(0.0, self.alpha)

            if randomize:

                if self.M.version == 1:
                    self.M.version = 2
                else:
                    self.M.version = 1



        if log != -1: print("Finished Training")

    ################################################################
    # QLEARN
    ################################################################
    def _lonr_train(self, t):

        n = 0  # One player

        currentState = self.M.startState
        done = False

        randomAction = None
        # print("-----")
        randomAct = False
        while done == False:

            # print("CS: ", currentState)
            # Check this - tiger game stuff
            if self.M.getStateRep(currentState) == None:
                done = True
                continue

            # Epsilon Greedy action selection
            if np.random.randint(0, 100) < int(self.epsilon):
                totalActions = self.M.getActions(currentState, 0)
                randomAction = np.random.randint(0, len(totalActions))
                randomAction = totalActions[randomAction]
                randomAct = True

            else:

                totalActions = []
                totalActionsProbs = []
                ta = self.M.getActions(currentState, 0)

                maxValue = -10000.0
                maxAction = None
                for aa in ta:#self.M.Q[n][currentState].keys():
                    if self.M.Q[n][currentState][aa] > maxValue:
                        maxValue = self.M.Q[n][currentState][aa]
                        maxAction = aa

                randomAction = maxAction
                randomAct = False

            # print("action: ", randomAction)

            nextPossStates = self.M.getNextStatesAndProbs(currentState, randomAction, 0)
            if nextPossStates == []:
                done = True
                continue

            # If there is only one successor state, pick that
            if len(nextPossStates) == 1:
                # nextPossStates is list of lists
                # nextPossStates = [[next_state, prob, reward]]
                nextState = nextPossStates[0][0]
            else:
                nextStates = []
                nextStateProbs = []
                for ns, nsp, _ in nextPossStates:
                    nextStates.append(ns)
                    nextStateProbs.append(nsp)

                nextState = np.random.choice(nextStates, p=nextStateProbs)


            if nextState == None:
                reward = self.M.getReward(currentState, randomAction, n, n)
                Value = 0.0 #reward

            else:
                reward = self.M.getReward(currentState, randomAction, n, n)
                Value = 0.0
                # for a in self.M.getActions(nextState, n):
                #     #print(n, nextState, a)
                #     Value += self.M.Q[n][nextState][a] * self.M.pi[n][nextState][a]



            # lamda = 0.0#1.0#0.90
            alpha = 0.01#99


            Value = -100000.0
            if nextState != None:
                for a_prime in self.M.Q[n][nextState].keys():
                    if self.M.Q[n][nextState][a_prime] > Value:
                        Value = self.M.Q[n][nextState][a_prime]
                # print("Updating: ", currentState, randomAction)
                self.M.Q[n][currentState][randomAction] = self.M.Q[n][currentState][randomAction] + (alpha * ((((reward + self.gamma * Value)) - self.M.Q[n][currentState][randomAction])))
                self.M.QSums[n][currentState][randomAction] += self.M.Q[n][currentState][randomAction]
                self.M.QTouched[n][currentState][randomAction] += 1.0
            else:
                # print("Will be done")
                # self.M.Q[n][currentState][randomAction] = self.M.getReward(currentState, randomAction, n, n)
                Value = 0.0
                self.M.Q[n][currentState][randomAction] = self.M.Q[n][currentState][randomAction] + (alpha * ((((reward + self.gamma * Value)) - self.M.Q[n][currentState][randomAction])))
                self.M.QSums[n][currentState][randomAction] += self.M.Q[n][currentState][randomAction]
                self.M.QTouched[n][currentState][randomAction] += 1.0
                done = True
                continue
            # self.regretUpdate(n, currentState, t)

            currentState = nextState

            # Check this - catching terminals
            if self.M.isTerminal(self.M.getStateRep(currentState)):
                a = self.M.getActions(currentState, 0)
                Value = 0.0
                for aa in a:
                    Value = self.M.getReward(currentState, aa, n, n)
                    # s, a_current, n, a_notN):
                    self.M.Q_bu[n][currentState][aa] = self.M.getReward(currentState, aa, n, n)
                    self.M.Q[n][currentState][aa] = self.M.getReward(currentState, aa, n, n)
                    self.M.QSums[n][currentState][aa] += self.M.Q[n][currentState][aa]
                    self.M.QTouched[n][currentState][aa] += 1.0


            # Don't update terminal states
            # Check if these are ever even hit
            if self.M.isTerminal(self.M.getStateRep(currentState)) == True:
                done = True
                continue

            if self.M.getStateRep(currentState) == None:
                done = True
                continue



################################################################
# LONR TWO PLAYER Asynchronous Value Iteration
################################################################
class LONR_AA(LONR):

    def __init__(self, M=None, parameters=None, regret_minimizers=None, dcfr=None):
        super().__init__(M=M, parameters=parameters, regret_minimizers=regret_minimizers, dcfr=dcfr)

    def lonr_train(self, iterations=-1, log=-1, randomize=False):


        if log != -1: print("Starting training..")
        for t in range(1, iterations+1):

            if self.M.version == 0:
                self.M.version = 1

            if (t+1) % log == 0:
                print("Iteration: ", t+1, " alpha:", self.alpha, " epsilon: ", self.epsilon)

            self._lonr_train(t=t)#, totalIterations=iterations, randomized=randomized)

            self.alpha *= self.alphaDecay
            self.alpha = max(0.0, self.alpha)



            if randomize:
                # if self.M.version == 1:
                #     self.M.version = 2
                # else:
                #     self.M.version = 1
                # TigerOnLeftProb = self.M.TLProb
                # v = np.random.choice([1,2], p=[TigerOnLeftProb, 1.0-TigerOnLeftProb])
                # self.M.version = v
                # TigerOnLeftProb = self.M.TLProb
                # v = np.random.choice([1, 2], p=[TigerOnLeftProb, 1.0 - TigerOnLeftProb])
                # # self.M.version = v
                if self.M.version == 1:
                    self.M.version = 2
                else:
                    self.M.version = 1


        if log != -1: print("Finished Training")

    ################################################################
    # LONR Asynchronous Value Iteration
    ################################################################
    def _lonr_train(self, t):

        #currentState = self.M.startState
        # n = np.random.randint(0, 2)
        # if n == 0:
        #     currentState = 1
        # else:
        #     currentState = 2
        # currentState = np.random.randint(1,3)
        currentState = 0
        #n = np.random.randint(0,2)
        # if currentState == 1:
        #     n = 0
        # else:
        #     n = 1

        done = False

        totalIts = 0
        # Episode loop - until terminal state is reached
        while done == False:

            # if currentState == 1:
            #     n = 0
            # else:
            #     n = 1

            for NN in range(2):
                #print("TOTAL ITERS: ", totalIts)
                totalIts += 1

                if totalIts % 4000 == 0:
                    print("TOT: ", totalIts)

                if totalIts > 20: # THIS IS NOW TOTAL ITERATIONS
                    done = True
                    continue

                # Get possible actions
                a = self.M.getActions(currentState, NN)
                #print("NN: ", NN, " A: ", a)

                # Update Q of actions
                self.QUpdate(NN, currentState, a, currentState)

                # Q Backup
                # for n in range(self.M.N):
                self.QBackup(NN)

                #s = currentState
                for aa in a:
                    self.M.QSums[NN][currentState][aa] += self.M.Q_bu[NN][currentState][aa]
                    self.M.QTouched[NN][currentState][aa] += 1.0

                # Update regrets
                # if n == 0:
                #     self.regretUpdateNEW(NN, self.M.getStateRep(currentState), t, 1.0, pi1)
                # elif n == 1:
                #     self.regretUpdateNEW(NN, self.M.getStateRep(currentState), t, 1.0, pi0)

                self.regretUpdate(NN, self.M.getStateRep(currentState), t)

                ###self.regretUpdateNEW(NN, self.M.getStateRep(currentState), t, 1.0, 1.0)# 1.0, 1.0)

            #done = True


            #.666111
            # if n == 0:
            #     if currentState == 1:
            #         pi1 *= self.M.pi[1][1]["NOOP"]
            #     else:
            #         pi1 *= self.M.pi[1][2][randomAction]
            # elif n == 1:
            #
            #     if currentState == 1:
            #         pi0 *= self.M.pi[0][1][randomAction]
            #     else:
            #         pi0 *= self.M.pi[0][2]["NOOP"]
            # (cfr_reach, reach) = (pi1, pi0) if n == 0 else (pi0, pi1)
            # self.regretUpdateNEW(n, self.M.getStateRep(currentState), t, cfr_reach, reach)


            # Epsilon Greedy action selection
            # if np.random.randint(0, 100) < int(self.epsilon):
            #     totalActions = self.M.getActions(currentState, n)
            #     randomAction = np.random.randint(0, len(totalActions))
            #     randomAction = totalActions[randomAction]
            #
            # else:
            #
            #     totalActions = []
            #     totalActionsProbs = []
            #     ta = self.M.getActions(currentState, n)
            #     for action in ta:
            #         totalActions.append(action)
            #         totalActionsProbs.append(self.M.pi[n][self.M.getStateRep(currentState)][action])
            #
            #     randomAction = np.random.choice(totalActions, p=totalActionsProbs)
            #
            #
            #
            # nextPossStates = self.M.getNextStatesAndProbs(currentState, randomAction, n)
            # if nextPossStates == []:
            #     done = True
            #     continue
            #
            # # If there is only one successor state, pick that
            # if len(nextPossStates) == 1:
            #     # nextPossStates is list of lists
            #     # nextPossStates = [[next_state, prob, reward]]
            #     currentState = nextPossStates[0][0]
            # else:
            #     nextStates = []
            #     nextStateProbs = []
            #     for ns, nsp, _ in nextPossStates:
            #         nextStates.append(ns)
            #         nextStateProbs.append(nsp)
            #
            #     currentState = np.random.choice(nextStates, p=nextStateProbs)

            #print("CS: ", currentState, "  RAC: ", randomAction , " N: ", n)

            # if n == 0:
            #     pi0 *= self.M.pi[0][currentState][randomAction]
            # else:
            #     pi1 *= self.M.pi[1][currentState][randomAction]

        # print("pi0: ", pi0)
        # print("pi1: ", pi1)
        # print("")


################################################################
# 2 player with CFR LONR Asynchronous Value Iteration
################################################################
class LONR_AAA(LONR):

    def __init__(self, M=None, parameters=None, regret_minimizers=None, dcfr=None):
        super().__init__(M=M, parameters=parameters, regret_minimizers=regret_minimizers, dcfr=dcfr)

    def lonr_train(self, iterations=-1, log=-1, randomize=False):

        if log != -1: print("Starting training..")
        for t in range(1, iterations + 1):

            if self.M.version == 0:
                self.M.version = 1

            if (t + 1) % log == 0:
                print("Iteration: ", t + 1, " alpha:", self.alpha, " epsilon: ", self.epsilon)

            self._lonr_train(t=t)  # , totalIterations=iterations, randomized=randomized)

            self.alpha *= self.alphaDecay
            self.alpha = max(0.0, self.alpha)

            if randomize:
                # if self.M.version == 1:
                #     self.M.version = 2
                # else:
                #     self.M.version = 1
                # TigerOnLeftProb = self.M.TLProb
                # v = np.random.choice([1,2], p=[TigerOnLeftProb, 1.0-TigerOnLeftProb])
                # self.M.version = v
                # TigerOnLeftProb = self.M.TLProb
                # v = np.random.choice([1, 2], p=[TigerOnLeftProb, 1.0 - TigerOnLeftProb])
                # # self.M.version = v
                if self.M.version == 1:
                    self.M.version = 2
                else:
                    self.M.version = 1

        if log != -1: print("Finished Training")

    ################################################################
    # LONR Asynchronous Value Iteration
    ################################################################
    def _lonr_train(self, t):


        currentState = np.random.randint(1, 3)
        # n = np.random.randint(0,2)
        if currentState == 1:
            n = 0
        else:
            n = 1

        done = False
        pi0 = 1.0
        pi1 = 1.0

        totalIts = 0
        # Episode loop - until terminal state is reached
        while done == False:

            if currentState == 1:
                n = 0
            else:
                n = 1

            for NN in range(2):
                # print("TOTAL ITERS: ", totalIts)
                totalIts += 1

                if totalIts % 4000 == 0:
                    print("TOT: ", totalIts)

                if totalIts > 2:  # THIS IS NOW TOTAL ITERATIONS
                    done = True
                    continue

                # Get possible actions
                a = self.M.getActions(currentState, NN)
                # print("NN: ", NN, " A: ", a)

                # Update Q of actions
                self.QUpdate(NN, currentState, a, currentState)

                # Q Backup
                # for n in range(self.M.N):
                self.QBackup(NN)

                # s = currentState
                for aa in a:
                    self.M.QSums[NN][currentState][aa] += self.M.Q_bu[NN][currentState][aa]
                    self.M.QTouched[NN][currentState][aa] += 1.0

                # Update regrets
                # if NN == 0:
                #     self.regretUpdateNEW(NN, self.M.getStateRep(currentState), t, pi1, pi0)
                # elif NN == 1:
                #     self.regretUpdateNEW(NN, self.M.getStateRep(currentState), t, pi0, pi1)

                # self.regretUpdate(NN, self.M.getStateRep(currentState), t)

                # self.regretUpdateNEW(NN, self.M.getStateRep(currentState), t, 1.0, 1.0)# 1.0, 1.0)

            # .666111

            # t_real = (t * 32) + totalIts
            (cfr_reach, reach) = (pi1, pi0) if n == 0 else (pi0, pi1)
            cfr_reach = 1.0
            reach = 1.0
            for nn in range(2):
                self.regretUpdateNEW(nn, self.M.getStateRep(currentState), t, cfr_reach, reach)

            # Epsilon Greedy action selection
            if np.random.randint(0, 100) < int(self.epsilon):
                totalActions = self.M.getActions(currentState, n)
                randomAction = np.random.randint(0, len(totalActions))
                randomAction = totalActions[randomAction]

            else:

                totalActions = []
                totalActionsProbs = []
                ta = self.M.getActions(currentState, n)
                for action in ta:
                    totalActions.append(action)
                    totalActionsProbs.append(self.M.pi[n][self.M.getStateRep(currentState)][action])

                randomAction = np.random.choice(totalActions, p=totalActionsProbs)

            nextPossStates = self.M.getNextStatesAndProbs(currentState, randomAction, n)
            if nextPossStates == []:
                done = True
                continue

            # If there is only one successor state, pick that
            if len(nextPossStates) == 1:
                # nextPossStates is list of lists
                # nextPossStates = [[next_state, prob, reward]]
                currentState = nextPossStates[0][0]
            else:
                nextStates = []
                nextStateProbs = []
                for ns, nsp, _ in nextPossStates:
                    nextStates.append(ns)
                    nextStateProbs.append(nsp)

                currentState = np.random.choice(nextStates, p=nextStateProbs)

            if n == 0:
                if currentState == 1:
                    pi1 *= self.M.pi[1][1]["NOOP"]
                else:
                    pi1 *= self.M.pi[1][2][randomAction]
            elif n == 1:
                if currentState == 1:
                    pi0 *= self.M.pi[0][1][randomAction]
                else:
                    pi0 *= self.M.pi[0][2]["NOOP"]

            # print("CS: ", currentState, "  RAC: ", randomAction , " N: ", n)

            # if n == 0:
            #     pi0 *= self.M.pi[0][currentState][randomAction]
            # else:
            #     pi1 *= self.M.pi[1][currentState][randomAction]

        # print("pi0: ", pi0)
        # print("pi1: ", pi1)
        # print("")




################################################################
# 2 player Bandit
################################################################
class LONR_AB(LONR):

    def __init__(self, M=None, parameters=None, regret_minimizers=None, dcfr=None):
        super().__init__(M=M, parameters=parameters, regret_minimizers=regret_minimizers, dcfr=dcfr)

    def lonr_train(self, iterations=-1, log=-1, randomize=False):

        if log != -1: print("Starting training..")
        for t in range(1, iterations + 1):

            if self.M.version == 0:
                self.M.version = 1

            if (t + 1) % log == 0:
                print("Iteration: ", t + 1, " alpha:", self.alpha, " epsilon: ", self.epsilon)

            self._lonr_train(t=t)  # , totalIterations=iterations, randomized=randomized)

            self.alpha *= self.alphaDecay
            self.alpha = max(0.0, self.alpha)

            if randomize:
                if self.M.version == 1:
                    self.M.version = 2
                else:
                    self.M.version = 1

        if log != -1: print("Finished Training")

    ################################################################
    # LONR
    ################################################################
    def _lonr_train(self, t):


        # currentState = np.random.randint(1, 3)


        # if t % 1000 == 0:
        #     print("Resetting player B's regretsums: t=", t)
        #     for s in self.M.regret_sums[1].keys():
        #         for a in self.M.regret_sums[1][s].keys():
        #             self.M.regret_sums[1][s][a] = 0.0

        currentState = self.M.getStartState()

        # print("Current Start State: ", currentState)

        # if currentState == 1:
        #     n = 0
        # else:
        #     n = 1
        #
        done = False
        # pi0 = 1.0
        # pi1 = 1.0

        totalIts = 0
        # Episode loop - until terminal state is reached
        while done == False:

            # if currentState == 1:
            #     n = 0
            # else:
            #     n = 1

            totalIts += 1
            if totalIts > 2: #Can be 10, 20, etc
                done = True
                continue

            if self.M.isTerminal(currentState):
                done = True
                continue

            # Epsilon Greedy action selection for n=0
            if np.random.randint(0, 100) < int(self.epsilon): #< -100: #
                totalActions = self.M.getActions(currentState, 0)
                randomAction = np.random.randint(0, len(totalActions))
                randomAction0 = totalActions[randomAction]
            else:
                totalActions = []
                totalActionsProbs = []
                ta = self.M.getActions(currentState, 0)
                for action in ta:
                    totalActions.append(action)
                    totalActionsProbs.append(self.M.pi[0][self.M.getStateRep(currentState)][action])
                randomAction0 = np.random.choice(totalActions, p=totalActionsProbs)

            # Epsilon Greedy action selection for n=1
            if np.random.randint(0, 100) < int(self.epsilon): #-100:#
                totalActions = self.M.getActions(currentState, 1)
                randomAction = np.random.randint(0, len(totalActions))
                randomAction1 = totalActions[randomAction]
            else:
                totalActions = []
                totalActionsProbs = []
                ta = self.M.getActions(currentState, 1)
                for action in ta:
                    totalActions.append(action)
                    totalActionsProbs.append(self.M.pi[1][self.M.getStateRep(currentState)][action])
                randomAction1 = np.random.choice(totalActions, p=totalActionsProbs)

            self.QUpdate(0, currentState, [randomAction0], currentState)
            self.QUpdate(1, currentState, [randomAction1], currentState)

            self.M.QSums[0][currentState][randomAction0] += self.M.Q_bu[0][currentState][randomAction0]
            self.M.QTouched[0][currentState][randomAction0] += 1.0
            self.M.QSums[1][currentState][randomAction1] += self.M.Q_bu[1][currentState][randomAction1]
            self.M.QTouched[1][currentState][randomAction1] += 1.0

            for nn in range(self.M.N):
                self.QBackup(nn)

            # self.regretUpdate2(0, self.M.getStateRep(currentState), t, randomAction0)
            # self.regretUpdate2(1, self.M.getStateRep(currentState), t, randomAction1)
            reg1 = self.regretUpdate2(0, self.M.getStateRep(currentState), t, randomAction0)
            reg2 = self.regretUpdate2(1, self.M.getStateRep(currentState), t, randomAction1)

            # if reg1 > 0 and reg2 > 0:
            #     print("POS: ", reg1, reg2, "ACT: ", randomAction0, randomAction1)
            #     self.M.regret_sums[0][self.M.getStateRep(currentState)][randomAction0] += 1.0
            #     self.M.regret_sums[1][self.M.getStateRep(currentState)][randomAction1] += 1.0

            n = 0
            currentState = self.M.getNextStates(n, currentState,randomAction0, randomAction1)
            if self.M.isTerminal(currentState):
                done = True
            # if n == 0:
            #     if currentState == 1:
            #         if randomAction0 == "SEND":
            #             currentState = 2
            #
            # if n == 1:
            #     if currentState == 2:
            #         if randomAction1 == "SEND":
            #             currentState = 1

            # done = True


            ##############################################

            # for NN in range(2):
            #     # print("TOTAL ITERS: ", totalIts)
            #     totalIts += 1
            #
            #     if totalIts % 4000 == 0:
            #         print("TOT: ", totalIts)
            #
            #     if totalIts > 2:  # THIS IS NOW TOTAL ITERATIONS
            #         done = True
            #         continue
            #
            #     # Get possible actions
            #     a = self.M.getActions(currentState, NN)
            #     # print("NN: ", NN, " A: ", a)
            #
            #     # Update Q of actions
            #     self.QUpdate(NN, currentState, a, currentState)
            #
            #     # Q Backup
            #     # for n in range(self.M.N):
            #     self.QBackup(NN)
            #
            #     # s = currentState
            #     for aa in a:
            #         self.M.QSums[NN][currentState][aa] += self.M.Q_bu[NN][currentState][aa]
            #         self.M.QTouched[NN][currentState][aa] += 1.0
            #
            #     # Update regrets
            #     # if NN == 0:
            #     #     self.regretUpdateNEW(NN, self.M.getStateRep(currentState), t, pi1, pi0)
            #     # elif NN == 1:
            #     #     self.regretUpdateNEW(NN, self.M.getStateRep(currentState), t, pi0, pi1)
            #
            #     # self.regretUpdate(NN, self.M.getStateRep(currentState), t)
            #
            #     # self.regretUpdateNEW(NN, self.M.getStateRep(currentState), t, 1.0, 1.0)# 1.0, 1.0)
            #
            # # .666111
            #
            # # t_real = (t * 32) + totalIts
            # (cfr_reach, reach) = (pi1, pi0) if n == 0 else (pi0, pi1)
            # cfr_reach = 1.0
            # reach = 1.0
            # for nn in range(2):
            #     self.regretUpdateNEW(nn, self.M.getStateRep(currentState), t, cfr_reach, reach)
            #
            # # Epsilon Greedy action selection
            # if np.random.randint(0, 100) < int(self.epsilon):
            #     totalActions = self.M.getActions(currentState, n)
            #     randomAction = np.random.randint(0, len(totalActions))
            #     randomAction = totalActions[randomAction]
            #
            # else:
            #
            #     totalActions = []
            #     totalActionsProbs = []
            #     ta = self.M.getActions(currentState, n)
            #     for action in ta:
            #         totalActions.append(action)
            #         totalActionsProbs.append(self.M.pi[n][self.M.getStateRep(currentState)][action])
            #
            #     randomAction = np.random.choice(totalActions, p=totalActionsProbs)
            #
            # nextPossStates = self.M.getNextStatesAndProbs(currentState, randomAction, n)
            # if nextPossStates == []:
            #     done = True
            #     continue
            #
            # # If there is only one successor state, pick that
            # if len(nextPossStates) == 1:
            #     # nextPossStates is list of lists
            #     # nextPossStates = [[next_state, prob, reward]]
            #     currentState = nextPossStates[0][0]
            # else:
            #     nextStates = []
            #     nextStateProbs = []
            #     for ns, nsp, _ in nextPossStates:
            #         nextStates.append(ns)
            #         nextStateProbs.append(nsp)
            #
            #     currentState = np.random.choice(nextStates, p=nextStateProbs)
            #
            # if n == 0:
            #     if currentState == 1:
            #         pi1 *= self.M.pi[1][1]["NOOP"]
            #     else:
            #         pi1 *= self.M.pi[1][2][randomAction]
            # elif n == 1:
            #     if currentState == 1:
            #         pi0 *= self.M.pi[0][1][randomAction]
            #     else:
            #         pi0 *= self.M.pi[0][2]["NOOP"]

