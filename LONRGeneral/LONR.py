
import numpy as np
import math


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

    def QUpdate(self, n, s, a_current2, randomS=None):

        # print("IN:  s: ", s, " RANDOMS: ", randomS)
        for a_current in a_current2:

            if self.M.isTerminal(s):

                # Value iteration
                reward = self.M.getReward(s, a_current, 0, 0)
                # if reward == None:
                #     reward = 0.0
                self.M.Q_bu[n][s][a_current] = reward
                continue


            # Get next states and transition probs for MDP
            # Get next states and other players policy for MG
            succs = self.M.getNextStatesAndProbs(s, a_current, n)
            # print("s: ", s, " a:", a_current, "   succs: ", succs)

            # print("  SUCCS: ", succs)
            Value = 0.0

            # Loop thru each next state and prob
            for s_prime, prob, reward in succs:
                # print(" - s_prime: ", s_prime, "  prob: ", prob, "  reward: ", reward)
                tempValue = 0.0

                # Loop thru actions in s_prime for player n
                for a_current_prime in self.M.getActions(s_prime, n):
                    tempValue += self.M.Q[n][self.M.getStateRep(s_prime)][a_current_prime] * self.M.pi[n][self.M.getStateRep(s_prime)][a_current_prime]

                reww = self.M.getReward(s, a_current, n, 0)

                if reww is None:
                    Value += prob * (reward + self.gamma * tempValue)
                else:
                    Value += prob * (reww + self.gamma * tempValue)

            self.M.Q_bu[n][self.M.getStateRep(s)][a_current] = (1.0 - self.alpha) * self.M.Q[n][self.M.getStateRep(s)][a_current] + (self.alpha) * Value
            print("Q: ", self.M.Q_bu[n][s][a_current], " VALUE: ", Value)

    def QBackup(self, n, WW=None):

        # def QBackup(self, n, WW=None):
        #
        for s in self.M.getStates():
            for a in self.M.getActions(s, n):
                print("QQ: ",  self.M.Q[n][s][a], " QQBU: ", self.M.Q_bu[n][s][a])
                self.M.Q[n][s][a] = self.M.Q_bu[n][s][a]

                self.M.QSums[n][self.M.getStateRep(s)][a] += self.M.Q[n][self.M.getStateRep(s)][a]

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

        # Skip terminal states
        if self.M.isTerminal(self.M.getStateRep(currentState)) == True:
            return

        for a in self.M.getActions(currentState, n):
            target += self.M.Q[n][self.M.getStateRep(currentState)][a] * \
                      self.M.pi[n][self.M.getStateRep(currentState)][a]

        for a in self.M.getActions(currentState, n):
            action_regret = self.M.Q[n][self.M.getStateRep(currentState)][a] - target

            if self.DCFR or self.RMPLUS:
                if action_regret > 0:
                    action_regret *= alphaWeight
                else:
                    action_regret *= betaWeight

            # RMPLUS = False
            if self.RMPLUS:
                self.M.regret_sums[n][self.M.getStateRep(currentState)][a] = max(0.0, self.M.regret_sums[n][self.M.getStateRep(currentState)][a] + action_regret)
            else:
                self.M.regret_sums[n][self.M.getStateRep(currentState)][a] += action_regret

        # Skip terminal states
        # if self.M.isTerminal(self.M.getStateRep(currentState)) == True:
        #     continue

        for a in self.M.getActions(self.M.getStateRep(currentState), n):

            if self.M.isTerminal(currentState):
                continue
            # Sum up total regret
            rgrt_sum = 0.0
            for k in self.M.regret_sums[n][self.M.getStateRep(currentState)].keys():
                rgrt_sum += self.M.regret_sums[n][self.M.getStateRep(currentState)][k] if self.M.regret_sums[n][self.M.getStateRep(currentState)][k] > 0 else 0.0

            # Check if this is a "trick"
            # Check if this can go here or not
            if rgrt_sum > 0:
                self.M.pi[n][self.M.getStateRep(currentState)][a] = (max(self.M.regret_sums[n][self.M.getStateRep(currentState)][a], 0.)) / rgrt_sum
            else:
                self.M.pi[n][self.M.getStateRep(currentState)][a] = 1.0 / len(self.M.getActions(self.M.getStateRep(currentState), n))

            # Add to policy sum
            if self.DCFR or self.RMPLUS:
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

        # Skip terminal states
        if self.M.isTerminal(self.M.getStateRep(currentState)) == True:
            return

        for a in self.M.getActions(currentState, n):
            target += self.M.Q_bu[n][self.M.getStateRep(currentState)][a] * self.M.pi[n][self.M.getStateRep(currentState)][a]

        for a in self.M.getActions(currentState, n):
            action_regret = self.M.Q_bu[n][self.M.getStateRep(currentState)][a] - target

            # if self.DCFR or self.RMPLUS:
            # if action_regret > 0:
            #     action_regret *= alphaWeight
            # else:
            #     action_regret *= betaWeight

            # RMPLUS = False
            # if self.RMPLUS:
            if a != action: continue

            self.M.regret_sums[n][self.M.getStateRep(currentState)][a] = max(0.0, self.M.regret_sums[n][self.M.getStateRep(currentState)][a] + action_regret)
            # else:
            # self.M.regret_sums[n][self.M.getStateRep(currentState)][a] += action_regret * gammaWeight



    # ################################################################
    # # Q Update
    # ################################################################
    # def QUpdate(self, n, s, a_current2, randomS=None):
    #
    #     for a_current in a_current2:
    #
    #         if self.M.isTerminal(s):
    #
    #             # Value iteration
    #             # if self.VI == True:
    #             #     # All except tiger game
    #             reward = 0.0
    #             if randomS is None:
    #                 reward = self.M.getReward(s, a_current, 0, 0)
    #             if reward == None:
    #                 reward = 0.0
    #             self.M.Q_bu[n][s][a_current] = reward
    #             continue
    #
    #
    #
    #         # Get next states and transition probs for MDP
    #         # Get next states and other players policy for MG
    #         succs = self.M.getNextStatesAndProbs(s, a_current, n)
    #         # print("s: ", s, " a:", a_current, "   succs: ", succs)
    #
    #         # print("  SUCCS: ", succs)
    #         Value = 0.0
    #
    #         # Loop thru each next state and prob
    #         for s_prime, prob, reward in succs:
    #             # print(" - s_prime: ", s_prime, "  prob: ", prob, "  reward: ", reward)
    #             tempValue = 0.0
    #
    #             # Loop thru actions in s_prime for player n
    #             for a_current_prime in self.M.getActions(s_prime, n):
    #                 tempValue += self.M.Q[n][self.M.getStateRep(s_prime)][a_current_prime] * self.M.pi[n][self.M.getStateRep(s_prime)][a_current_prime]
    #
    #             print("tempValue: ", tempValue)
    #             reww = self.M.getReward(s, a_current, n, 0)
    #
    #             if reww is None:
    #                 Value += prob * (reward + self.gamma * tempValue)
    #             else:
    #                 Value += prob * (reww + self.gamma * tempValue)
    #
    #         print("Value: ", Value)
    #         self.M.Q_bu[n][self.M.getStateRep(s)][a_current] = (1.0 - self.alpha) * self.M.Q[n][self.M.getStateRep(s)][a_current] + (self.alpha) * Value
    #
    #     # # actions can be a list of actions or a single action
    #     # for a in actions:
    #     #
    #     #     if self.M.isTerminal(s):
    #     #         reward = self.M.getReward(s, a, 0, 0)
    #     #         # print("S: ", s, " a: ", a, " terminal: ", s, " reward: ", reward)
    #     #         self.M.Q_bu[n][self.M.getStateRep(s)][a] = reward
    #     #         continue
    #     #
    #     #
    #     #     # Get next states and transition probs for MDP
    #     #     # Get next states and other players policy for MG
    #     #     succs = self.M.getNextStatesAndProbs(s, a, n)
    #     #
    #     #     #print("SUCCS: ", succs)
    #     #     Value = 0.0
    #     #
    #     #     #Loop thru each next state and prob
    #     #     for s_prime, prob, rew in succs:
    #     #
    #     #         # print("  sprime: ", s_prime)
    #     #         tempValue = 0.0
    #     #
    #     #         # Loop thru actions in s_prime for player n
    #     #         ###if s_prime != None:
    #     #         for a_current_prime in self.M.getActions(s_prime, n):
    #     #             tempValue += self.M.Q[n][self.M.getStateRep(s_prime)][a_current_prime] * self.M.pi[n][self.M.getStateRep(s_prime)][a_current_prime]
    #     #
    #     #
    #     #         # This part is here to correct for the different implementatiosn where in some, the
    #     #         # terminals are included in the successors, and others where they are but FIX THIS
    #     #         reward = self.M.getReward(s, a, n, 0)
    #     #
    #     #         # print("Reward for s: ", s, " a: ", a, "  : ", rew)#ard)
    #     #         if reward is None:
    #     #             Value += prob * (rew + self.gamma * tempValue)
    #     #         else:
    #     #         #print("S: ", self.M.getStateRep(s), " a: ", a, " sprime: ", s_prime, " reward: ", reward)
    #     #             Value += prob * (reward + self.gamma * tempValue)
    #     #
    #     #     # Left in, as alpha=1.0 knocks off the first term.
    #     #     # print("Setting rep(s): ", self.M.getStateRep(s), " to value: ", Value)
    #     #     ###if self.M.getStateRep(s) != None:
    #     #     self.M.Q_bu[n][self.M.getStateRep(s)][a] = (1.0 - self.alpha)*self.M.Q[n][self.M.getStateRep(s)][a] + (self.alpha)*Value
    #
    #
    # ################################################################
    # # Q Backup
    # ################################################################
    # def QBackup(self, n, WW=None):
    #
    #     for s in self.M.getStates():
    #         for a in self.M.getActions(s, n):
    #             self.M.Q[n][s][a] = self.M.Q_bu[n][s][a]
    #             print("Q: ", self.M.Q[n][s][a])
    #             self.M.QSums[n][self.M.getStateRep(s)][a] += self.M.Q[n][self.M.getStateRep(s)][a]
    #
    #
    # ################################################################
    # # Regret/Policy Update
    # ################################################################
    # # TODO:
    # # check that this is correct
    # def regretUpdate(self, n, currentState, t):
    #
    #     # Terminal: No updates, return
    #     if self.M.isTerminal(self.M.getStateRep(currentState)):
    #         return
    #
    #     if self.M.getStateRep(currentState) == None:
    #         return
    #
    #     # Make all of them based on DCFR
    #
    #     # RM+ is
    #     # alphaR = inf
    #     # betaR = -inf
    #     # gammaR = 2
    #
    #     alphaR = 3.0 / 2.0      # accum pos regrets
    #     betaR = 0.0             # accum neg regrets
    #     gammaR = 2.0            # contribution to avg strategy
    #
    #     alphaW = pow(t, self.alphaDCFR)
    #     alphaWeight = (alphaW / (alphaW + 1))
    #
    #     betaW = pow(t, self.betaDCFR)
    #     betaWeight = (betaW / (betaW + 1))
    #
    #     gammaWeight = pow((t / (t + 1)), self.gammaDCFR)
    #
    #
    #     target = 0.0
    #
    #
    #     for a in self.M.getActions(currentState, n):
    #         target += self.M.Q[n][self.M.getStateRep(currentState)][a] * self.M.pi[n][self.M.getStateRep(currentState)][a]
    #
    #     for a in self.M.getActions(currentState, n):
    #         action_regret = self.M.Q[n][self.M.getStateRep(currentState)][a] - target
    #
    #         if self.DCFR:
    #             if action_regret > 0:
    #                 action_regret *= alphaWeight
    #             else:
    #                 action_regret *= betaWeight
    #
    #         #RMPLUS = False
    #         if self.RMPLUS:
    #             self.M.regret_sums[n][self.M.getStateRep(currentState)][a] = max(0.0, self.M.regret_sums[n][self.M.getStateRep(currentState)][a] + action_regret)
    #         else:
    #             self.M.regret_sums[n][self.M.getStateRep(currentState)][a] += action_regret
    #
    #
    #
    #     for a in self.M.getActions(self.M.getStateRep(currentState), n):
    #
    #         if self.M.isTerminal(currentState):
    #             continue
    #         # Sum up total regret
    #         rgrt_sum = 0.0
    #         for k in self.M.regret_sums[n][self.M.getStateRep(currentState)].keys():
    #             rgrt_sum += self.M.regret_sums[n][self.M.getStateRep(currentState)][k] if self.M.regret_sums[n][self.M.getStateRep(currentState)][k] > 0 else 0.0
    #
    #         # Check if this is a "trick"
    #         # Check if this can go here or not
    #         if rgrt_sum > 0:
    #             self.M.pi[n][self.M.getStateRep(currentState)][a] = (max(self.M.regret_sums[n][self.M.getStateRep(currentState)][a], 0.)) / rgrt_sum
    #         else:
    #             self.M.pi[n][self.M.getStateRep(currentState)][a] = 1.0 / len(self.M.getActions(self.M.getStateRep(currentState), n))
    #
    #         # Add to policy sum
    #         if self.DCFR:
    #             self.M.pi_sums[n][self.M.getStateRep(currentState)][a] += self.M.pi[n][self.M.getStateRep(currentState)][a] * gammaWeight
    #         else:
    #             self.M.pi_sums[n][self.M.getStateRep(currentState)][a] += self.M.pi[n][self.M.getStateRep(currentState)][a]
    #





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

        for t in range(1, iterations + 1):

            if (t + 0) % log == 0:
                print("Iteration: ", t + 0, " alpha: ", self.alpha, " gamma: ", self.gamma)

            # Call one full update via LONR-V
            self._lonr_train(t=t)

            # No-op unless alphaDecay is not 1.0
            self.alpha *= self.alphaDecay
            self.alpha = max(0.0, self.alpha)

            if randomize:
                if self.M.version == 1:
                    self.M.version = 2
                else:
                    self.M.version = 1

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
                    a = self.M.getActions(s, 0)
                    self.QUpdate(n, s, a, randomS=s)

                    continue

                # Loop through actions of current player n
                # for a in self.M.getActions(s, n):
                a = self.M.getActions(self.M.getStateRep(s), n)
                #if self.randomize == False:
                self.QUpdate(n, s, a, randomS=None)
                #else:
                #    self.QUpdate(n, s=s, a_current2=a, randomS=s)



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

    def __init__(self, M=None, parameters=None, regret_minimizers=None):
        super().__init__(M=M, parameters=parameters, regret_minimizers=regret_minimizers)

    def lonr_train(self, iterations=-1, log=-1, randomize=False):


        if log != -1: print("Starting training..")
        for t in range(1, iterations+1):

            if (t+1) % log == 0:
                print("Iteration: ", t+1, " alpha:", self.alpha, " epsilon: ", self.epsilon)

            self._lonr_train(t=t)#, totalIterations=iterations, randomized=randomized)

            self.alpha *= self.alphaDecay
            self.alpha = max(0.0, self.alpha)

            if randomize:

                TigerOnLeftProb = self.M.TLProb
                v = np.random.choice([1,2], p=[TigerOnLeftProb, 1.0-TigerOnLeftProb])
                self.M.version = v
                # if self.M.version == 1:
                #     self.M.version = 2
                #     # self.M.startState = "rootTR"
                # else:
                #     self.M.version = 1
                #     # self.M.startState = "rootTL"


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
            for n in range(self.M.N):
                self.QBackup(n)

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





####################################################################
# LONR Asynchronous Value Iteration - Version 2&3
####################################################################
class LONR_A2(LONR):

    def __init__(self, M=None, parameters=None, regret_minimizers=None):
        super().__init__(M=M, parameters=parameters, regret_minimizers=regret_minimizers)

    def lonr_train(self, iterations=-1, log=-1, randomize=False):


        if log != -1: print("Starting training..")
        for t in range(1, iterations+1):

            if t % log == 0:
                print("Iteration: ", t, " alpha:", self.alpha, " epsilon: ", self.epsilon)

            self._lonr_train(t=t)#, totalIterations=iterations, randomized=randomized)

            self.alpha *= self.alphaDecay
            self.alpha = max(0.0, self.alpha)

            if randomize:
                if self.M.version == 1:
                    self.M.version = 2
                    # self.M.startState = "rootTR"
                else:
                    self.M.version = 1
                    # self.M.startState = "rootTL"


        if log != -1: print("Finished Training")

    ################################################################
    # LONR Asynchronous Value Iteration 2&3
    ################################################################
    def _lonr_train(self, t):


        currentState = self.M.startState


        done = False
        n = 0  # One player

        # Episode loop - until terminal state is reached
        while done == False:

            #print("CS: ", currentState)
            # if self.M.isTerminal(self.M.getStateRep(currentState)) == True:
            #     done = True
            #     continue

            # 1. Pick a random action epsilon greedy
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

            #print("  randomAction: ", randomAction)
            # 2. Get the possible successor states
            nextPossStates = self.M.getNextStatesAndProbsGrid(currentState, randomAction, 0)

            finalAction = None
            # Pick a successor state as nextState
            # If there is only one successor state, pick that
            if len(nextPossStates) == 0:
                lalala = 1
                nextState = None
                nextProb  = 1.0
                nextReward = 0.0
                nextAction = None
            elif len(nextPossStates) == 1:
                # nextPossStates is list of lists
                # nextPossStates = [[next_state, prob, reward]]
                nextState = nextPossStates[0][0]
                nextProb = nextPossStates[0][1]
                nextReward = nextPossStates[0][2]
                nextAction = nextPossStates[0][3]
                #nextReward = self.M.getReward(currentState, nextAction, 0, 0)
            else:
                nextStates = []
                nextStateProbs = []
                nextStateRewards = []
                nextStateActions = []
                for ns, nsp, nsw, na in nextPossStates:
                    nextStates.append(ns)
                    nextStateProbs.append(nsp)
                    nextStateRewards.append(nsw)
                    nextStateActions.append(na)

                #print("NS: ", nextStates)
                nextIndex = np.random.choice(len(nextStates), p=nextStateProbs)
                #print("NI: ", nextIndex)
                nextState = nextStates[nextIndex]#np.random.choice(nextStates, p=nextStateProbs)
                nextProb = nextStateProbs[nextIndex]
                nextReward = nextStateRewards[nextIndex]
                nextAction = nextStateActions[nextIndex]

                nextReward = self.M.getReward(currentState, nextAction,0,0)


            tempValue = 0.0

            # Loop thru actions in s_prime for player n
            if nextState != None:
                for a_current_prime in self.M.getActions(nextState, n):
                    tempValue += self.M.Q[n][self.M.getStateRep(nextState)][a_current_prime] * self.M.pi[n][self.M.getStateRep(nextState)][a_current_prime]

            Value = nextProb * (nextReward + self.gamma * tempValue)

            if self.M.getStateRep(currentState) != None:
                self.M.Q_bu[n][self.M.getStateRep(currentState)][randomAction] = (1.0 - self.alpha)*self.M.Q[n][self.M.getStateRep(currentState)][randomAction] + (self.alpha)*Value

            # Check this - tiger game stuff
            if self.M.getStateRep(currentState) == None:
                done = True
                continue

            # Check this - catching terminals
            if self.M.isTerminal(self.M.getStateRep(currentState)):
                a = self.M.getActions(currentState, 0)
                self.QUpdate(n, self.M.getStateRep(currentState), a, self.M.getStateRep(currentState))
                for n in range(self.M.N):
                    self.QBackup(n)
                done = True
                continue

            self.M.Q[0][self.M.getStateRep(currentState)][randomAction] = self.M.Q_bu[n][self.M.getStateRep(currentState)][randomAction]
            self.M.QSums[0][self.M.getStateRep(currentState)][randomAction] += self.M.Q[n][self.M.getStateRep(currentState)][randomAction]
            self.M.QTouched[n][self.M.getStateRep(currentState)][randomAction] += 1.0



            # if self.M.isTerminal(self.M.getStateRep(currentState)):
            #     a = self.M.getActions(self.M.getStateRep(currentState), 0)
            #     self.QUpdate(n, self.M.getStateRep(currentState), a, self.M.getStateRep(currentState))
            #     done = True
            #     continue


            # Get possible actions
            #a = self.M.getActions(currentState, 0)

            # Update Q of actions
            #self.QUpdate(n, currentState, a, currentState)


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

            if nextState == None:
                done = True
            # Update regrets
            self.regretUpdate(n, self.M.getStateRep(currentState), t)


            currentState = nextState





class LONR_A3(LONR):

    def __init__(self, M=None, parameters=None, regret_minimizers=None):
        super().__init__(M=M, parameters=parameters, regret_minimizers=regret_minimizers)

    def lonr_train(self, iterations=-1, log=-1, randomize=False):


        if log != -1: print("Starting training..")
        for t in range(1, iterations+1):

            if (t+1) % log == 0:
                print("Iteration: ", t+1, " alpha:", self.alpha, " epsilon: ", self.epsilon)

            self._lonr_train(t=t)#, totalIterations=iterations, randomized=randomized)

            self.alpha *= self.alphaDecay
            self.alpha = max(0.0, self.alpha)

            if randomize:

                TigerOnLeftProb = self.M.TLProb
                v = np.random.choice([1,2], p=[TigerOnLeftProb, 1.0-TigerOnLeftProb])
                self.M.version = v
                # if self.M.version == 1:
                #     self.M.version = 2
                #     # self.M.startState = "rootTR"
                # else:
                #     self.M.version = 1
                #     # self.M.startState = "rootTL"


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
            for n in range(self.M.N):
                self.QBackup(n)

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
    # QLearning
    ################################################################


    def getQValue(self, state, action):
        return self.M.Q[0][state][action]

    def computeValueFromQValues(self, state):
        legalActions = self.M.getActions(state,0)
        if len(legalActions) != 0:
            qValues = []
            for action in legalActions:
                qValues.append(self.getQValue(state, action))
            return max(qValues)
        else:

            return 0.0
        #util.raiseNotDefined()

    def computeActionFromQValues(self, state):

        legalActions = self.M.getActions(state, 0)

        if len(legalActions) != 0:
            stateValues = []
            possibleActions = []
            for action in legalActions:
                stateValues.append(self.getQValue(state, action))
                possibleActions.append(action)

            maxValue = max(stateValues)
            bestIndices = [index for index in range(len(stateValues)) if stateValues[index] == maxValue]
            chosenIndex = np.random.choice(bestIndices)
            return possibleActions[chosenIndex]
        else:

            return "exit"

    def getAction(self, state):
        if self.M.isTerminal(state):
            return "exit"
        if np.random.randint(0, 100) < self.epsilon:
            return np.random.randint(0, 4)
        else:
            action = self.getPolicy(state)
            return action

    def update(self, state, action, nextState, reward):
        gamma = self.gamma
        alpha = self.alpha

        sample = reward + gamma * self.getValue(nextState)
        self.M.Q[0][state][action] = self.M.Q[0][state][action] + self.alpha * (sample - self.M.Q[0][state][action])

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


    def qlearning_train(self, iterations=-1, log=-1, randomize=False):

        if log != -1: print("Starting training..")

        for t in range(1, iterations + 1):

            if (t + 0) % log == 0:
                print("Iteration: ", t + 0, " alpha: ", self.alpha, " gamma: ", self.gamma, " alphaDecay: ", self.alphaDecay)

            # Call one full update via LONR-V
            self._qlearning_train(t=t)

            # No-op unless alphaDecay is not 1.0
            self.alpha *= self.alphaDecay
            self.alpha = max(0.0, self.alpha)

            # if randomize:
            #     if self.M.version == 1:
            #         self.M.version = 2
            #     else:
            #         self.M.version = 1

        if log != -1: print("Finish Training")


    def _qlearning_train(self, t):


        currentState = self.M.startState

        done = False
        n = 0  # One player

        # Episode loop - until terminal state is reached
        while done == False:

            # print("Current State: ", currentState)

            if self.M.isTerminal(currentState):
                a = self.M.getActions(currentState, 0)
                for a in self.M.getActions(currentState, 0):
                    self.M.Q[0][currentState][a] = self.M.getReward(currentState, a, 0, 0)#(1.0-self.alpha)*self.M.Q[0][currentState][a] + self.alpha * self.M.getReward(currentState, a, 0, 0)
                done = True
                continue


            randomAction = self.getAction(currentState)

            s_prime = self.M.getMove(currentState, randomAction)
            rew = self.M.livingReward# -1.0# succs[0][2]

            self.update(currentState, randomAction, s_prime, rew)

            currentState = s_prime





################################################################
# MISC
#
#     def _lonr_value_iteration(self, t):
#
#         WW = None
#         # Q Update
#         # All except Tiger Game
#         # if self.randomize == False:
#
#         for n in range(self.M.N):
#
#             # Loop through all states
#             for s in self.M.getStates():
#
#                 # Loop through actions of current player n
#                 # for a in self.M.getActions(s, n):
#                 a = self.M.getActions(self.M.getStateRep(s), n)
#                 if self.randomize == False:
#                     self.QUpdate(n, s, a, randomS=None)
#                 else:
#                     self.QUpdate(n, s=s, a_current2=a, randomS=s)
#
#         # Tiger Game
#         # elif self.randomize == True:
#         #     print("Tiger game - value iteration")
#         #     r = np.random.randint(0, 2)
#         #     #if r == 0:
#         #     if self.alt == 1:
#         #         WW = self.M.totalStatesLeft
#         #         self.alt = 2
#         #     else:
#         #         WW = self.M.totalStatesRight
#         #         self.alt = 1
#         #
#         #     print("WW: ")
#         #     print(WW)
#         #     for n in range(self.M.N):
#         #
#         #         # Loop through all states
#         #         for s in WW: #self.M.getStates(): #WW:
#         #
#         #             print(" S in loop: ", s)
#         #             # Loop through actions of current player n
#         #             # for a in self.M.getActions(s, n):
#         #             aa = self.M.getActions(s, n)
#         #             # if self.randomize == False:
#         #             #     self.QUpdate(n, s, a, randomS=None)
#         #             # else:
#         #             print("AA: ", aa)
#         #             for a in aa:
#         #                 #print(" a: ", a)
#         #                 al = []
#         #                 al.append(a)
#         #                 print(" a: ", a)
#         #                 self.QUpdate(n, s=s, a_current2=al, randomS=s)
#
#         # Q Backup
#         for n in range(self.M.N):
#             self.QBackup(n) #, WW=WW)
#
#         # Update regret sums, pi, pi sums
#
#         if self.randomize == False:
#             for n in range(self.M.N):
#                 for s in self.M.getStates():
#                     self.regretUpdate(n, s, t)
#         else:
#             for n in range(self.M.N):
#                 for s in self.M.getStates(): #WW: #self.M.totalStatesLeft: #WW:
#                     if self.M.isTerminal(s): continue
#                     self.regretUpdate(n, s, t)
#
#
#
#
#
#     def lonr_online(self, iterations=-1, log=-1, randomized=False):
#
#         print("Starting training..")
#         for t in range(1, iterations+1):
#
#             if (t+1) % log == 0:
#                 print("Iteration: ", t+1, " alpha:", self.alpha, " epsilon: ", self.epsilon)
#
#             self._lonr_online(t=t, totalIterations=iterations, randomized=randomized)
#
#             self.alpha *= self.alphaDecay
#             self.alpha = max(0.0, self.alpha)
#
#
#         print("Finished Training")
#
#
#     def _lonr_online(self, t=0, totalIterations=-1, randomized=False):
#         """
#
#         :param t: current iteration
#         :param totalIterations: total iterations
#         :param randomized: Randomizes the start state.
#                         True: For Tiger Game, to switch between TigerOnLeft, TigerOnRight
#                         False: For GridWorld, only one start state.
#         :return:
#         """
#
#
#         # For Tiger Game, randomize the MDP it sees
#         if randomized:
#
#             # Get the top root
#             startStates = self.M.getNextStatesAndProbs(self.M.startState, None, 0)
#
#             totalStartStates = []
#             totalStartStateProbs = []
#
#             # Pick Tiger on Left/Right based on probability set in M
#             for nextState, nextStateProb, _ in startStates:
#                 totalStartStates.append(nextState)
#                 totalStartStateProbs.append(nextStateProb)
#
#             # Randomly pick TigerOnLeft/Right based on M.TLProb (1-M.TLProb = TRProb)
#             currentState = np.random.choice(totalStartStates, p=totalStartStateProbs)
#
#         # For GridWorld, set currentState to the startState
#         else:
#
#             currentState = self.M.startState
#
#         #print("RANDOMIZED: ", randomized)
#         done = False
#         n = 0  # One player
#
#
#
#         # Episode loop - until terminal state is reached
#         while done == False:
#
#             #print("Current state: ", currentState)
#             if self.EXP3 == False:
#                 if self.M.isTerminal(currentState) == True:
#                     # for a in self.M.getActions(currentState, 0):
#                     #     # Note, the reward is via the actual state, so there is no getStateRep()
#                     #     self.M.Q_bu[n][self.M.getStateRep(currentState)][a] = (1.0 - self.alpha) * self.M.Q[n][self.M.getStateRep(currentState)][a] + self.alpha * self.M.getReward(currentState,a,a,a)
#                     #print("TEEEEEEEEEEEEEEEETMMR")
#                     a = self.M.getActions(currentState, 0)
#                     if self.LONR == True:
#                         self.QUpdate(n, currentState, a, currentState)
#                     else:
#                         #print("LALALALLALA")
#                         self.QLearningUpdate(n, currentState, a, currentState)
#                     # print("DONE = ", done)
#                     done = True
#                     continue
#
#                 a = self.M.getActions(currentState, 0)
#                 if self.LONR == True and self.EXP3 == False:
#                     self.QUpdate(n, currentState, a, currentState)
#                 else:
#                     self.QLearningUpdate(n, currentState, a, currentState)
#                 # totStates keeps track of which states need Qvalue copying
#                 #   as not all states need to be backed up, only the ones visited
#                 # totStates = []
#                 # totStates.append(currentState)
#
#             # Q Backup
#             for n in range(self.M.N):
#                 self.QBackup(n)
#
#
#             # Don't update terminal states
#             if self.EXP3 == False:
#                 if self.M.isTerminal(self.M.getStateRep(currentState)) == True:
#                     done = True
#                     continue
#
#             if self.EXP3 == False:
#
#                 if self.LONR == True:
#                     self.regretUpdate(n, self.M.getStateRep(currentState), t)
#                 else:
#                     noRegretUpdate = True
#                     #self.QLearningPIUpdate()
#
#                 # Epsilon Greedy action selection
#                 if np.random.randint(0, 100) < int(self.epsilon):
#                     totalActions = self.M.getActions(currentState, 0)
#                     randomAction = np.random.randint(0, len(totalActions))
#                     randomAction = totalActions[randomAction]
#                     #print("RAndom action: ", randomAction)
#
#                 else:
#
#                     totalActions = []
#                     totalActionsProbs = []
#                     ta = self.M.getActions(currentState, 0)
#                     for action in ta:
#                         totalActions.append(action)
#                         totalActionsProbs.append(self.M.pi[n][self.M.getStateRep(currentState)][action])
#                     #print(totalActionsProbs)
#                     if self.LONR == True:
#                         randomAction = np.random.choice(totalActions, p=totalActionsProbs)
#                     else:
#                         randomActionValue = max(totalActionsProbs)
#                         randomAction = totalActionsProbs.index(randomActionValue)
#                         #print("Random Action: ", randomAction)
#
#             #Exp 3 is true
#             else:
#
#                 randomAction = self.exp3Update(n, currentState, 4, 0, 0.5, t)
#             # randomAction picked, now simulate taking action
#             #       GridWorld: This will handle non-determinism, if there is non-determinism
#             #       TigerGame: This will either get the one next state OR
#             #                       if action is LISTEN, it will return next state based on
#             #                       observation accuracy aka (85/15) or (15/85), which is
#             #                       equivalent to hearing a growl left or growl right
#             nextPossStates = self.M.getNextStatesAndProbs(currentState, randomAction, 0)
#             #print("SS: ", nextPossStates)
#
#             #print("NPS: ", nextPossStates)
#             # If there is only one successor state, pick that
#             if len(nextPossStates) == 1:
#                 # nextPossStates is list of lists
#                 # nextPossStates = [[next_state, prob, reward]]
#                 currentState = nextPossStates[0][0]
#             else:
#                 nextStates = []
#                 nextStateProbs = []
#                 for ns, nsp,_ in nextPossStates:
#                     nextStates.append(ns)
#                     nextStateProbs.append(nsp)
#                 #print("Nextposs: ", nextPossStates)
#                 currentState = np.random.choice(nextStates, p=nextStateProbs)
#             print("CURRENT STATE: ", currentState, )
#
#             if self.EXP3 == True:
#                 for n in range(self.M.N):
#                     self.QBackup(n)
#                 if self.M.isTerminal(self.M.getStateRep(currentState)) == True:
#                     done = True
#                     continue
#             # for n in range(self.M.N):
#             #     self.QBackup(n)
#             # if self.LONR == False:
#             #     aa = []
#             #     aa.append(randomAction)
#             #     self.QLearningUpdate(n, currentState, aa, currentState)
#             # print("Current state: ", currentState)






###############################################################


class LONR_B(LONR):

    def __init__(self, M=None, parameters=None, regret_minimizers=None):
        super().__init__(M=M, parameters=parameters, regret_minimizers=regret_minimizers)

        # 1. Set all weights to 1.0 (done in GridWorld MDP)


    ################################################################
    # LONR Bandit
    ################################################################
    def lonr_train(self, iterations=-1, log=-1, randomize=False):

        if log != -1: print("Starting training..")

        self.totalIterations = iterations

        for t in range(1, iterations + 1):

            if (t + 0) % log == 0:
                print("Iteration: ", t + 0, " alpha: ", self.alpha, " gamma: ", self.gamma)

            # Call one full update via LONR-V
            self._lonr_train(t=t)

            # No-op unless alphaDecay is not 1.0
            # self.alpha *= self.alphaDecay
            # self.alpha = max(0.0, self.alpha)
            #
            # if randomize:
            #     if self.M.version == 1:
            #         self.M.version = 2
            #     else:
            #         self.M.version = 1

        if log != -1: print("Finish Training")

    ######################################
    ## LONR Bandit
    ######################################
    def _lonr_train(self, t):

        # Set start state or randomize start state
        currentState = self.M.startState
        #currentState = np.random.randint(0, 37)


        # Settings
        done = False
        n = 0  # One player
        nextAction = None

        # Episode loop - until terminal state is reached
        while done == False:

            # Periodic output
            if t % 12500 == 0:
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
                done = True
                continue

            # everything done in exp3, just get the action it used
            # Note: return action and state for non-determinism (might be different here)
            nextAction = self.exp3(currentState, t)

            s_prime = self.M.getMove(currentState, nextAction)

            rew = self.M.getReward(currentState, nextAction, 0, 0)

            Value = 0.0
            for a_prime in self.M.getActions(s_prime, n):
                Value += self.M.pi[n][s_prime][a_prime] * self.M.Q[n][s_prime][a_prime]

            x = rew + self.gamma * Value

            self.M.Q_bu[0][currentState][nextAction] = self.alpha*x + (1.0 - self.alpha)*self.M.Q[0][currentState][nextAction] #(rew + self.gamma * Value)(1.0 / self.M.pi[0][currentState][nextAction]) *

            for aa in self.M.Q[0][currentState].keys():

                if aa == nextAction:
                    self.M.Q[0][currentState][aa] = self.M.Q_bu[0][currentState][aa]
                    self.M.QSums[0][currentState][aa] += self.M.Q[0][currentState][aa]
                    self.M.QTouched[0][currentState][aa] += 1.0
                else:
                    self.M.Q[0][currentState][aa] = 0.0
                    self.M.Q_bu[0][currentState][aa] = 0.0

            if verbose: print("")
            if verbose: print("Current State: ", currentState, "  currentAction: ", nextAction, " t=", t)

            nextState = self.M.getMove(currentState, nextAction)
            currentState = nextState



    # def draw(self, weights, gamma=0.0):
    #
    #     # Sum up the (exponential) weights and sum
    #     theSum = 0.0
    #     theWeights = []
    #     for w in weights:
    #         theSum += math.exp(w)# * gamma / 4.0)
    #         theWeights.append(math.exp(w))# * gamma / 4.0))
    #
    #     # Normalize
    #     for i in range(len(theWeights)):
    #         theWeights[i] = theWeights[i] / theSum
    #
    #     # Draw action from prob dist
    #     c = np.random.choice([0,1,2,3], p=theWeights)
    #     return c

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


    def exp3(self, currentState, t):

        # Keep around for decaying exp3 gamma
        #gamma = max(1.0 - (float(float(4.0/2.0)*t) / float(self.totalIterations)), 0.1)

        # Show occasional printout
        if t % 12500 == 0:
            verbose = True
        else:
            verbose = False

        rewardMin = -1.0
        rewardMax = 10.0

        numActions = 4.0
        n = 0
        gamma = 0.2
        maxGap = 20.0



        # Get current weights for 0, 1, 2 ,3
        # Only for weights above
        weights = []
        maxWeight = -10000000
        if verbose: print("Weights: ")
        for w in sorted(self.M.weights[n][currentState].keys()):
            if verbose: print("   ", w, ": ", self.M.weights[n][currentState][w])
            if self.M.weights[n][currentState][w] > maxWeight:
                maxWeight = self.M.weights[n][currentState][w]

        if verbose: print("Max Weight: ", maxWeight)

        maxWeightGap = maxWeight - maxGap
        if verbose: print("Only considering those above maxWeightGap: ", maxWeightGap, ", else 0")
        for w in sorted(self.M.weights[n][currentState].keys()):
            if self.M.weights[n][currentState][w] > maxWeightGap:
                weights.append(self.M.weights[n][currentState][w] - maxWeightGap)
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
        for a in sorted(self.M.pi[n][currentState].keys()):
            if verbose: print("Setting pi of s: ", currentState, " a: ", a, " : ", pd[a])
            if self.M.weights[n][currentState][a] > maxWeightGap:
                self.M.pi[n][currentState][a] = pd[a]
                self.M.pi_sums[n][currentState][a] += piSUMDist[int(a)]
            else:
                self.M.pi[n][currentState][a] = pd[a]
                #self.M.pi_sums[n][currentState][a] += piSUMDist2[int(a)]

        # Select a random action from pd
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
        scaledReward = (x - rewardMin) / (rewardMax - rewardMin)
        if scaledReward > 1.0 and t % 2000 == 0:
            print("SCALED REWARD: ",scaledReward, " currentState: ", currentState)
        if scaledReward < 0 and t % 2000 == 0:
            print("SCALED REWARD: ", scaledReward, " currentState: ", currentState)
        # x = max(x, 0.0)
        # x = min(x, 1.0)

        # Get expected reward
        estimatedReward = scaledReward


        runningEstimatedReward = estimatedReward / pd[randomAction]

        # Find min weight
        # currentRunningWeights = []
        for r in sorted(self.M.runningRewards[0][currentState].keys()):
            if r == randomAction:
                self.M.runningRewards[0][currentState][r] += runningEstimatedReward# / self.M.pi[n][currentState][r]
                self.M.Q_bu[0][currentState][r] = runningEstimatedReward# / self.M.pi[n][currentState][r]
            else:
                self.M.Q_bu[0][currentState][r] = 0.0



        # Set Weight for t+1
        # w_t+1() = w_t()e^(x_hat * gamma / K) = 1.0 ^ e^A * e^b
        # Keep sum of weights and put in exponent when calculating prob dist above
        if verbose: print(runningEstimatedReward)
        for j in sorted(self.M.weights[0][currentState].keys()):
            if j == randomAction:
                self.M.weights[0][currentState][j] += self.M.Q_bu[0][currentState][j] * gamma / float(numActions) #*= math.exp(self.M.runningRewards[0][currentState][j] * gamma / float(numActions))
                #self.M.weights[0][currentState][j] *= math.exp(self.M.runningRewards[0][currentState][j] * gamma / float(numActions))

        if verbose: print(self.M.weights[0][4])

        return randomAction


class LONR_B2(LONR):

    def __init__(self, M=None, parameters=None, regret_minimizers=None):
        super().__init__(M=M, parameters=parameters, regret_minimizers=regret_minimizers)

        # 1. Set all weights to 1.0 (done in GridWorld MDP)


    ################################################################
    # LONR Bandit
    ################################################################
    def lonr_train(self, iterations=-1, log=-1, randomize=False):

        if log != -1: print("Starting training..")
        gamma = 0.2#1.0
        for t in range(1, iterations + 1):

            if (t + 0) % log == 0:
                print("Iteration: ", t + 0, " alpha: ", self.alpha, " gamma: ", self.gamma, " othergamma: ", gamma)

            # Call one full update via LONR-V
            self._lonr_train(t=t, gamma=gamma)
            #gamma *= 0.99999
            #gamma = max(1.0 - (float((1.0/2.0)*t) / float(iterations )), 0.0)
        if log != -1: print("Finish Training")

    ######################################
    ## LONR Bandit
    ######################################
    def _lonr_train(self, t, gamma=0.0):


        currentState = self.M.startState

        #currentState = np.random.randint(0, 12)
        done = False
        n = 0  # One player

        # print("")
        # print("")
        # print("---------------------------------")
        # print("New episode")

        nextAction = None
        # Episode loop - until terminal state is reached
        c = 0
        visited = []
        while done == False:

            if currentState not in visited:
                visited.append(currentState)
            if t % 3500 == 0:
                verbose = True
            else:
                verbose = False
            # verbose = True
            if verbose: print("")
            if verbose: print("Current State: ", currentState, "  currentAction: ", nextAction, " t=", t)

            if self.M.isTerminal(currentState):
                for a in self.M.getActions(currentState, n):
                    self.M.Q[n][currentState][a] = self.M.getReward(currentState, a,0,0)
                    self.M.Q_bu[n][currentState][a] = self.M.getReward(currentState, a, 0, 0)
                    #print("Final: set s: ", currentState, " to: ", self.M.getReward(currentState, a,0,0))
                done = True
                continue

            nextAction = self.exp3(currentState, t, gamma=gamma)

            s_prime = self.M.getMove(currentState, nextAction)

            rew = self.M.getReward(currentState, nextAction, 0, 0)

            Value = 0.0
            for a_prime in self.M.getActions(s_prime, n):
                Value += self.M.pi[n][s_prime][a_prime] * self.M.Q[n][s_prime][a_prime]

            x = rew + self.gamma * Value

            self.M.Q_bu[0][currentState][nextAction] = (self.alpha * x ) + ((1 - self.alpha) * self.M.Q[0][currentState][nextAction]) #(rew + self.gamma * Value)(1.0 / self.M.pi[0][currentState][nextAction]) *
            #self.M.Q[0][currentState][nextAction] = self.M.Q_bu[0][currentState][nextAction]

            #self.regretUpdate2(0, currentState, t)

            nextState = self.M.getMove(currentState, nextAction)
            currentState = nextState


        for s in visited:
            for a in self.M.getActions(s, 0):
                self.M.Q[0][s][a] = self.M.Q_bu[0][s][a]
                self.M.QSums[0][s][a] += self.M.Q[0][s][a]
                self.M.QTouched[0][s][a] += 1.0

        for s in visited:
            for a in self.M.getActions(s, 0):
                self.regretUpdate2(0, s, t, a)
        visited = []
            #done = False
            # if self.M.isTerminal(currentState):
            #     done = True
            #     continue
            #
            # if nextState == 1:
            #     for i in range(100):
            #         print("JUST HIT 7777777777777777************************************************")

            #def exp3Update(self, n, currentState, numActions, t, rewardMin=-100.0, rewardMax=200.0):
            # nextAction = self.exp3Update(n, currentState, 4, t)
            #
            # self.regretUpdate(n, self.M.getStateRep(currentState), t)
            #
            # for n in range(self.M.N):
            #     for s in range(len(self.M.getStates())):
            #         for a in range(len(self.M.getActions(s,n))):
            #             self.M.Q[n][s][a] = self.M.Q_bu[n][s][a]
            #
            # nextState = self.M.getMove(currentState, nextAction)
            #
            # currentState = nextState
            #
            # if self.M.isTerminal(currentState):
            #     done = True
            #     # for a in self.M.getActions(currentState, n):
            #     #     self.M.Q[n][currentState][a] = self.M.getReward(currentState,a,0,0)
            #     continue




    def exp3(self, currentState, t, gamma=0.0):

        regrets = {}
        regretSum = 0.0
        for r in sorted(self.M.regret_sums[0][currentState].keys()):
            # print("Regret for action: ", r)
            regrets[r] = self.M.regret_sums[0][currentState][r]
            regretSum += self.M.regret_sums[0][currentState][r]

        # print("RegretSum: ", regretSum)
        #gamma = 0.25

        for r in sorted(self.M.pi[0][currentState].keys()):
            if regretSum > 0:
                self.M.pi[0][currentState][r] = (1.0 - gamma)*(max(regrets[r], 0.0) / regretSum) + (gamma / len(regrets))
            else:
                self.M.pi[0][currentState][r] = 1.0 / len(regrets)

            if regretSum > 0:
                self.M.pi_sums[0][currentState][r] += max(regrets[r], 0.0) / regretSum



        pi = {}
        pSum = 0.0
        for r in sorted(self.M.pi[0][currentState].keys()):
            pi[r] = self.M.pi[0][currentState][r]
            pSum = 1.0#+= pi[r]

        # print("PI: ", pi, " PiSum: ", sum(pi))
        piNorm = [p / sum(pi) for p in pi]
        piNorm = []
        for p in sorted(pi.keys()):
            piNorm.append(pi[p] / pSum)


        rAct = np.random.choice([0,1,2,3], p=piNorm)

        ###
        # if t % 6200 == 0:
        #
        #     for s in self.M.getStates():
        #         pi = {}
        #         pSum = 0.0
        #         for r in sorted(self.M.regret_sums[0][s].keys()):
        #             pi[r] = self.M.regret_sums[0][s][r]
        #             pSum += pi[r]
        #
        #         # print("PI: ", pi, " PiSum: ", sum(pi))
        #         piNorm = [p / sum(pi) for p in pi]
        #         piNorm = []
        #         if pSum <= 0:
        #             pSum = 1.0
        #         for p in sorted(pi.keys()):
        #             self.M.pi[0][s][p] = pi[p] / pSum


        return rAct
        # pi = [0, 0, 0, 0]
        #
        # regretSum = sum(regrets)
        # for r in range(4):
        #     if regretSum > 0:
        #         pi[r] = (1.0 - gamma)*(max(regrets[r], 0.0) / regretSum) + (gamma / len(regrets))
        #     else:
        #         pi[r] = 1.0 / len(regrets)

        # if t % 50 == 0:
        #     verbose = True
        # else:
        #     verbose = False
        #
        # rewardMin = -100.0
        # rewardMax = 200.0
        #
        # numActions = 4.0
        # n = 0
        # gamma = 0.15
        #
        # # get current weights for 0, 1, 2 ,3
        # weights = []
        # if verbose: print("Weights: ")
        # for w in sorted(self.M.regret_sums[n][currentState].keys()):
        #     if verbose: print("   ", w, ": ", self.M.regret_sums[n][currentState][w])
        #     weights.append(self.M.regret_sums[n][currentState][w] * gamma / float(numActions))
        #
        # if verbose: print("Final weights list: ", weights) # init: 1.0, 1.0, 1.0, 1.0
        #
        # # Get probDist from weights
        # pd = self.distr(weights, gamma)
        #
        # if verbose: print("ProbDist: ", pd) # init: 0.25, 0.25, 0.25, 0.25
        #
        # # Dist without randomness (for final pi sum calc)
        # piSUMDist = self.distr(weights, gamma=0.0)
        #
        # # set pi as probDistbution
        # for a in sorted(self.M.pi[n][currentState].keys()):
        #     if verbose: print("Setting pi of s: ", currentState, " a: ", a, " : ", pd[a])
        #     self.M.pi[n][currentState][a] = pd[a]
        #     self.M.pi_sums[n][currentState][a] += piSUMDist[a]
        #
        # # Select a random action from pd
        # # if sum(pd) != 1:
        # #     pd = [0.25, 0.25, 0.25, 0.25]
        # randomAction = np.random.choice([0,1,2,3], p=pd)
        # if verbose: print("Selected random action: ", randomAction)
        #
        # #Observe reward for action randomAction
        # rew = self.M.getReward(currentState, randomAction, 0, 0)
        # s_prime = self.M.getMove(currentState, randomAction)
        # Value = 0.0
        # for a_prime in self.M.getActions(s_prime, n):
        #     Value += self.M.pi[n][s_prime][a_prime] * self.M.Q[n][s_prime][a_prime]
        #
        # # This is the entire "reward", taken from paper
        # x = rew + self.gamma*Value
        #
        # # Scale the reward
        # scaledReward = (x - rewardMin) / (rewardMax - rewardMin)
        #
        # # Get expected reward
        # estimatedReward = scaledReward / pd[randomAction]
        #
        # # Add to running sum of rewards
        # self.M.runningRewards[0][currentState][randomAction] += estimatedReward
        # runningEstimatedReward = self.M.runningRewards[0][currentState][randomAction]
        #
        # # Find min reward
        # # currentRunningRewards = []
        # # for r in self.M.runningRewards[0][currentState].keys():
        # #     currentRunningRewards.append(self.M.runningRewards[0][currentState][r])
        # #
        # # minRunningReward = min(currentRunningRewards)
        # #
        # #
        # # for r in self.M.runningRewards[0][currentState].keys():
        # #     self.M.runningRewards[0][currentState][r] = self.M.runningRewards[0][currentState][r] - minRunningReward
        # #
        # # runningEstimatedReward = runningEstimatedReward - minRunningReward
        #
        #
        # # Find min weight
        # # currentRunningWeights = []
        # # for r in self.M.weights[0][currentState].keys():
        # #     currentRunningWeights.append(self.M.weights[0][currentState][r])
        # #
        # # minWeight = min(currentRunningWeights)
        # # if minWeight > 1:
        # #     for r in self.M.weights[0][currentState].keys():
        # #         self.M.weights[0][currentState][r] = self.M.weights[0][currentState][r] + minWeight
        #
        # # TODO: Cap of gap betwen max1 and max2
        #
        # # Set Weight for t+1
        # numActions = 4.0
        # if verbose: print(runningEstimatedReward)
        #
        # # self.M.weights[n][currentState][randomAction] += math.exp(runningEstimatedReward * gamma / float(numActions))
        # self.M.regret_sums[n][currentState][randomAction] += runningEstimatedReward #* gamma / float(numActions)
        # # if self.M.weights[n][currentState][randomAction] > 1000000:
        # #     self.M.weights[n][currentState][randomAction] = 1000000
        # #ooof had this: math.exp(math.exp(runningEstimatedReward * gamma / float(numActions)))

        #return randomAction



 # if maxWeight - maxWeight2 > thresh:
        #
        #     lp = []
        #     for w in weights:
        #         if w > maxWeight - maxWeight2:
        #             cc = gamma / len(weights)
        #         else:
        #             cc = 0.0 #(1.0 - gamma) * (w / theSum) + (gamma / len(weights))
        #         lp.append(cc)
        # else:

# get the max weight
        # #print("Weights: ", weights)
        # maxWeight = max(weights)
        # #print("Max Weight: ", maxWeight)
        # weigths_copy = weights.copy()
        # weigths_copy.remove(maxWeight)
        # maxWeight2 = max(weigths_copy)
        # #print(maxWeight, "  ", maxWeight2)
        #
        # #if maxWeight - maxWeight2 > 100000000.0:
        # #    gamma = 1.0
        #
        # thresh = 100000.0 #maxWeight - maxWeight2


################################################################################

# if WW is None:
        #     for s in self.M.getStates():
        #         for a in self.M.getActions(s, n):
        #             #self.M.Q[n][s][a] = self.M.Q_bu[n][s][a]
        #             self.M.Q[n][self.M.getStateRep(s)][a] = self.M.Q_bu[n][self.M.getStateRep(s)][a]
        #             self.M.QSums[n][self.M.getStateRep(s)][a] += self.M.Q[n][self.M.getStateRep(s)][a]
        #
        # else:
        #     for s in WW:
        #         for a in self.M.getActions(s, n):
        #             self.M.Q[n][self.M.getStateRep(s)][a] = self.M.Q_bu[n][self.M.getStateRep(s)][a]
        #             self.M.QSums[n][self.M.getStateRep(s)][a] += self.M.Q[n][self.M.getStateRep(s)][a]


##################################################################################

# # Update regret sums, pi, pi sums
        # if self.randomize == False:
        #     for n in range(self.M.N):
        #         for s in self.M.getStates():
        #             self.regretUpdate(n, s, t)
        # else:
        #     for n in range(self.M.N):
        #         for s in self.M.getStates(): #WW: #self.M.totalStatesLeft: #WW:
        #             if self.M.isTerminal(s): continue
        #             self.regretUpdate(n, s, t)


###################################################################################

# # For Tiger Game, randomize the MDP it sees
        # if randomized:
        #
        #     # Get the top root
        #     startStates = self.M.getNextStatesAndProbs(self.M.startState, None, 0)
        #
        #     totalStartStates = []
        #     totalStartStateProbs = []
        #
        #     # Pick Tiger on Left/Right based on probability set in M
        #     for nextState, nextStateProb, _ in startStates:
        #         totalStartStates.append(nextState)
        #         totalStartStateProbs.append(nextStateProb)
        #
        #     # Randomly pick TigerOnLeft/Right based on M.TLProb (1-M.TLProb = TRProb)
        #     currentState = np.random.choice(totalStartStates, p=totalStartStateProbs)
        #
        # # For GridWorld, set currentState to the startState
        # else:

#####################################################################################

# def lonr_value_iteration(self, iterations=-1, log=-1):
    #
    #
    #
    #     print("Starting training..")
    #     tt = 0
    #     for t in range(1, iterations+1):
    #
    #         if (t+1) % log == 0:
    #             print("Iteration: ", t+1, " alpha: ", self.alpha)
    #
    #
    #         self._lonr_value_iteration(t=t)
    #
    #         # No-op unless alphaDecay is not 1.0
    #         self.alpha *= self.alphaDecay
    #         self.alpha = max(0.0, self.alpha)
    #
    #     print("Finish Training")