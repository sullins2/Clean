
import numpy as np


class MDP(object):

    def __init__(self):
        self.totalPlayers = 1
        self.totalStates = 1

        self.Q = None
        self.Q_bu = None
        self.QSums = None
        self.pi = None
        self.pi_sums = None
        self.regret_sums = None

    def getActions(self, s, n):
        """ Returns list of actions

        """
        raise NotImplementedError("Please implement information_set method")

    def getNextStatesAndProbs(self, s, a_current, n_current):
        """ Returns list of [ [next_state, prob, rew] ]

        """
        raise NotImplementedError("Please implement information_set method")


    def getReward(self, s, a_current, n, a_notN):
        """ s=state, a_current=actionOfPlayer n, n is current player, a_notN is action of other player
            Joint Actions required for Markov Games
            a_notN can be set to 0 for MDPs and ignored

        Returns reward(s, a, a_other)

        """
        raise NotImplementedError("Please implement information_set method")

    def getStates(self):
        """Return List of States

        """
        raise NotImplementedError("Please implement information_set method")

    def isTerminal(self, s):
        raise NotImplementedError("Please implement information_set method")


    def getMove(self, s, a):
        """Used to get actual move in the environment for O-LONR
        """
        raise NotImplementedError("Please implement information_set method")

class LONR(object):

    # M: Markov game (MDP, markov game, tiger game)

    def __init__(self, M=None, gamma=0.99, alpha=0.99, epsilon=10):
        self.M = M
        #self.N = M.totalPlayers
        #self.S = M.totalStates

        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon


    def lonr_value_iteration(self, iterations=-1, log=-1):

        print("Starting training..")
        for t in range(1, iterations+1):

            if (t+1) % log == 0:
                print("Iteration: ", t+1, " alpha: ", self.alpha)

            self._lonr_value_iteration(t=t)
            #print("")
            # self.alpha *= 0.9999
            # self.alpha = max(0.0, self.alpha)

        print("Finish Training")

    def _lonr_value_iteration(self, t):

        # Loop through all players
        for n in range(self.M.N):
            #print("N: ", n)

            # Loop through all states
            for s in self.M.getStates():
                #print(" s: ", s)

                # Loop through actions of current player n
                for a_current in self.M.getActions(s, n):
                    #print("  a_current: ", a_current)

                    if self.M.isTerminal(s):
                        # This is here because terminals have actions like "exit"
                        # to collect the reward. To make the code easier, the terminals
                        # have the same actions as everywhere else, but each action is an "exit"
                        #
                        # This was a remnant of when I was using numpy arrays and couldn't set the size
                        # for one index different than the rest easily (Q[totalStates][totalActions] for
                        # instance, but Q[terminals][oneExitAction]
                        #
                        # But now the MDP class handles that, and if I change them all to dicts/maps
                        # and not numpy arrays, I wont need this special case here.
                        #
                        # This works out below if/when an expectation is taken because the initial
                        # uniform policy never changes so the expected value = reward
                        #
                        # I might possibly change it so that instead of setting it here,
                        # M.getActions(terminal) returns one exit action
                        self.M.Q_bu[n][s][a_current] = self.M.getReward(s, a_current)
                        #print("Terminal: ", s)
                        continue

                    # Get next states and transition probs for MDP
                    # Get next states and other players policy for MG
                    succs = self.M.getNextStatesAndProbs(s, a_current, n)
                    #print("s: ", s, " a:", a_current, "   succs: ", succs)


                    Value = 0.0

                    #Loop thru each next state and prob
                    for s_prime, prob, reward in succs:
                        #print(" - s_prime: ", s_prime, "  prob: ", prob, "  reward: ", reward)
                        tempValue = 0.0

                        # Loop thru actions in s_prime for player n
                        for a_current_prime in self.M.getActions(s_prime, n):
                            tempValue += self.M.Q[n][s_prime][a_current_prime] * self.M.pi[n][s_prime][a_current_prime]

                        Value += prob * (reward + self.gamma * tempValue)

                    self.M.Q_bu[n][s][a_current] = (1.0 - self.alpha)*self.M.Q[n][s][a_current] + (self.alpha)*Value


        ####################################################################################################

        # Back up Q
        for n in range(self.M.N):
            for s in self.M.getStates():
                for a in self.M.getActions(s, n):
                    self.M.Q[n][s][a] = self.M.Q_bu[n][s][a]
                    self.M.QSums[n][s][a] += self.M.Q[n][s][a]

        # Regrets / policy updates, etc
        # For each player

        iters = t + 1
        alphaR = 3.0 / 2.0  # accum pos regrets
        betaR = 0.0         # accum neg regrets
        gammaR = 2.0        # contribution to avg strategy

        alphaW = pow(iters, alphaR)
        alphaWeight = (alphaW / (alphaW + 1))

        betaW = pow(iters, betaR)
        betaWeight = (betaW / (betaW + 1))

        gammaWeight = pow((iters / (iters + 1)), gammaR)

        for n in range(self.M.N):

            # For each state
            for s in self.M.getStates():

                # Skip terminals
                if self.M.isTerminal(s): continue

                target = 0.0
                for a in self.M.getActions(s, n):
                    target += self.M.Q[n][s][a] * self.M.pi[n][s][a]

                for a in self.M.getActions(s, n):
                    action_regret = self.M.Q[n][s][a] - target

                    if action_regret > 0:
                        action_regret *= alphaWeight
                    else:
                        action_regret *= betaWeight

                    self.M.regret_sums[n][s][a] += action_regret
                    #self.M.regret_sums[n][s][a] = max(0.0, self.M.regret_sums[n][s][a] + action_regret)

            for s in self.M.getStates():

                # Skip terminals
                if self.M.isTerminal(s): continue

                rgrt_sum = 0.0
                for a in self.M.getActions(s, n):
                    rgrt_sum += max(0.0, self.M.regret_sums[n][s][a])

                for a in self.M.getActions(s, n):
                    if rgrt_sum > 0:
                        self.M.pi[n][s][a] = (max(self.M.regret_sums[n][s][a], 0.0) / rgrt_sum)
                    else:
                        self.M.pi[n][s][a] = 1. / len(self.M.getActions(s, n))

                    self.M.pi_sums[n][s][a] += self.M.pi[n][s][a] * gammaWeight


    def lonr_online(self, iterations=-1, log=-1):

        print("Starting training..")
        for t in range(1, iterations+1):

            if (t+1) % log == 0:
                print("Iteration: ", t+1)

            self._lonr_online(t=t)

        print("Finish Training")


    def _lonr_online(self, startState=36, t=0, totalIterations=-1):
        # Start episode at the prescribed start state
        currentState = startState

        # Will only update states actually visited
        statesVisited = []
        #statesVisited.append(currentState)

        done = False

        n = 0
        #print("")
        # Loop until terminal state is reached
        while done == False:

            if currentState not in statesVisited:
                statesVisited.append(currentState)
            #print(currentState)
            # Terminal - Set Q as Reward (ALL terminals have one action - "exit" which receives reward)
            if self.M.isTerminal(currentState):
                for a in self.M.getActions(currentState, 0):
                    self.M.Q_bu[n][currentState][a] = self.M.getReward(currentState, 0, 0 ,0)
                done = True
                continue

            # Loop through all actions
            for a in self.M.getActions(currentState,0):

                # Get successor states
                succs = self.M.getNextStatesAndProbs(currentState, a, 0)

                Value = 0.0
                for s_prime, prob, rew in succs:
                    if s_prime not in statesVisited:
                        statesVisited.append(s_prime)
                    tempValue = 0.0
                    for a_prime in self.M.getActions(s_prime, 0):

                        tempValue += self.M.Q[n][s_prime][a_prime] * self.M.pi[n][s_prime][a_prime]
                    Value += prob * (self.M.getReward(currentState,0,0,0) + self.gamma * tempValue)

                DECAY_ALPHA = False  # For Q Value convergence
                if DECAY_ALPHA:
                    # self.alpha = (float(totalIterations) - float(iters)) / float(totalIterations)
                    self.M.Q_bu[n][currentState][a] = (1.0 - self.alpha) * self.M.Q[n][currentState][a] + self.alpha * Value
                else:
                    self.M.Q_bu[n][currentState][a] = Value





            # self.epsilon = (((float(totalIterations) / 10.0) - float(iters)) / (float(totalIterations) / 10.0)) * 20.0
            # self.epsilon = max(0.0, self.epsilon)
            # eppRand = float(np.random.randint(0, 100))
            # self.alpha = (float(totalIterations) - float(iters)) / float(totalIterations)

            #print(statesVisited)
            # Copy Q Values over for states visited
            # States visited are current state and all s_primes
            for s in statesVisited: #self.M.getStates():
                for a in self.M.getActions(currentState, 0):
                    self.M.QSums[n][s][a] += self.M.Q_bu[n][s][a]
                    self.M.Q[n][s][a] = self.M.Q_bu[n][s][a]



            # Update regret sums
            for s in statesVisited: # THIS IS ONLY ONE STATE

                # Don't update terminal states
                if self.M.isTerminal(s): # THIS IS ONLY ONE STATE
                    continue

                # Calculate regret - this variable name needs a better name
                target = 0.0

                for a in self.M.getActions(s, 0):
                    target += self.M.Q[n][s][a] * self.M.pi[n][s][a]

                for a in self.M.getActions(s, 0):
                    action_regret = self.M.Q[n][s][a] - target

                    # RM+ works, reg dont know
                    #self.M.regret_sums[n][s][a] += action_regret
                    self.M.regret_sums[n][s][a] = max(0.0, self.M.regret_sums[n][s][a] + action_regret)

            # # Regret Match
            for s in statesVisited:

                # Skip terminal states
                if self.M.isTerminal(s):  # THIS IS ONLY ONE STATE
                    continue

                for a in self.M.getActions(s, 0):
                    rgrt_sum = sum(filter(lambda x: x > 0, self.M.regret_sums[n][s]))

                    # Check if this is a "trick"
                    # Check if this can go here or not
                    if rgrt_sum > 0:
                        self.M.pi[n][s][a] = (max(self.M.regret_sums[n][s][a], 0.) / rgrt_sum)
                    else:
                        self.M.pi[n][s][a] = 1.0 / len(self.M.regret_sums[n][s])

                    # Add to policy sum
                    self.M.pi_sums[n][s][a] += self.M.pi[n][s][a]


            # Epsilon greedy action selection
            # EPS_GREEDY = True
            # randomAction = None
            # if EPS_GREEDY:
            if np.random.randint(0, 100) < self.epsilon:
                randomAction = np.random.randint(0, 4)
            else:
                # UP, RIGHT, DOWN, LEFT
                randomAction = np.random.choice([0, 1, 2, 3], p=self.M.pi[n][currentState])

            #else:
               #previously experiment with epsilon decay

            statesVisited = []
            currentState = self.M.getMove(currentState, randomAction)