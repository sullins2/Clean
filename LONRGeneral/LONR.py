



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


    def getReward(self, s, a_current):
        """Returns reward(s, a, a_other)

        """
        raise NotImplementedError("Please implement information_set method")

    def getStates(self):
        """Return List of States

        """
        raise NotImplementedError("Please implement information_set method")

    def isTerminal(self, s):
        raise NotImplementedError("Please implement information_set method")

class LONR(object):

    # M: Markov game (MDP, markov game, tiger game)

    def __init__(self, M=None, alpha=0.99, gamma=0.99):
        self.M = M
        #self.N = M.totalPlayers
        #self.S = M.totalStates

        self.gamma = gamma
        self.alpha = alpha


    def lonr_value_iteration(self, iterations=-1, log=-1):

        print("Starting training..")
        for t in range(1, iterations+1):

            if (t+1) % log == 0:
                print("Iteration: ", t+1)

            self._lonr_value_iteration(t=t)

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
                        self.M.Q_bu[n][s][a_current] = self.M.getReward(s, a_current)
                        #print("Terminal: ", s)
                        continue

                    # Get next states and transition probs for MDP
                    # Get next states and other players policy for MG
                    succs = self.M.getNextStatesAndProbs(s, a_current, n)
                    #print("   succs: ", succs)


                    Value = 0.0

                    #Loop thru each next state and prob
                    for s_prime, prob, reward in succs:

                        tempValue = 0.0

                        # Loop thru actions in s_prime for player n
                        for a_current_prime in self.M.getActions(s_prime, n):
                            tempValue += self.M.Q[n][s_prime][a_current_prime] * self.M.pi[n][s_prime][a_current_prime]

                        Value += prob * (reward + self.gamma * tempValue)

                    self.M.Q_bu[n][s][a_current] = Value# self.alpha*self.M.Q[n][s][a_current] + (1.0 - self.alpha)*Value


        ####################################################################################################

        # Back up Q
        for n in range(self.M.N):
            for s in self.M.getStates():
                for a in self.M.getActions(s, n):
                    self.M.Q[n][s][a] = self.M.Q_bu[n][s][a]
                    self.M.QSums[n][s][a] += self.M.Q[n][s][a]

        # For each player
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

                    self.M.regret_sums[n][s][a] += action_regret

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

                    self.M.pi_sums[n][s][a] += self.M.pi[n][s][a]


    #def lonr_play(self, iterations=0, log=-1):
