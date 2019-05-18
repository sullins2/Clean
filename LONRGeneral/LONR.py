
import numpy as np


class MDP(object):

    def __init__(self, startState=None):

        # Check if these are used
        self.totalPlayers = 1
        self.totalStates = 1

        # Initialized here but must be defined/created in created class
        self.Q = None
        self.Q_bu = None
        self.QSums = None
        self.pi = None
        self.pi_sums = None
        self.regret_sums = None

        self.startState = startState

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


    # The following are only needed for O-LONR

    def getMove(self, s, a):
        """Used to get actual move in the environment for O-LONR
        """
        raise NotImplementedError("Please implement information_set method")

    def getStateRep(self, s):
        """ O-LONR use only

            For MDPs: simply returns state
            For TigerGame: Maps both to one (EX: TLL & TRL -> TLL) aka one set of Q Values
        """
        raise NotImplementedError("Please implement information_set method")

class LONR(object):

    # M: Markov game (MDP, markov game, tiger game)

    def __init__(self, M=None, gamma=0.99, alpha=0.99, epsilon=10, alphaDecay=1.0, DCFR=False):
        self.M = M
        #self.N = M.totalPlayers
        #self.S = M.totalStates

        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.alphaDecay = alphaDecay


        self.DCFR = DCFR

    def lonr_value_iteration(self, iterations=-1, log=-1):

        print("Starting training..")
        for t in range(1, iterations+1):

            if (t+1) % log == 0:
                print("Iteration: ", t+1, " alpha: ", self.alpha)

            self._lonr_value_iteration(t=t)
            #print("")
            self.alpha *= self.alphaDecay
            self.alpha = max(0.0, self.alpha)

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
                        self.M.Q_bu[n][s][a_current] = (1.0 - self.alpha)*self.M.Q[n][s][a_current] * + self.alpha*self.M.getReward(s, a_current, 0,0)
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

                    if self.DCFR:
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

                    if self.DCFR:
                        self.M.pi_sums[n][s][a] += self.M.pi[n][s][a] * gammaWeight
                    else:
                        self.M.pi_sums[n][s][a] += self.M.pi[n][s][a]


    def lonr_online(self, iterations=-1, log=-1, randomized=False):

        print("Starting training..")
        for t in range(1, iterations+1):

            if (t+1) % log == 0:
                print("Iteration: ", t+1, " alpha:", self.alpha)

            self._lonr_online(t=t, totalIterations=iterations, randomized=randomized)

            self.alpha *= self.alphaDecay
            self.alpha = max(0.0, self.alpha)

        print("Finish Training")


    def _lonr_online(self, t=0, totalIterations=-1, randomized=False):
        """

        :param t: current iteration
        :param totalIterations: total iterations
        :param randomized: Randomizes the start state.
                        True: For Tiger Game, to switch between TigerOnLeft, TigerOnRight
                        False: For GridWorld, only one start state.
        :return:
        """


        # For Tiger Game, randomize the MDP it sees
        if randomized:

            # Get the top root
            startStates = self.M.getNextStatesAndProbs(self.M.startState, None, 0)

            totalStartStates = []
            totalStartStateProbs = []

            # Pick Tiger on Left/Right based on probability set in M
            for nextState, nextStateProb, _ in startStates:
                totalStartStates.append(nextState)
                totalStartStateProbs.append(nextStateProb)

            # Randomly pick TigerOnLeft/Right based on M.TLProb (1-M.TLProb = TRProb)
            currentState = np.random.choice(totalStartStates, p=totalStartStateProbs)

        # For GridWorld, set currentState to the startState
        else:
            currentState = self.M.startState


        done = False
        n = 0  # One player

        # Episode loop - until terminal state is reached
        while done == False:

            # Terminal - Set Q as Reward (ALL terminals have one action - "exit" which receives reward)
            # Note:
            #       GridWorld Terminals have the same 4 actions.
            #           All get set here to the full reward
            #           pi for each action remains 1 / totalActions
            #       I intend to remove this part and include terminal rewards below in successors
            #       But for now, in the expectation taken below, the value is correct:
            #           For example:
            #               Q[term] = (1/4)Reward + (1/4)Reward + (1/4)Reward + (1/4)Reward = Reward
            #
            #       Also, this backup is split via self.alpha.
            #           For GridWorld, this does nothing (self.alpha = 1.0, removes the first part)
            #           For Tiger Game, this allows shared Q's to converge and not be overwritten
            if self.M.isTerminal(currentState) == True:
                for a in self.M.getActions(currentState, 0):
                    # Note, the reward is via the actual state, so there is no getStateRep()
                    self.M.Q_bu[n][self.M.getStateRep(currentState)][a] = (1.0 - self.alpha) * self.M.Q[n][self.M.getStateRep(currentState)][a] + self.alpha * self.M.getReward(currentState,a,a,a)
                done = True
                continue


            # totStates keeps track of which states need Qvalue copying
            #   as not all states need to be backed up, only the ones visited
            totStates = []
            totStates.append(currentState)

            # Loop through all actions
            for a in self.M.getActions(currentState, 0):

                # Get successor states
                succs = self.M.getNextStatesAndProbs(currentState, a, 0)

                Value = 0.0
                # Loop through each s', T(s' | s, a)
                #   - Last parameter in loop is reward, which eventually will be included
                #       -
                for s_prime, prob, _ in succs:

                    tempValue = 0.0
                    for a_prime in self.M.getActions(s_prime, 0):
                        tempValue += self.M.Q[n][self.M.getStateRep(s_prime)][a_prime] * self.M.pi[n][self.M.getStateRep(s_prime)][a_prime]
                    Value += prob * (self.M.getReward(currentState, a, 0,0) + self.gamma * tempValue)

                    if s_prime not in totStates:
                        totStates.append(s_prime)

                self.M.Q_bu[n][self.M.getStateRep(currentState)][a] = (1.0 - self.alpha) * self.M.Q[n][self.M.getStateRep(currentState)][a] + self.alpha * Value


            # Copy Q Values over for states visited
            for s in totStates:
                for a in self.M.getActions(s, 0):
                    self.M.Q[n][self.M.getStateRep(s)][a] = self.M.Q_bu[n][self.M.getStateRep(s)][a]
                    # self.Qsums[self.tigergame.getStateRep(s)][a] += self.Q_bu[self.tigergame.getStateRep(s)][a]


            # Don't update terminal states
            # if self.M.isTerminal(self.M.getStateRep(currentState)) == True:
            #     continue

            # Calculate regret - this variable name needs a better name
            target = 0.0

            for a in self.M.getActions(currentState, 0):
                target += self.M.Q[n][self.M.getStateRep(currentState)][a] * self.M.pi[n][self.M.getStateRep(currentState)][a]

            for a in self.M.getActions(currentState, 0):
                action_regret = self.M.Q[n][self.M.getStateRep(currentState)][a] - target

                RMPLUS = True
                if RMPLUS:
                    self.M.regret_sums[n][self.M.getStateRep(currentState)][a] = max(0.0, self.M.regret_sums[n][self.M.getStateRep(currentState)][a] + action_regret)
                else:
                    self.M.regret_sums[n][self.M.getStateRep(currentState)][a] += action_regret


            # Skip terminal states
            # if self.M.isTerminal(self.M.getStateRep(currentState)) == True:
            #     continue

            for a in self.M.getActions(self.M.getStateRep(currentState), 0):

                # Sum up total regret
                rgrt_sum = 0.0
                for k in self.M.regret_sums[n][self.M.getStateRep(currentState)].keys():
                    rgrt_sum += self.M.regret_sums[n][self.M.getStateRep(currentState)][k] if self.M.regret_sums[n][self.M.getStateRep(currentState)][k] > 0 else 0.0

                # Check if this is a "trick"
                # Check if this can go here or not
                if rgrt_sum > 0:
                    self.M.pi[n][self.M.getStateRep(currentState)][a] = (max(self.M.regret_sums[n][self.M.getStateRep(currentState)][a], 0.)) / rgrt_sum
                else:
                    self.M.pi[n][self.M.getStateRep(currentState)][a] = 1.0 / len(self.M.getActions(self.M.getStateRep(currentState),0))

                # Add to policy sum
                self.M.pi_sums[n][self.M.getStateRep(currentState)][a] += self.M.pi[n][self.M.getStateRep(currentState)][a]

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
                #print(totalActionsProbs)
                randomAction = np.random.choice(totalActions, p=totalActionsProbs)

            # randomAction picked, now simulate taking action
            #       GridWorld: This will handle non-determinism, if there is non-determinism
            #       TigerGame: This will either get the one next state OR
            #                       if action is LISTEN, it will return next state based on
            #                       observation accuracy aka (85/15) or (15/85), which is
            #                       equivalent to hearing a growl left or growl right
            nextPossStates = self.M.getNextStatesAndProbs(currentState, randomAction, 0)

            # If there is only one successor state, pick that
            if len(nextPossStates) == 1:
                # nextPossStates is list of lists
                # nextPossStates = [[next_state, prob, reward]]
                currentState = nextPossStates[0][0]

            # More than one possible successor, pick via probabilities
            #       GridWorld non-determ:
            #           3 states: 1-self.noise, (1-self.noise)/2, (1-self.noise)/2
            #               ex: 0.8, 0.1, 0.1 for randomAction, to the side, to the side
            #
            #       Tiger Game:
            #           Only happens when randomAction is Listen
            #               Return is based on which side tiger is on, and the obs accuracy
            else:
                nextStates = []
                nextStateProbs = []
                for ns, nsp,_ in nextPossStates:
                    nextStates.append(ns)
                    nextStateProbs.append(nsp)
                currentState = np.random.choice(nextStates, p=nextStateProbs)

