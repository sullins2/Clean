import numpy as np



class Agent:
    """Agent

    """

    def __init__(self, gridworld):

        # Save GridWorld
        self.gridWorld = gridworld

        # Set number of total states
        self.numberOfStates = self.gridWorld.cols * self.gridWorld.rows

        # Set the number of total actions
        self.numberOfActions = 4

        # Parameters
        self.epsilon = 10   # epsilon greedy: percent it will take random
        self.gamma = 1.0    # discount factor
        # self.alpha = 1.0  # No learning rate

        # living cost
        self.livingCost = -1.0

        # Q-Values
        self.Q = np.zeros((self.numberOfStates, self.numberOfActions))
        self.Q_bu = np.zeros((self.numberOfStates, self.numberOfActions))
        self.Qsums = np.zeros((self.numberOfStates, self.numberOfActions))

        # Policies
        self.pi = np.zeros((self.numberOfStates, self.numberOfActions))  # policy (if needed)
        self.pi_sums = np.zeros((self.numberOfStates, self.numberOfActions))

        # Regret Sums
        self.regret_sums = np.zeros((self.numberOfStates, self.numberOfActions))  # regret_sum (if needed)


        # Initialize policy to be uniform over total actions
        for s in range(self.numberOfStates):
            for a in range(self.numberOfActions):
                self.pi[s][a] = 1.0 / self.numberOfActions



    ################################################################
    # Value Iteration version of LONR
    ################################################################
    def train_lonr_value_iteration(self, iterations=100, log=-1):
        """Train LONR Value Iteration


        """

        print("Starting LONR agent training..")

        for t in range(1, iterations + 1):

            self._train_lonr_value_iteration(t=t)

            if (t + 1) % log == 0:
                print("Iteration:", t + 1)

        print("Finished LONR agent training.")

    def _train_lonr_value_iteration(self, t=0):
        """ One complete value iteration

        """

        for s in range(self.numberOfStates):

            # Terminal - Set Q as Reward (no actions in terminal states)
            # - alternatively, one action which is 'exit' which gets full reward
            if type(self.gridWorld.grid[s]) != str:
                for a in range(self.numberOfActions):
                    self.Q_bu[s][a] = self.gridWorld.getReward(s)
                continue

            # For walls , possibly (to conform to other setups)
            # if grid[s] == '#':
            #     continue

            # Loop through all actions in current state s
            for a in range(self.numberOfActions):

                # Get successor states
                succs = self.gridWorld.getNextStatesAndProbs(s,a)


                Value = 0.0
                for _, s_prime, prob in succs:
                    tempValue = 0.0
                    for a_prime in range(self.numberOfActions):
                        tempValue += self.Q[s_prime][a_prime] * self.pi[s_prime][a_prime]
                    Value += prob * (self.gridWorld.getReward(s) + self.gamma * tempValue)

                self.Q_bu[s][a] = Value

        # Copy over Q Values
        for s in range(self.numberOfStates):
            for a in range(self.numberOfActions):
                self.Q[s][a] = self.Q_bu[s][a]
                self.Qsums[s][a] += self.Q[s][a]


        # Regrets and policy updates

        # Update regret sums
        for s in range(self.numberOfStates):

            # Skip updating terminals
            if type(self.gridWorld.grid[s]) != str:
                continue

            target = 0.0

            for a in range(self.numberOfActions):
                target += self.Q[s][a] * self.pi[s][a]

            for a in range(self.numberOfActions):
                action_regret = self.Q[s][a] - target
                self.regret_sums[s][a] += action_regret# max(0.0, self.regret_sums[s][a] + action_regret)

        # Regret Match
        for s in range(self.numberOfStates):

            # Skip updating terminals
            if type(self.gridWorld.grid[s]) != str:
                continue

            for a in range(self.numberOfActions):
                rgrt_sum = sum(filter(lambda x: x > 0, self.regret_sums[s]))

                # Check if this is a "trick"
                # Check if this can go here or not
                if rgrt_sum > 0:
                    self.pi[s][a] = (max(self.regret_sums[s][a], 0.) / rgrt_sum)
                else:
                    self.pi[s][a] = 1.0 / len(self.regret_sums[s])

                # Does this go above or below
                self.pi_sums[s][a] += self.pi[s][a]


    ################################################################
    # Online version of LONR
    ################################################################
    def train_lonr_online(self, iterations=100, log=-1):
        """Train LONR online

        """
        for t in range(1, iterations + 1):

            #print("Qt: ", max(self.Q[3]), end='')
            self._train_lonr_online(iters=t)

            #print("MaxQ: ", max(self.Q[36]), "  Q*pi: ", np.dot(self.Q[36], self.pi[36]))
            #print("  Qt+1: ", max(self.Q[3]))

            if (t + 1) % log == 0:
                print("Iteration: ", (t + 1))


    def _train_lonr_online(self, startState=36, iters=0):
        """ One episode of O-LONR (online learning)

        """

        # Start episode at the prescribed start state
        currentState = startState

        # Will only update states actually visited
        statesVisited = []
        statesVisited.append(currentState)

        done = False

        # Loop until terminal state is reached
        while done == False:

            # Terminal - Set Q as Reward (ALL terminals have one action - "exit" which receives reward)
            if type(self.gridWorld.grid[currentState]) != str:
                for a in range(self.numberOfActions):
                    self.Q_bu[currentState][a] = self.gridWorld.getReward(currentState)
                done = True
                continue

            # Loop through all actions
            for a in range(self.numberOfActions):

                # Get successor states
                succs = self.gridWorld.getNextStatesAndProbs(currentState, a)

                Value = 0.0
                for _, s_prime, prob in succs:
                    tempValue = 0.0
                    for a_prime in range(self.numberOfActions):
                        tempValue += self.Q[s_prime][a_prime] * self.pi[s_prime][a_prime]
                    Value += prob * (self.gridWorld.getReward(currentState) + self.gamma*tempValue)

                self.Q_bu[currentState][a] = Value  # + self.gridWorld.getReward(s)

            # Epsilon greedy action selection

            if np.random.randint(0, 100) < self.epsilon:
                randomAction = np.random.randint(0, 4)
            else:
                randomAction = np.random.choice([0,1,2,3], p=self.pi[currentState])


            # Gets an actual noisy move if noise > 0
            # Else it will take the specified move.
            # In both cases, bumps into walls are checked
            currentState = self.gridWorld.getMove(currentState, randomAction)
            statesVisited.append(currentState)

        # Copy Q Values over for states visited
        for s in statesVisited:
            for a in range(self.numberOfActions):
                self.Q[s][a] = self.Q_bu[s][a]
                self.Qsums[s][a] += self.Q_bu[s][a]

        # for s in range(self.numberOfStates):
        #     for a in range(self.numberOfActions):
        #         self.Q[s][a] = self.Q_bu[s][a]
        #         self.Qsums[s][a] += self.Q_bu[s][a]



        # Update regret sums
        for s in statesVisited:

            # Don't update terminal states
            if type(self.gridWorld.grid[s]) != str:
                continue

            # Calculate regret - this variable name needs a better name
            target = 0.0

            for a in range(self.numberOfActions):
                target += self.Q[s][a] * self.pi[s][a]

            for a in range(self.numberOfActions):
                action_regret = self.Q[s][a] - target

                #self.regret_sums[s][a] += action_regret
                self.regret_sums[s][a] = max(0.0, self.regret_sums[s][a] + action_regret)

        # # Regret Match
        for s in statesVisited:

            # Skip terminal states
            if type(self.gridWorld.grid[s]) != str:
                continue

            for a in range(self.numberOfActions):
                rgrt_sum = sum(filter(lambda x: x > 0, self.regret_sums[s]))

                # Check if this is a "trick"
                # Check if this can go here or not
                if rgrt_sum > 0:
                    self.pi[s][a] = (max(self.regret_sums[s][a], 0.) / rgrt_sum)
                else:
                    self.pi[s][a] = 1.0 / len(self.regret_sums[s])

                # Add to policy sum
                self.pi_sums[s][a] += self.pi[s][a]


