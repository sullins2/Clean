

import numpy as np







class Agent:


    def __init__(self, gridworld):

        # Save GridWorld
        self.gridWorld = gridworld

        # Set number of total states
        self.numberOfStates = self.gridWorld.cols * self.gridWorld.rows

        # Set the number of total actions
        self.numberOfActions = 4

        # Parameters
        self.epsilon = 1.0
        self.gamma =  1.0
        self.alpha = 1.0

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


        for s in range(self.numberOfStates):
            for a in range(self.numberOfActions):
                self.Q[s][a] = -111111.0
                self.Q_bu[s][a] = -111111.0



    def train(self, iterations=0):
        print("Starting lonr agent training..")
        r_sums = []
        for i in range(1, iterations + 1):

            self.simulate_lonr(t=i)

            if True:
                if i % 1000 == 0 and i > 0:
                    print("Iteration:", i)
            # r_sums.append(r_sum)
        print("Finished LONR Training.")


    def train_olonr(self, totalIterations=100):
        for t in range(totalIterations):
            if (t + 1) % 1000 == 0:
                print("Iteration: ", (t + 1))
            self.lonr_episode(iters=t)

    def lonr_episode(self, startState=36, iters=0):

        currentState = startState
        #for s in range(self.numberOfStates):
        done = False

        print("")

        while done == False:

            print("Current state: ", currentState)
            # Terminal - Set Q as Reward (no actions in terminal states)
            if type(self.gridWorld.grid[currentState]) != str:
                for a in range(self.numberOfActions):
                    self.Q_bu[currentState][a] = self.gridWorld.getReward(currentState)
                done = True
                continue

            # Loop through all actions
            Value = 0
            for a in range(self.numberOfActions):

                # Get successor states
                succs = self.gridWorld.getNextStatesAndProbs(currentState, a)

                # if s == 36:
                #     print("SUCS: ", succs)
                Value = 0.0
                for s_prime, prob in succs:
                    tempValue = 0.0
                    for a_prime in range(self.numberOfActions):
                        tempValue += self.Q[s_prime][a_prime] * self.pi[s_prime][a_prime]
                    Value += prob * (self.gridWorld.getReward(currentState) + tempValue)

                self.Q_bu[currentState][a] = Value  # + self.gridWorld.getReward(s)

            randomAction = np.random.randint(0, 4)

            #randomAction = np.random.choice([0,1,2,3], p=self.pi[currentState])

            currentState = self.gridWorld.getMove(currentState, randomAction)

        for s in range(self.numberOfStates):
            for a in range(self.numberOfActions):
                self.Q[s][a] = self.Q_bu[s][a]
                self.Qsums[s][a] += self.Q[s][a]

        RM = True
        if RM:

            # Update regret sums
            for s in range(self.numberOfStates):

                if type(self.gridWorld.grid[s]) != str:
                    continue

                target = 0.0

                for a in range(self.numberOfActions):
                    target += self.Q[s][a] * self.pi[s][a]

                for a in range(self.numberOfActions):
                    action_regret = self.Q[s][a] - target

                    self.regret_sums[s][a] += action_regret  # max(0.0, self.regret_sums[s][a] + action_regret)

            # # Regret Match
            for s in range(self.numberOfStates):

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

        # #for s in range(self.numberOfStates):
        # currentState = startState
        #
        # if iters < 11:
        # #if np.random.randint(0, 100) < 20:
        #     currentState = np.random.randint(0, self.numberOfStates)
        #
        # done = False
        #
        # ts = []
        #
        # while done == False:
        #
        #     action = np.random.randint(0,4)
        #     #action = np.random.choice([0, 1, 2, 3], p=self.pi[currentState])
        #
        #     if type(self.gridWorld.grid[currentState]) != str:
        #         for a in range(self.numberOfActions):
        #             self.Q_bu[currentState][a] = self.gridWorld.getReward(currentState)
        #         done = True
        #         continue
        #
        #
        #     # Get successor states
        #     succs = self.gridWorld.getNextStatesAndProbs(currentState,action)
        #
        #     Value = 0
        #
        #     for s_prime, prob in succs:
        #         temp = 0
        #         for a_prime in range(self.numberOfActions):
        #             temp += self.Q[s_prime][a_prime] * self.pi[s_prime][a_prime]
        #         Value += prob * (self.gamma * temp)
        #
        #
        #     self.Q_bu[currentState][action] = Value + self.gridWorld.getReward(currentState)
        #
        #     ts.append(currentState)
        #     currentState = self.gridWorld.getMove(currentState, action)
        #
        # #print(self.Q[4])
        #
        #
        #     for ss in range(self.numberOfStates):
        #         for a in range(self.numberOfActions):
        #             self.Q[ss][a] = self.Q_bu[ss][a]
        #             self.Qsums[ss][a] += self.Q[ss][a]
        #
        #
        #
        # RM = True
        # if RM:
        #
        #     # # Update regret sums
        #     for s in range(self.numberOfStates):
        #
        #         if type(self.gridWorld.grid[s]) != str:
        #             continue
        #
        #         target = 0.0
        #
        #         for a in range(self.numberOfActions):
        #             target += self.Q[s][a] * self.pi[s][a]
        #
        #         gammaR = 2.0
        #         gammaWeight = pow(((iters + 1) / ((iters + 1) + 1)), gammaR)
        #
        #         for a in range(self.numberOfActions):
        #             action_regret = self.Q[s][a] - target
        #
        #             alphaR = 3.0 / 2.0  # accum pos regrets
        #             betaR = 0.0  # accum neg regrets
        #             gammaR = 2.0  # contribution to avg strategy
        #
        #             alphaW = pow(iters + 1, alphaR)
        #             alphaWeight = (alphaW / (alphaW + 1))
        #
        #             betaW = pow(iters + 1, betaR)
        #             betaWeight = (betaW / (betaW + 1))
        #
        #
        #
        #             if action_regret > 0:
        #                 action_regret *= alphaWeight
        #             else:
        #                 action_regret *= betaWeight
        #
        #             #self.regret_sums[s][a] += action_regret
        #             self.regret_sums[s][a] = max(0.0, self.regret_sums[s][a] + action_regret)
        #
        #
        #     # # Regret Match
        #     for s in range(self.numberOfStates):
        #
        #         if type(self.gridWorld.grid[s]) != str:
        #             continue
        #
        #         for a in range(self.numberOfActions):
        #             rgrt_sum = sum(filter(lambda x: x > 0, self.regret_sums[s]))
        #
        #             # Check if this is a "trick"
        #             # Check if this can go here or not
        #             if rgrt_sum > 0:
        #                 self.pi[s][a] = (max(self.regret_sums[s][a], 0.) / rgrt_sum)
        #             else:
        #                 self.pi[s][a] = 1.0 / len(self.regret_sums[s])
        #
        #             gammaR = 2.0
        #             gammaWeight = pow(((iters + 1) / ((iters + 1) + 1)), gammaR)
        #
        #             self.pi_sums[s][a] += self.pi[s][a] * gammaWeight
        #
        #         # Does this go above or below
        #             # self.pi_sums[s][a] += self.pi[s][a]



    def simulate_lonr(self, t=0):
        """ One complete value iteration

        """

        for s in range(self.numberOfStates):

            # Terminal - Set Q as Reward (no actions in terminal states)
            if type(self.gridWorld.grid[s]) != str:
                for a in range(self.numberOfActions):
                    self.Q_bu[s][a] = self.gridWorld.getReward(s)
                continue

            # Eventually, for walls
            # if grid[s] == '#':
            #     continue

            # Loop through all actions
            Value = 0
            for a in range(self.numberOfActions):

                # Get successor states
                succs = self.gridWorld.getNextStatesAndProbs(s,a)

                # if s == 36:
                #     print("SUCS: ", succs)
                Value = 0.0
                for s_prime, prob in succs:
                    tempValue = 0.0
                    for a_prime in range(self.numberOfActions):
                        tempValue += self.Q[s_prime][a_prime] * self.pi[s_prime][a_prime]
                    Value += prob * (self.gridWorld.getReward(s) + tempValue)

                self.Q_bu[s][a] = Value# + self.gridWorld.getReward(s)


        print("Q t + 1: ", max(self.Q_bu[36]))
        print("Q t    : ", self.Q[36])

        for s in range(self.numberOfStates):
            for a in range(self.numberOfActions):
                self.Q[s][a] = self.Q_bu[s][a]
                self.Qsums[s][a] += self.Q[s][a]




        RM = True
        if RM:

            # Update regret sums
            for s in range(self.numberOfStates):

                if type(self.gridWorld.grid[s]) != str:
                    continue

                target = 0.0

                for a in range(self.numberOfActions):
                    target += self.Q[s][a] * self.pi[s][a]


                for a in range(self.numberOfActions):
                    action_regret = self.Q[s][a] - target

                    self.regret_sums[s][a] += action_regret# max(0.0, self.regret_sums[s][a] + action_regret)

            # # Regret Match
            for s in range(self.numberOfStates):

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

