

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
        self.gamma = 1.0
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


    def simulate_lonr(self, t=0):


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
            for a in range(self.numberOfActions):

                # Get successor states
                succs = self.gridWorld.getNextStatesAndProbs(s,a)

                Value = 0

                for s_prime, prob in succs:
                    temp = 0
                    for a_prime in range(self.numberOfActions):
                        temp += self.Q[s_prime][a_prime] * self.pi[s_prime][a_prime]
                    Value += prob * (self.gamma * temp)


                self.Q_bu[s][a] = Value + self.gridWorld.getReward(s)


        #print(self.Q[4])

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

                    self.regret_sums[s][a] += action_regret

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