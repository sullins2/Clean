# ###############################################################################
## MARKOV GAME LONR
###############################################################################
# PLAYER 1, STATE 1

import numpy as np


class Agent:

    def __init__(self):
        self.num_states = 1
        self.state_size = 1

        self.num_actions = 2
        self.n_actions = 2

        self.one_action = 1
        self.two_actions = 2

        # Player 1, State 1
        self.Qvalues11 = np.zeros((self.num_states, self.two_actions))
        self.QValue_backup11 = np.zeros((self.num_states, self.two_actions))
        self.pi11 = np.zeros((self.num_states, self.two_actions))

        self.regret_sums11 = np.zeros((self.num_states, self.two_actions))

        # Init uniform over actions
        for s in range(self.num_states):
            for a in range(self.two_actions):
                self.pi11[s][a] = 1.0 / float(self.two_actions)

        self.pi_sums11 = np.zeros((self.num_states, self.two_actions))

        # PLAYER 2, STATE 1
        state_size = 1
        n_actions = 1

        self.Qvalues21 = np.zeros((self.num_states, self.one_action))
        self.QValue_backup21 = np.zeros((self.num_states, self.one_action))
        self.pi21 = np.zeros((self.num_states, self.one_action))

        self.regret_sums21 = np.zeros((self.num_states, self.one_action))

        # Init uniform over actions
        for s in range(self.num_states):
            for a in range(self.one_action):
                self.pi21[s][a] = 1 / float(self.one_action)

        self.pi_sums21 = np.zeros((state_size, n_actions))

        # PLAYER 1, STATE 2
        state_size = 1
        n_actions12 = 1
        self.Qvalues12 = np.zeros((self.num_states, self.one_action))
        self.QValue_backup12 = np.zeros((self.num_states, self.one_action))
        self.pi12 = np.zeros((self.num_states, self.one_action))

        self.regret_sums12 = np.zeros((self.num_states, self.one_action))

        # Init uniform over actions
        for s in range(self.num_states):
            for a in range(self.one_action):
                self.pi12[s][a] = 1.0 / float(self.one_action)

        self.pi_sums12 = np.zeros((self.num_states, self.one_action))

        # state_size = 1
        # n_actions = 2

        # PLAYER 2, STATE 2
        state_size = 1
        n_actions12 = 2
        self.Qvalues22 = np.zeros((self.num_states, self.two_actions))
        self.QValue_backup22 = np.zeros((self.num_states, self.two_actions))
        self.pi22 = np.zeros((self.num_states, self.two_actions))

        self.regret_sums22 = np.zeros((self.num_states, self.two_actions))

        # Init uniform over actions
        for s in range(self.num_states):
            for a in range(self.two_actions):
                self.pi22[s][a] = 1.0 / float(self.two_actions)

        self.pi_sums22 = np.zeros((self.num_states, self.two_actions))

        state_size = 1
        n_actions = 2

        # ACTION 0 = K
        # ACTION 1 = S
        self.KEEP = 0
        self.SEND = 1
        self.NOOP = 2

        self.alpha = 1.0#0.9999
        self.gamma = 0.75

        LEPS = 50000
        reward = 0.0
        ss = 0


        self.pi12[0][0] = 1.0
        self.pi21[0][0] = 1.0

        alt = 1
        cc = 0


    def train_lonr(self, iterations=0, log=-1):
        print("Starting lonr agent training..")
        for i in range(1, iterations + 1):

            self._train_lonr(t=i)


            if (i+1) % log == 0:
                print("Iteration:", i + 1)

        print("Finished LONR Training.")

    def _train_lonr(self, t):

        #self.alpha = abs(float(t) - 100000.0 ) / 100000.0


        # PLAYER 1 UPDATES
        for s in range(self.state_size + 1):

            if s == 0:
                for a in range(self.two_actions):

                    if a == self.KEEP:

                        reward = 1.0
                        Value = 0.0
                        for action_ in range(self.two_actions):
                            Value += self.Qvalues11[0][action_] * self.pi11[0][action_]
                        #self.QValue_backup11[s][a] = self.Qvalues11[s][a] + self.alpha * (reward + self.gamma * Value - self.Qvalues11[s][a])
                        self.QValue_backup11[s][a] = reward + self.gamma * Value

                    elif a == self.SEND:
                        reward = 0.0

                        Value = 0.0
                        for action_ in range(1):
                            Value += self.Qvalues12[0][action_]# * self.pi21[0][action_] #pi2 should be 1.0 always, Q[NOOP] updated in state = 2
                        #self.QValue_backup11[s][a] = self.Qvalues11[s][a] + self.alpha * (reward + self.gamma * Value - self.Qvalues11[s][a])
                        self.QValue_backup11[s][a] = reward + self.gamma * Value



            elif s == 1:

                #action == NOOP
                a = 0

                reward = 0.0

                Value = 0.0 #(3.0 * (7.0 / 12.0))# + (7.0 * (7.0 / 12.0))
                for action_ in range(self.two_actions):
                    Value += self.Qvalues11[0][action_] * self.pi11[0][action_]
                #QValue_backup12[0][a] = ((7.0 / 12.0) * (3.0 + gamma * Qvalues12[0][a])) + ((5.0 / 12.0)*(gamma * Value))
                #self.QValue_backup12[0][a] = ((self.pi22[0][self.KEEP]) * (3.0 + self.gamma * self.Qvalues12[0][a])) + ((self.pi22[0][self.SEND]) *(self.gamma * Value))
                self.QValue_backup12[0][a] = ((self.pi22[0][self.KEEP]) * (3.0 + self.gamma * self.Qvalues12[0][a])) + ((self.pi22[0][self.SEND]) * (self.gamma * Value))

        #else:
        # PLAYER 2 UPDATES
        for s in range(self.state_size + 1):

            if s == 0:
                for a in range(self.two_actions):

                    if a == self.KEEP:

                        reward = 1.0
                        Value = 0.0
                        for action_ in range(self.two_actions):
                            Value += self.Qvalues22[0][action_] * self.pi22[0][action_]
                        #self.QValue_backup22[0][a] = self.Qvalues22[0][a] + self.alpha * (reward + self.gamma * Value - self.Qvalues22[0][a])
                        self.QValue_backup22[0][a] = reward + self.gamma * Value

                    elif a == self.SEND:

                        reward = 0.0

                        Value = 0.0
                        for action_ in range(1):
                            Value += self.Qvalues21[0][action_]# * self.pi12[0][action_] #pi2 should be 1.0 always, Q[NOOP] updated in state = 2
                        #self.QValue_backup22[0][a] = self.Qvalues22[0][a] + self.alpha * (reward + self.gamma * Value - self.Qvalues22[0][a])
                        self.QValue_backup22[0][a] = reward + self.gamma * Value



            elif s == 1:

                # action == NOOP
                a = 0

                reward = 0.0

                Value = 0.0  # (3.0 * (7.0 / 12.0))# + (7.0 * (7.0 / 12.0))
                for action_ in range(2):
                    Value += self.Qvalues22[0][action_] * self.pi22[0][action_]
                self.QValue_backup21[0][a] = (self.pi11[0][self.KEEP] * (0.0 + self.gamma * self.Qvalues21[0][a])) + ((self.pi11[0][self.SEND]) * (3.0 + self.gamma * Value))


        # DCFR STUFF
        alphaR = 3.0 / 2.0  # accum pos regrets
        betaR = 0.0         # accum neg regrets
        gammaR = 2.0        # contribution to avg strategy

        alphaW = pow(t+1, alphaR)
        alphaWeight = (alphaW / (alphaW + 1))

        betaW = pow(t+1, betaR)
        betaWeight = (betaW / (betaW + 1))

        gammaWeight = pow((t+1 / ((t+1) + 1)), gammaR)

        #if alt == 1:
        for s in range(self.state_size):
            for a in range(2):
                self.Qvalues11[s][a] = self.QValue_backup11[s][a]

        for s in range(self.state_size):
            for a in range(1):
                self.Qvalues12[s][a] = self.QValue_backup12[s][a]

        #else:
        for s in range(self.state_size):
            for a in range(2):
                self.Qvalues22[s][a] = self.QValue_backup22[s][a]

        for s in range(self.state_size):
            for a in range(1):
                self.Qvalues21[s][a] = self.QValue_backup21[s][a]

        #if alt == 1:
        # Update regret sums PLAYER 1
        for s in range(self.state_size):

            target = 0.0
            for a in range(self.two_actions):
                target += self.Qvalues11[s][a] * self.pi11[s][a]

            for a in range(self.two_actions):
                action_regret = self.Qvalues11[s][a] - target
                # if action_regret > 0:
                #     action_regret *= alphaWeight
                # else:
                #     action_regret *= betaWeight

                self.regret_sums11[s][a] += action_regret
                #self.regret_sums11[s][a] = max(0.0, self.regret_sums11[s][a] + action_regret)


        regretSum = sum(filter(lambda x: x > 0, self.regret_sums11[0]))

        for a in range(self.two_actions):
            self.pi11[0][a] = max(self.regret_sums11[0][a], 0.) / regretSum if regretSum > 0 else 1. / 2.0
            self.pi_sums11[0][a] += self.pi11[0][a] * gammaWeight



        # Update regret sums PLAYER 2
        for s in range(self.state_size):

            target = 0.0
            for a in range(self.two_actions):
                target += self.Qvalues22[s][a] * self.pi22[s][a]

            for a in range(self.two_actions):
                action_regret = self.Qvalues22[s][a] - target

                # WORKING
                # if action_regret > 0:
                #     action_regret *= alphaWeight
                # else:
                #     action_regret *= betaWeight

                self.regret_sums22[s][a] += action_regret
                #self.regret_sums22[s][a] = max(0.0, self.regret_sums22[s][a] + action_regret)

        regretSum2 = sum(filter(lambda x: x > 0, self.regret_sums22[0]))

        for a in range(self.two_actions):
            self.pi22[0][a] = max(self.regret_sums22[0][a], 0.) / regretSum2 if regretSum2 > 0 else 1.0 / 2.0
            self.pi_sums22[0][a] += self.pi22[0][a] * gammaWeight



lonrAgent = Agent()
lonrAgent.train_lonr(iterations=100000, log=10000)

print("PLAYER 1")
print("    QValues11:")
print("       ", lonrAgent.Qvalues11)

print(     "QValues12:")
print("       ", lonrAgent.Qvalues12)

print("     PI11")
print("       ", lonrAgent.pi11)

print("     PI SUMS11")
print("       ", lonrAgent.pi_sums11)

print("")
print("PLAYER 2")
print("    QValues21:")
print("       ", lonrAgent.Qvalues21)

print("QValues22:")
print("       ", lonrAgent.Qvalues22)

print("     PI22")
print("       ", lonrAgent.pi22)

print("     PI SUMS22")
print("       ", lonrAgent.pi_sums22)


# print("DD: ", pi_sums11.shape[1])
ns = 0.0
for x in range(lonrAgent.pi_sums11.shape[1]):
    #print(pi_sums11[0][x])
    ns += lonrAgent.pi_sums11[0][x]

for x in range(lonrAgent.pi_sums11.shape[1]):
    lonrAgent.pi_sums11[0][x] /= ns

ns = 0.0
for x in range(lonrAgent.pi_sums22.shape[1]):
    # print(pi_sums11[0][x])
    ns += lonrAgent.pi_sums22[0][x]

for x in range(lonrAgent.pi_sums22.shape[1]):
    lonrAgent.pi_sums22[0][x] /= ns


print("FINAL: ")
print("PI11SUMS: ", lonrAgent.pi_sums11)
print("PI22SUMS: ", lonrAgent.pi_sums22)